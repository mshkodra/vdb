#include "sift.h"

#include "bench.h"

#include "brute_index.h"
#include "distance.h"
#include "hnsw_index.h"
#include "ivf_index.h"

#include <sys/stat.h>  // mkdir for the snapshot cache dir

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "serialize.h"  // put<>/Reader — index snapshots reuse these primitives

namespace {

// ============================================================================
//  EXPERIMENT GRID  —  this block is yours to design (see CLAUDE.md).
//  Everything below it is plumbing. Change these; predict before running.
//
//  Cost warning (predict these before you hit run):
//   * HNSW build is one serial insert of 1M vectors at efConstruction=200 —
//     the dominant HNSW cost (minutes).
//   * IVF build runs Lloyd k-means over ALL 1M base vectors (the current
//     train() clusters and populates from the same array). Cost is
//     O(N * nlist * dim * iters) = 1e6 * 1024 * 128 * ITERS. That is the
//     single biggest wall-clock item in this run. Tune IVF_KMEANS_ITERS to
//     trade build time against centroid quality (=> recall).
// ============================================================================
constexpr size_t DIM = 128;  // SIFT descriptors are 128-d

// HNSW. search width at layer 0 is max(ef, K), so recall@100 only *moves* once
// ef >= 100 — hence a separate ef grid per K, each starting at ef >= K.
constexpr size_t HNSW_M   = 16;
constexpr size_t HNSW_EFC = 200;  // efConstruction (build-time candidate width)
const std::vector<size_t> HNSW_EF_K10  = {10, 16, 32, 64, 128, 256};
const std::vector<size_t> HNSW_EF_K100 = {100, 128, 200, 320, 512, 800};

// IVF. nlist ~ sqrt(N) ~= 1000; nprobe is the recall/speed knob. nprobe=256 was
// dropped: at a quarter of all cells it collapses to ~brute cost for a sliver of
// recall, and its recall pass over the query set dominated re-run time.
constexpr size_t IVF_NLIST        = 1024;
constexpr size_t IVF_KMEANS_ITERS = 15;
const std::vector<size_t> IVF_NPROBE = {1, 4, 8, 16, 32, 64, 128};

// Recall Ks reported for every method (ground truth ships top-100).
const std::vector<size_t> RECALL_K = {10, 100};

// Evaluation budget. Recall is a per-query average, so a 2k-query subset is a
// statistically stable estimate (±<0.01) at a fraction of the cost — the full
// 10k pass at high nprobe was the re-run bottleneck. Latency uses its own subset
// with min-of-reps to cut noise.
constexpr size_t Q_RECALL   = 2000;   // subset for recall (stable average)
constexpr size_t Q_TIMING   = 1000;   // subset for latency/QPS
constexpr int    TIMING_REPS = 5;
// Brute at N=1M is ~tens of ms/query; anchor it on a small query subset only.
constexpr size_t Q_BRUTE    = 200;
constexpr int    BRUTE_REPS = 3;
// ============================================================================

// ---- index snapshot cache ---------------------------------------------------
// Serialize the *finished* IVF/HNSW structure so a re-run loads it in seconds
// instead of re-training/re-building for ~20 min. This is snapshot, not WAL:
// replaying a WAL of 1M INSERTs would just redo the build. A small header carries
// the build params; a mismatch (or a --rebuild) forces a fresh build. The build
// time is stored too, so cached runs still report the true build cost.

constexpr uint32_t CACHE_MAGIC   = 0x53494654;  // "SIFT"
constexpr uint32_t CACHE_VERSION = 1;

void save_index_snapshot(const std::string& path, double build_ms,
                         const std::vector<uint64_t>& params,
                         const vdb::Index& idx) {
    std::vector<uint8_t> buf;
    vdb::put<uint32_t>(buf, CACHE_MAGIC);
    vdb::put<uint32_t>(buf, CACHE_VERSION);
    vdb::put<double>(buf, build_ms);
    vdb::put<uint32_t>(buf, static_cast<uint32_t>(params.size()));
    for (uint64_t p : params) vdb::put<uint64_t>(buf, p);
    idx.serialize(buf);

    const std::string tmp = path + ".tmp";
    std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
    if (!f) { std::printf("  !! cannot write cache %s\n", tmp.c_str()); return; }
    f.write(reinterpret_cast<const char*>(buf.data()),
            static_cast<std::streamsize>(buf.size()));
    f.close();
    std::rename(tmp.c_str(), path.c_str());  // replace atomically
}

// Returns true iff the file exists, header + params match, and deserialize
// succeeds; on any mismatch/corruption returns false so the caller rebuilds.
bool load_index_snapshot(const std::string& path,
                         const std::vector<uint64_t>& params, vdb::Index& idx,
                         double& build_ms_out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    std::vector<uint8_t> buf((std::istreambuf_iterator<char>(f)),
                             std::istreambuf_iterator<char>());
    vdb::Reader r(buf.data(), buf.size());
    try {
        if (r.get<uint32_t>() != CACHE_MAGIC)   return false;
        if (r.get<uint32_t>() != CACHE_VERSION) return false;
        build_ms_out = r.get<double>();
        if (r.get<uint32_t>() != params.size()) return false;
        for (uint64_t exp : params)
            if (r.get<uint64_t>() != exp) return false;
        idx.deserialize(r);
    } catch (const std::exception& e) {
        std::printf("  !! cache load failed (%s) — rebuilding\n", e.what());
        return false;
    }
    return true;
}

// ---- .fvecs / .ivecs readers ------------------------------------------------
// Format: each record is [int32 dim][dim payload]. No file header, no count;
// N = filesize / (4 + dim*payload_bytes). The dim prefix repeats per record.

std::vector<float> read_fvecs(const std::string& path, size_t& n_out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "!! cannot open %s\n", path.c_str()); std::exit(1); }
    f.seekg(0, std::ios::end);
    const std::streamoff bytes = f.tellg();
    f.seekg(0, std::ios::beg);

    int32_t d = 0;
    f.read(reinterpret_cast<char*>(&d), 4);
    if (d <= 0) { std::fprintf(stderr, "!! bad dim in %s\n", path.c_str()); std::exit(1); }
    const size_t dim = static_cast<size_t>(d);
    const size_t rec = 4 + dim * 4;
    const size_t n   = static_cast<size_t>(bytes) / rec;

    std::vector<float> out(n * dim);
    f.seekg(0, std::ios::beg);
    for (size_t i = 0; i < n; ++i) {
        int32_t dd = 0;
        f.read(reinterpret_cast<char*>(&dd), 4);  // skip the repeated dim prefix
        f.read(reinterpret_cast<char*>(out.data() + i * dim),
               static_cast<std::streamsize>(dim) * 4);
    }
    n_out = n;
    return out;
}

std::vector<int32_t> read_ivecs(const std::string& path, size_t& n_out, size_t& dim_out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "!! cannot open %s\n", path.c_str()); std::exit(1); }
    f.seekg(0, std::ios::end);
    const std::streamoff bytes = f.tellg();
    f.seekg(0, std::ios::beg);

    int32_t d = 0;
    f.read(reinterpret_cast<char*>(&d), 4);
    const size_t dim = static_cast<size_t>(d);
    const size_t rec = 4 + dim * 4;
    const size_t n   = static_cast<size_t>(bytes) / rec;

    std::vector<int32_t> out(n * dim);
    f.seekg(0, std::ios::beg);
    for (size_t i = 0; i < n; ++i) {
        int32_t dd = 0;
        f.read(reinterpret_cast<char*>(&dd), 4);
        f.read(reinterpret_cast<char*>(out.data() + i * dim),
               static_cast<std::streamsize>(dim) * 4);
    }
    n_out = n;
    dim_out = dim;
    return out;
}

// ---- evaluation helpers -----------------------------------------------------

// recall@K against the shipped ground truth. gt is row-major [n_query x gt_dim];
// row q's first K entries are the true K nearest base-vector ids. Because we
// insert base vectors in file order, InternalId == base row == gt value.
template <typename Index>
double recall_at_k(const Index& idx, const int32_t* gt, size_t gt_dim,
                   const float* queries, size_t Q, size_t K) {
    size_t hits = 0;
    std::unordered_set<InternalId> truth;
    truth.reserve(K * 2);
    for (size_t q = 0; q < Q; ++q) {
        truth.clear();
        for (size_t j = 0; j < K; ++j)
            truth.insert(static_cast<InternalId>(gt[q * gt_dim + j]));
        for (auto& [id, d] : idx.search(queries + q * DIM, K))
            if (truth.count(id)) ++hits;
    }
    return static_cast<double>(hits) / static_cast<double>(Q * K);
}

double percentile(const std::vector<double>& sorted, double p) {
    if (sorted.empty()) return 0.0;
    size_t idx = static_cast<size_t>(std::ceil(p / 100.0 * sorted.size()));
    if (idx > 0) --idx;
    return sorted[std::min(idx, sorted.size() - 1)];
}

struct QStats {
    double qps, mean_us, p50_us, p95_us, p99_us;
};

// Per-query latency = min over `reps` (warm-cache, noise floor); percentiles
// across the query subset; QPS from the mean of the per-query mins.
template <typename Index>
QStats time_queries(const Index& idx, const float* queries, size_t Q, size_t K,
                    int reps) {
    for (size_t q = 0; q < Q; ++q)  // warm-up, discarded
        bench::do_not_optimize(idx.search(queries + q * DIM, K).size());

    std::vector<double> per_us(Q);
    for (size_t q = 0; q < Q; ++q) {
        // Finite sentinel, not infinity(): -ffast-math implies -ffinite-math-only,
        // which makes infinities UB and lets the compiler miscompile this min.
        double best = std::numeric_limits<double>::max();
        for (int r = 0; r < reps; ++r) {
            auto t0 = bench::clk::now();
            auto res = idx.search(queries + q * DIM, K);
            auto t1 = bench::clk::now();
            bench::do_not_optimize(res.size());
            best = std::min(best,
                std::chrono::duration<double, std::micro>(t1 - t0).count());
        }
        per_us[q] = best;
    }
    std::sort(per_us.begin(), per_us.end());
    double sum = 0.0;
    for (double x : per_us) sum += x;
    const double mean = sum / Q;
    return {1e6 / mean, mean, percentile(per_us, 50), percentile(per_us, 95),
            percentile(per_us, 99)};
}

double build_ms_of(const std::function<void()>& build) {
    auto t0 = bench::clk::now();
    build();
    auto t1 = bench::clk::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// ---- CSV emission -----------------------------------------------------------

struct Row {
    std::string method, knob;
    long        k, knob_val;
    double      build_ms, recall, qps, mean_us, p50_us, p95_us, p99_us;
};

void write_csv(const std::vector<Row>& rows, const std::string& path,
               const std::string& variant, bool append) {
    std::FILE* f = std::fopen(path.c_str(), append ? "a" : "w");
    if (!f) {
        std::printf("!! could not open %s (mkdir -p docs/results first)\n", path.c_str());
        return;
    }
    if (!append)
        std::fprintf(f, "dataset,method,k,knob,knob_val,build_ms,recall,qps,"
                        "mean_us,p50_us,p95_us,p99_us,variant\n");
    for (const auto& r : rows)
        std::fprintf(f, "sift1m,%s,%ld,%s,%ld,%.1f,%.4f,%.1f,%.3f,%.3f,%.3f,%.3f,%s\n",
                     r.method.c_str(), r.k, r.knob.c_str(), r.knob_val, r.build_ms,
                     r.recall, r.qps, r.mean_us, r.p50_us, r.p95_us, r.p99_us,
                     variant.c_str());
    std::fclose(f);
    std::printf("\nwrote %zu rows -> %s  (variant=%s)\n",
                rows.size(), path.c_str(), variant.c_str());
}

}  // namespace

void run_sift(const char* data_dir, const char* methods, const char* label_arg) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);  // live progress even when piped
    const std::string dir = data_dir;
    const std::string which = methods ? methods : "all";
    // `which` is "all" or a comma list of {brute,ivf,hnsw}, optionally with the
    // token "rebuild" to bypass the snapshot cache and force a fresh build.
    const bool run_all       = which.find("all") != std::string::npos;
    const bool force_rebuild = which.find("rebuild") != std::string::npos;
    auto want = [&](const char* m) {
        return run_all || which.find(m) != std::string::npos;
    };
    // A subset run appends to the CSV so methods can be filled in independently.
    const bool append = !run_all;

    // `label` tags a run so before/after variants live in separate CSVs and carry
    // a `variant` column (e.g. "stdfn" vs "inlined"). Empty label => default file.
    const std::string label   = (label_arg && *label_arg) ? label_arg : "";
    const std::string variant = label.empty() ? "run" : label;
    const std::string csv_path = label.empty()
        ? std::string("docs/results/sift1m.csv")
        : "docs/results/sift1m_" + label + ".csv";

    const std::string cache_dir = dir + "/cache";
    ::mkdir(cache_dir.c_str(), 0755);  // ignore EEXIST

    // Indexes are templated on the distance functor (vdb::L2) so the inner loop
    // inlines and auto-vectorizes — no std::function on the hot path.

    std::printf("\n######## SIFT1M: recall-vs-QPS on real data ########\n");
    std::printf("loading from %s ...\n", dir.c_str());

    size_t n_base = 0, n_query = 0, n_gt = 0, gt_dim = 0;
    auto base    = read_fvecs(dir + "/sift_base.fvecs", n_base);
    auto queries = read_fvecs(dir + "/sift_query.fvecs", n_query);
    auto gt      = read_ivecs(dir + "/sift_groundtruth.ivecs", n_gt, gt_dim);
    std::printf("  base=%zu  query=%zu  gt=%zu x %zu  (dim=%zu)\n",
                n_base, n_query, n_gt, gt_dim, DIM);

    const size_t q_recall = std::min(Q_RECALL, n_query);
    const size_t q_timing = std::min(Q_TIMING, n_query);
    const size_t q_brute  = std::min(Q_BRUTE,  n_query);
    std::vector<Row> rows;

    // ---- brute anchor: recall ~1.0 (validates gt alignment + metric), and the
    //      QPS floor every ANN curve is measured against. Timed on a subset. ----
    if (want("brute")) {
        vdb::BruteIndex<vdb::L2> brute(DIM);
        double build = build_ms_of([&] {
            for (size_t i = 0; i < n_base; ++i) brute.add(base.data() + i * DIM);
        });
        std::printf("\n[brute]  build %.0f ms (N=%zu)\n", build, n_base);
        for (size_t K : RECALL_K) {
            double rec = recall_at_k(brute, gt.data(), gt_dim, queries.data(), q_brute, K);
            auto ts = time_queries(brute, queries.data(), q_brute, K, BRUTE_REPS);
            rows.push_back({"brute", "none", (long)K, 0, build, rec, ts.qps,
                            ts.mean_us, ts.p50_us, ts.p95_us, ts.p99_us});
            std::printf("  K=%-3zu recall@%zu %.4f | %8.1f QPS | p95 %.0f us "
                        "(subset of %zu queries)\n",
                        K, K, rec, ts.qps, ts.p95_us, q_brute);
        }
    }

    // ---- IVF: one trained index (k-means over all 1M), sweep nprobe. ----
    if (want("ivf")) {
        vdb::IVFIndex<vdb::L2> ivf({DIM, IVF_NLIST, IVF_NPROBE.front(), IVF_KMEANS_ITERS});
        const std::string cache = cache_dir + "/ivf.snap";
        const std::vector<uint64_t> params = {DIM, n_base, IVF_NLIST, IVF_KMEANS_ITERS};
        double build = 0.0;
        if (!force_rebuild && load_index_snapshot(cache, params, ivf, build)) {
            std::printf("\n[IVF]  loaded cache %s (nlist=%zu, orig build %.0f ms)\n",
                        cache.c_str(), IVF_NLIST, build);
        } else {
            std::printf("\n[IVF]  training (nlist=%zu, iters=%zu) over %zu vectors — "
                        "this is the slow one...\n", IVF_NLIST, IVF_KMEANS_ITERS, n_base);
            build = build_ms_of([&] { ivf.train(base.data(), n_base); });
            std::printf("  build %.0f ms\n", build);
            save_index_snapshot(cache, build, params, ivf);
            std::printf("  cached -> %s\n", cache.c_str());
        }
        for (size_t np : IVF_NPROBE) {
            ivf.set_nprobe(np);
            for (size_t K : RECALL_K) {
                double rec = recall_at_k(ivf, gt.data(), gt_dim, queries.data(), q_recall, K);
                auto ts = time_queries(ivf, queries.data(), q_timing, K, TIMING_REPS);
                rows.push_back({"ivf", "nprobe", (long)K, (long)np, build, rec, ts.qps,
                                ts.mean_us, ts.p50_us, ts.p95_us, ts.p99_us});
                std::printf("  nprobe=%-4zu K=%-3zu recall@%zu %.4f | %8.1f QPS | "
                            "p95 %.0f us\n", np, K, K, rec, ts.qps, ts.p95_us);
            }
        }
    }

    // ---- HNSW: one built graph, separate ef sweep per K (width = max(ef,K)). ----
    if (want("hnsw")) {
        vdb::HNSWIndex<vdb::L2> hnsw(
            {DIM, HNSW_M, HNSW_M, 2 * HNSW_M, HNSW_EFC, 0.0f});
        const std::string cache = cache_dir + "/hnsw.snap";
        const std::vector<uint64_t> params = {DIM, n_base, HNSW_M, HNSW_EFC};
        double build = 0.0;
        if (!force_rebuild && load_index_snapshot(cache, params, hnsw, build)) {
            std::printf("\n[HNSW]  loaded cache %s (M=%zu, orig build %.0f ms)\n",
                        cache.c_str(), HNSW_M, build);
        } else {
            std::printf("\n[HNSW]  building (M=%zu, efC=%zu) — 1M serial inserts...\n",
                        HNSW_M, HNSW_EFC);
            build = build_ms_of([&] {
                for (size_t i = 0; i < n_base; ++i) {
                    hnsw.add(base.data() + i * DIM);
                    if ((i + 1) % 100000 == 0)
                        std::printf("    ... %zu / %zu inserted\n", i + 1, n_base);
                }
            });
            std::printf("  build %.0f ms\n", build);
            save_index_snapshot(cache, build, params, hnsw);
            std::printf("  cached -> %s\n", cache.c_str());
        }

        const std::vector<size_t>* ef_grid[2] = {&HNSW_EF_K10, &HNSW_EF_K100};
        for (size_t ki = 0; ki < RECALL_K.size(); ++ki) {
            size_t K = RECALL_K[ki];
            const auto& efs = *ef_grid[std::min(ki, (size_t)1)];
            for (size_t ef : efs) {
                hnsw.set_ef_search(ef);
                double rec = recall_at_k(hnsw, gt.data(), gt_dim, queries.data(), q_recall, K);
                auto ts = time_queries(hnsw, queries.data(), q_timing, K, TIMING_REPS);
                rows.push_back({"hnsw", "ef", (long)K, (long)ef, build, rec, ts.qps,
                                ts.mean_us, ts.p50_us, ts.p95_us, ts.p99_us});
                std::printf("  ef=%-4zu K=%-3zu recall@%zu %.4f | %8.1f QPS | "
                            "p95 %.0f us\n", ef, K, K, rec, ts.qps, ts.p95_us);
            }
        }
    }

    write_csv(rows, csv_path, variant, append);
    std::printf("plot with:  python3 docs/plot_sift.py %s\n", csv_path.c_str());
}
