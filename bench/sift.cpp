#include "sift.h"

#include "bench.h"

#include "brute_index.h"
#include "distance.h"
#include "hnsw_index.h"
#include "ivf_index.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <limits>
#include <string>
#include <unordered_set>
#include <vector>

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

// IVF. nlist ~ sqrt(N) ~= 1000; nprobe is the recall/speed knob.
constexpr size_t IVF_NLIST        = 1024;
constexpr size_t IVF_KMEANS_ITERS = 15;
const std::vector<size_t> IVF_NPROBE = {1, 4, 8, 16, 32, 64, 128, 256};

// Recall Ks reported for every method (ground truth ships top-100).
const std::vector<size_t> RECALL_K = {10, 100};

// Evaluation budget. Recall is averaged over the full query set (the canonical
// SIFT number); latency is measured on a subset with min-of-reps to cut noise.
constexpr size_t Q_RECALL   = 10000;  // full SIFT query set
constexpr size_t Q_TIMING   = 1000;   // subset for latency/QPS
constexpr int    TIMING_REPS = 5;
// Brute at N=1M is ~tens of ms/query; anchor it on a small query subset only.
constexpr size_t Q_BRUTE    = 200;
constexpr int    BRUTE_REPS = 3;
// ============================================================================

const char* const CSV_PATH = "docs/results/sift1m.csv";

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
        double best = std::numeric_limits<double>::infinity();
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

void write_csv(const std::vector<Row>& rows, bool append) {
    std::FILE* f = std::fopen(CSV_PATH, append ? "a" : "w");
    if (!f) {
        std::printf("!! could not open %s (mkdir -p docs/results first)\n", CSV_PATH);
        return;
    }
    if (!append)
        std::fprintf(f, "dataset,method,k,knob,knob_val,build_ms,recall,qps,"
                        "mean_us,p50_us,p95_us,p99_us\n");
    for (const auto& r : rows)
        std::fprintf(f, "sift1m,%s,%ld,%s,%ld,%.1f,%.4f,%.1f,%.3f,%.3f,%.3f,%.3f\n",
                     r.method.c_str(), r.k, r.knob.c_str(), r.knob_val, r.build_ms,
                     r.recall, r.qps, r.mean_us, r.p50_us, r.p95_us, r.p99_us);
    std::fclose(f);
    std::printf("\nwrote %zu rows -> %s\n", rows.size(), CSV_PATH);
}

}  // namespace

void run_sift(const char* data_dir, const char* methods) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);  // live progress even when piped
    const std::string dir = data_dir;
    const std::string which = methods ? methods : "all";
    // A subset run appends to the CSV so brute/IVF/HNSW can be resumed
    // independently (e.g. after a crash mid-build) without re-paying builds.
    const bool run_all = (which == "all");
    auto want = [&](const char* m) {
        return run_all || which.find(m) != std::string::npos;
    };
    auto metric = vdb::metric_fn(vdb::Metric::L2);

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
        vdb::BruteIndex brute(DIM, metric);
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
        vdb::IVFIndex ivf({DIM, IVF_NLIST, IVF_NPROBE.front(), IVF_KMEANS_ITERS}, metric);
        std::printf("\n[IVF]  training (nlist=%zu, iters=%zu) over %zu vectors — "
                    "this is the slow one...\n", IVF_NLIST, IVF_KMEANS_ITERS, n_base);
        double build = build_ms_of([&] { ivf.train(base.data(), n_base); });
        std::printf("  build %.0f ms\n", build);
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
        vdb::HNSWIndex hnsw(
            {DIM, HNSW_M, HNSW_M, 2 * HNSW_M, HNSW_EFC, 0.0f}, metric);
        std::printf("\n[HNSW]  building (M=%zu, efC=%zu) — 1M serial inserts...\n",
                    HNSW_M, HNSW_EFC);
        double build = build_ms_of([&] {
            for (size_t i = 0; i < n_base; ++i) {
                hnsw.add(base.data() + i * DIM);
                if ((i + 1) % 100000 == 0)
                    std::printf("    ... %zu / %zu inserted\n", i + 1, n_base);
            }
        });
        std::printf("  build %.0f ms\n", build);

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

    write_csv(rows, !run_all);
    std::printf("plot with:  python3 docs/plot_sift.py\n");
}
