#include "filter.h"

#include "bench.h"
#include "filter_planner.h"
#include "snapshot.h"
#include "vdb.h"

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
#include <numeric>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

using namespace vdb;

namespace {

// ============================================================================
//  EXPERIMENT GRID  —  this block is yours to design (see CLAUDE.md).
//  Everything below it is plumbing. Change these; predict before running.
//
//  Cost warning (predict these before you hit run):
//   * Two HNSW builds over the full 1M-vector base set (one per label
//     distribution) — each the same cost as bench/sift.cpp's HNSW build
//     (minutes). Cached to <data_dir>/cache/filter_<dist>.snap via VDB's own
//     save_snapshot/load_snapshot, so a re-run loads in seconds. VDBConfig
//     doesn't expose HNSW's M/efC knobs — `make_index_` builds HNSWConfig with
//     its defaults (M=16, Mmax0=32, ef=200), which happen to match
//     bench/sift.cpp's own chosen grid.
//   * K-means label generation (kmeans_labels) costs N*NC*iters distance
//     computations = 1e6 * 100 * 15 = 1.5e9 — ~10x cheaper than IVF's own
//     N*nlist*iters build in sift.cpp (nlist=1024), but still not free.
//   * The selectivity sweep is expensive at BOTH ends, for different reasons:
//     pre-filter's cost is O(N*s*dim), so it's slowest at the high-s end;
//     post-filter's over-fetch `want = K+N(1-s)` approaches N as s -> 0, so
//     HNSW's search degenerates toward a near-full graph traversal — plausibly
//     slower per query than brute force. Q_TIMING/TIMING_REPS are kept small
//     (unlike sift.cpp's 1000/5) specifically to bound this.
// ============================================================================
constexpr size_t DIM = 128;  // SIFT descriptors are 128-d

constexpr size_t NC           = 100;  // k-means clusters for correlated labels
constexpr size_t KMEANS_ITERS = 15;
constexpr unsigned LABEL_SEED = 7;

// Log-spaced, bracketing the ~3% percolation threshold from
// docs/design/METADATA_DETAILS.md §1.3.
const std::vector<double> SELECTIVITY = {0.5,  0.25,   0.1,    0.05,  0.025,
                                         0.01, 0.005, 0.0025, 0.001};

constexpr size_t K           = 10;
constexpr size_t Q_TIMING    = 100;
constexpr int    TIMING_REPS = 2;

// Subsample cap on the 1M-vector base set. Two full HNSW builds over 1M vectors is
// the intended "SIFT1M" scope, but under host contention that's not reliably
// tractable in one sitting; capping keeps a run honest-but-tractable. Set to a value
// >= n_base (or std::numeric_limits<size_t>::max()) for the uncapped full run.
constexpr size_t N_CAP = 100000;
// ============================================================================

// ---- .fvecs reader (mirrors bench/sift.cpp's — each driver keeps its own copy,
//      the established convention here; see e.g. clustered_flat() in compare.cpp
//      and main.cpp). Format: each record is [int32 dim][dim float32 payload], no
//      file header/count. ----
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

// ---- k-means label generation (standalone: mirrors IVFIndex<Dist>::train's Lloyd
//      iteration, but decoupled from ivf_index.h entirely — this is a benchmark-only
//      labeling step, not a reusable index, and IVFIndex's inverted_lists_/centroids_
//      are private with no accessor). ----
std::vector<uint32_t> kmeans_labels(const std::vector<float>& data, size_t n, size_t dim,
                                    size_t nc, size_t iters, unsigned seed) {
    std::mt19937 rng(seed);
    std::vector<size_t> perm(n);
    std::iota(perm.begin(), perm.end(), 0);
    std::shuffle(perm.begin(), perm.end(), rng);

    std::vector<std::vector<float>> centroids(nc, std::vector<float>(dim));
    for (size_t i = 0; i < nc; ++i) {
        const float* src = data.data() + perm[i] * dim;
        std::copy(src, src + dim, centroids[i].begin());
    }

    auto nearest = [&](const float* v) {
        uint32_t best = 0;
        float best_d = std::numeric_limits<float>::max();
        for (uint32_t c = 0; c < nc; ++c) {
            const float* cv = centroids[c].data();
            float d = 0.0f;
            for (size_t x = 0; x < dim; ++x) {
                const float diff = v[x] - cv[x];
                d += diff * diff;
            }
            if (d < best_d) { best_d = d; best = c; }
        }
        return best;
    };

    std::vector<uint32_t> assign(n);
    for (size_t iter = 0; iter < iters; ++iter) {
        std::vector<std::vector<float>> sums(nc, std::vector<float>(dim, 0.0f));
        std::vector<size_t> counts(nc, 0);
        for (size_t i = 0; i < n; ++i) {
            const float* v = data.data() + i * dim;
            const uint32_t c = nearest(v);
            assign[i] = c;
            float* s = sums[c].data();
            for (size_t x = 0; x < dim; ++x) s[x] += v[x];
            counts[c]++;
        }
        for (size_t c = 0; c < nc; ++c) {
            if (counts[c] == 0) continue;
            const float inv = 1.0f / static_cast<float>(counts[c]);
            for (size_t x = 0; x < dim; ++x) centroids[c][x] = sums[c][x] * inv;
        }
    }
    for (size_t i = 0; i < n; ++i) assign[i] = nearest(data.data() + i * dim);
    return assign;
}

// Uniform-random rank in [0,1) per vector, independent of embedding position —
// the "correlation footgun" case (docs/design/METADATA_DETAILS.md §1.2).
std::vector<double> uniform_ranks(size_t n, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    std::vector<double> out(n);
    for (size_t i = 0; i < n; ++i) out[i] = u(rng);
    return out;
}

// K-means-cluster-correlated rank: each cluster gets a random, shuffled band of
// width 1/NC within [0,1); a vector's rank is its cluster's band plus jitter within
// that band. As the selectivity threshold `s` sweeps up past a cluster's band, the
// whole cluster turns on together — the match set becomes a union of whole clusters,
// spatially clustered in embedding space (the realistic case per §1.2).
std::vector<double> kmeans_ranks(const std::vector<float>& data, size_t n, size_t dim,
                                 unsigned seed) {
    auto assign = kmeans_labels(data, n, dim, NC, KMEANS_ITERS, seed);

    std::vector<size_t> order(NC);
    std::iota(order.begin(), order.end(), 0);
    std::mt19937 order_rng(seed + 1);
    std::shuffle(order.begin(), order.end(), order_rng);
    std::vector<double> band(NC);
    for (size_t i = 0; i < NC; ++i) band[order[i]] = static_cast<double>(i) / static_cast<double>(NC);

    std::mt19937 jitter_rng(seed + 2);
    std::uniform_real_distribution<double> jitter(0.0, 1.0 / static_cast<double>(NC));
    std::vector<double> out(n);
    for (size_t i = 0; i < n; ++i) out[i] = band[assign[i]] + jitter(jitter_rng);
    return out;
}

std::vector<AttrSpec> rank_schema() {
    return {{"rank", AttrType::Float64, /*indexed=*/true}};
}

double build_ms_of(const std::function<void()>& build) {
    auto t0 = bench::clk::now();
    build();
    auto t1 = bench::clk::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
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

// Per-query latency = min over `reps`; percentiles across the query subset; QPS
// from the mean of the per-query mins. Same shape as bench/sift.cpp's time_queries,
// generalized over a callable (VDB::search/search_prefiltered aren't the uniform
// vdb::Index interface sift.cpp's version is templated on).
template <typename Fn>
QStats time_queries(Fn&& fn, const float* queries, size_t Q, int reps) {
    for (size_t q = 0; q < Q; ++q)  // warm-up, discarded
        bench::do_not_optimize(fn(queries + q * DIM).size());

    std::vector<double> per_us(Q);
    for (size_t q = 0; q < Q; ++q) {
        // Finite sentinel, not infinity(): -ffast-math implies -ffinite-math-only,
        // which makes infinities UB and lets the compiler miscompile this min.
        double best = std::numeric_limits<double>::max();
        for (int r = 0; r < reps; ++r) {
            auto t0 = bench::clk::now();
            auto res = fn(queries + q * DIM);
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
    const double mean = sum / static_cast<double>(Q);
    return {1e6 / mean, mean, percentile(per_us, 50), percentile(per_us, 95),
            percentile(per_us, 99)};
}

// Pre-filter is exact regardless of IndexKind (tests/test_filter.cpp proves this
// directly against a brute oracle), so it stands in as the filtered ground truth
// here — no separate O(N) brute-force pass needed. Denominator is Q*K since every
// `s` in the sweep matches far more than K vectors (smallest is s=0.001 * 1e6 =
// 1000 matches).
double post_recall_vs_prefilter(const VDB& db, const Predicate& pred, const float* queries,
                                size_t Q) {
    size_t hits = 0;
    for (size_t q = 0; q < Q; ++q) {
        const auto post = db.search(queries + q * DIM, K, pred);
        const auto pre  = db.search_prefiltered(queries + q * DIM, K, pred);
        std::unordered_set<ExternalId> pre_set(pre.begin(), pre.end());
        for (ExternalId id : post)
            if (pre_set.count(id)) ++hits;
    }
    return static_cast<double>(hits) / static_cast<double>(Q * K);
}

struct Row {
    std::string labels;    // "uniform" | "kmeans"
    std::string strategy;  // "post" | "pre"
    double      s;
    double      build_ms;
    double      recall_vs_prefilter;  // 1.0 for "pre" rows by definition
    double      qps, mean_us, p50_us, p95_us, p99_us;
    double      want;  // post-filter's over-fetch width K+N(1-s); 0 on "pre" rows
    double      scan;  // pre-filter's scan volume N*s*dim; 0 on "post" rows
};

void write_csv(const std::vector<Row>& rows, const std::string& path) {
    std::FILE* f = std::fopen(path.c_str(), "w");
    if (!f) {
        std::printf("!! could not open %s (mkdir -p docs/results first)\n", path.c_str());
        return;
    }
    std::fprintf(f, "labels,strategy,s,build_ms,recall_vs_prefilter,qps,mean_us,"
                    "p50_us,p95_us,p99_us,want,scan\n");
    for (const auto& r : rows)
        std::fprintf(f, "%s,%s,%.6f,%.1f,%.4f,%.1f,%.3f,%.3f,%.3f,%.3f,%.1f,%.1f\n",
                     r.labels.c_str(), r.strategy.c_str(), r.s, r.build_ms,
                     r.recall_vs_prefilter, r.qps, r.mean_us, r.p50_us, r.p95_us,
                     r.p99_us, r.want, r.scan);
    std::fclose(f);
    std::printf("\nwrote %zu rows -> %s\n", rows.size(), path.c_str());
}

}  // namespace

void run_filter_bench(const char* data_dir) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);  // live progress even when piped
    const std::string dir = data_dir;
    const std::string cache_dir = dir + "/cache";
    ::mkdir(cache_dir.c_str(), 0755);  // ignore EEXIST

    std::printf("\n######## Filtered search: post- vs pre-filter crossover on SIFT1M "
                "########\n");
    std::printf("loading from %s ...\n", dir.c_str());

    size_t n_base = 0, n_query = 0;
    auto base    = read_fvecs(dir + "/sift_base.fvecs", n_base);
    auto queries = read_fvecs(dir + "/sift_query.fvecs", n_query);
    std::printf("  base=%zu  query=%zu  (dim=%zu)\n", n_base, n_query, DIM);
    if (n_base > N_CAP) {
        std::printf("  capping to N_CAP=%zu (see the EXPERIMENT GRID comment)\n", N_CAP);
        n_base = N_CAP;
    }

    const size_t q_timing = std::min(Q_TIMING, n_query);

    std::vector<Row> rows;
    std::vector<CalibrationPoint> post_points, pre_points;

    auto run_distribution = [&](const std::string& label, std::vector<double> ranks) {
        const std::string cache = cache_dir + "/filter_" + label + ".snap";

        VDBConfig cfg;
        cfg.kind   = IndexKind::HNSW;
        cfg.dim    = DIM;
        cfg.metric = Metric::L2;
        cfg.schema = rank_schema();
        VDB db(cfg);

        double build_ms = 0.0;
        bool loaded = false;
        try {
            load_snapshot(db, cache);
            loaded = true;
        } catch (const std::exception&) {
            // No cache yet, or it doesn't match this VDB's config — build fresh.
        }

        if (loaded) {
            std::printf("\n[%s]  loaded cache %s (N=%zu)\n", label.c_str(), cache.c_str(),
                        db.size());
        } else {
            std::printf("\n[%s]  building HNSW over %zu vectors — this is the slow "
                        "one...\n", label.c_str(), n_base);
            build_ms = build_ms_of([&] {
                for (size_t i = 0; i < n_base; ++i) {
                    Record r;
                    r.attrs = {attr_float(ranks[i])};
                    db.insert(base.data() + i * DIM, r);
                    if ((i + 1) % 200000 == 0)
                        std::printf("    ... %zu / %zu inserted\n", i + 1, n_base);
                }
            });
            std::printf("  build %.0f ms\n", build_ms);
            save_snapshot(db, cache, 0);
            std::printf("  cached -> %s\n", cache.c_str());
        }

        for (double s : SELECTIVITY) {
            const Predicate pred = pred_range(0, attr_float(0.0), attr_float(s));

            auto post_fn = [&](const float* q) { return db.search(q, K, pred); };
            auto pre_fn  = [&](const float* q) { return db.search_prefiltered(q, K, pred); };

            const auto post_ts = time_queries(post_fn, queries.data(), q_timing, TIMING_REPS);
            const auto pre_ts  = time_queries(pre_fn, queries.data(), q_timing, TIMING_REPS);
            const double recall = post_recall_vs_prefilter(db, pred, queries.data(), q_timing);

            const double N    = static_cast<double>(n_base);
            const double want = static_cast<double>(K) + N * (1.0 - s);
            const double scan = N * s * static_cast<double>(DIM);

            rows.push_back({label, "post", s, build_ms, recall, post_ts.qps, post_ts.mean_us,
                            post_ts.p50_us, post_ts.p95_us, post_ts.p99_us, want, 0.0});
            rows.push_back({label, "pre", s, build_ms, 1.0, pre_ts.qps, pre_ts.mean_us,
                            pre_ts.p50_us, pre_ts.p95_us, pre_ts.p99_us, 0.0, scan});

            post_points.push_back({want, post_ts.mean_us});
            pre_points.push_back({scan, pre_ts.mean_us});

            std::printf("  s=%-8.4f post: %8.1f QPS (p95 %6.0f us, recall %.3f) | "
                        "pre: %8.1f QPS (p95 %6.0f us)\n",
                        s, post_ts.qps, post_ts.p95_us, recall, pre_ts.qps, pre_ts.p95_us);
        }
    };

    run_distribution("uniform", uniform_ranks(n_base, LABEL_SEED));
    run_distribution("kmeans", kmeans_ranks(base, n_base, DIM, LABEL_SEED));

    const std::string csv_path = "docs/results/filter.csv";
    write_csv(rows, csv_path);
    std::printf("plot with:  python3 docs/plot_filter.py %s\n", csv_path.c_str());

    // Calibrate the planner from this run's own measurements and report it, so the
    // numbers just produced are immediately checked against the model they justify.
    const FilterCalibration calib = calibrate(post_points, pre_points);
    std::printf("\n[planner] calibrated c_index=%.6f us/unit(want)  "
                "c_scan=%.6f us/unit(N*s*dim)\n", calib.c_index, calib.c_scan);
    std::printf("[planner] predicted choice per selectivity (K=%zu, N=%zu, dim=%zu):\n",
                K, n_base, DIM);
    for (double s : SELECTIVITY) {
        const FilterStrategy pick = plan_strategy(calib, n_base, DIM, K, s);
        std::printf("    s=%-8.4f -> %s\n", s, pick == FilterStrategy::Post ? "post" : "pre");
    }
}
