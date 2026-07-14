#include "compare.h"

#include "bench.h"

#include "brute_index.h"
#include "distance.h"
#include "hnsw_index.h"
#include "ivf_index.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <limits>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

namespace {

// ============================================================================
//  EXPERIMENT GRID  —  this block is yours to design (see CLAUDE.md).
//  Everything below it is plumbing. Change these; predict before running.
// ============================================================================
constexpr size_t N   = 2000;    // vectors per dataset (HNSW build is the limiter)
constexpr size_t DIM = 128;
constexpr size_t K   = 10;      // neighbours per query
constexpr size_t Q   = 200;     // query set size

// clustered-data shape
constexpr size_t CLUSTERS = 100;
constexpr float  SPREAD   = 0.10f;

// IVF: fixed nlist = round(sqrt(N)); sweep nprobe.
const std::vector<size_t> IVF_NPROBE = {1, 4, 8, 16, 32, 64,
                                        static_cast<size_t>(std::sqrt((double)N))};
constexpr size_t IVF_KMEANS_ITERS = 25;

// HNSW: fixed M; sweep ef (search-time knob). Build once per (dataset, M).
constexpr size_t HNSW_M = 16;
constexpr size_t HNSW_EF_CONSTRUCTION = 100;  // build-time candidate width
const std::vector<size_t> HNSW_EF = {10, 20, 40, 80, 160, 320};
// ============================================================================

const char* const CSV_PATH = "docs/results/frontier.csv";

// ---- data generators (self-contained; mirror bench/main.cpp) ----------------

std::vector<float> random_flat(size_t n, size_t dim, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    std::vector<float> v(n * dim);
    for (auto& x : v) x = u(rng);
    return v;
}

std::vector<float> clustered_flat(size_t n, size_t dim, size_t nc, float spread,
                                  uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    std::normal_distribution<float> noise(0.0f, spread);

    std::vector<float> centers(nc * dim);
    for (auto& x : centers) x = u(rng);

    std::uniform_int_distribution<size_t> pick(0, nc - 1);
    std::vector<float> v(n * dim);
    for (size_t i = 0; i < n; ++i) {
        size_t c = pick(rng);
        for (size_t d = 0; d < dim; ++d)
            v[i * dim + d] = centers[c * dim + d] + noise(rng);
    }
    return v;
}

// ---- evaluation helpers -----------------------------------------------------

std::vector<std::unordered_set<InternalId>> truth_sets(
    const vdb::BruteIndex& oracle, const float* queries, size_t dim) {
    std::vector<std::unordered_set<InternalId>> truth(Q);
    for (size_t q = 0; q < Q; ++q)
        for (auto& [id, d] : oracle.search(queries + q * dim, K))
            truth[q].insert(id);
    return truth;
}

template <typename Index>
double recall_at_k(const Index& idx,
                   const std::vector<std::unordered_set<InternalId>>& truth,
                   const float* queries, size_t dim) {
    size_t hits = 0;
    for (size_t q = 0; q < Q; ++q)
        for (auto& [id, d] : idx.search(queries + q * dim, K))
            if (truth[q].count(id)) ++hits;
    return static_cast<double>(hits) / (Q * K);
}

double percentile(const std::vector<double>& sorted, double p) {
    if (sorted.empty()) return 0.0;
    size_t idx = static_cast<size_t>(std::ceil(p / 100.0 * sorted.size()));
    if (idx > 0) --idx;
    return sorted[std::min(idx, sorted.size() - 1)];
}

struct QStats {
    double qps;
    double mean_us, p50_us, p95_us, p99_us;
};

// Per-query latency: min over `reps` for each query, then percentiles across
// the query set. QPS derived from the mean of the per-query mins.
template <typename Index>
QStats time_queries(const Index& idx, const float* queries, size_t dim,
                    int reps = 7) {
    for (size_t q = 0; q < Q; ++q)  // warm-up pass, discarded
        bench::do_not_optimize(idx.search(queries + q * dim, K).size());

    std::vector<double> per_us(Q);
    for (size_t q = 0; q < Q; ++q) {
        double best = std::numeric_limits<double>::infinity();
        for (int r = 0; r < reps; ++r) {
            auto t0 = bench::clk::now();
            auto res = idx.search(queries + q * dim, K);
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

// ---- CSV emission -----------------------------------------------------------

struct Row {
    std::string dataset, method, knob;
    long        knob_val;
    double      build_ms, recall, qps, mean_us, p50_us, p95_us, p99_us;
};

void write_csv(const std::vector<Row>& rows) {
    std::FILE* f = std::fopen(CSV_PATH, "w");
    if (!f) {
        std::printf("!! could not open %s for writing "
                    "(mkdir -p docs/results first)\n", CSV_PATH);
        return;
    }
    std::fprintf(f, "dataset,method,knob,knob_val,build_ms,recall,qps,"
                    "mean_us,p50_us,p95_us,p99_us\n");
    for (const auto& r : rows)
        std::fprintf(f, "%s,%s,%s,%ld,%.3f,%.4f,%.1f,%.3f,%.3f,%.3f,%.3f\n",
                     r.dataset.c_str(), r.method.c_str(), r.knob.c_str(),
                     r.knob_val, r.build_ms, r.recall, r.qps, r.mean_us,
                     r.p50_us, r.p95_us, r.p99_us);
    std::fclose(f);
    std::printf("\nwrote %zu rows -> %s\n", rows.size(), CSV_PATH);
}

double build_ms_of(const std::function<void()>& build) {
    auto t0 = bench::clk::now();
    build();
    auto t1 = bench::clk::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// ---- one dataset: build all three, sweep knobs, collect rows ----------------

void run_dataset(const char* name, const std::vector<float>& data,
                 const std::vector<float>& queries, std::vector<Row>& out) {
    auto metric = vdb::metric_fn(vdb::Metric::L2);
    const size_t nlist = static_cast<size_t>(std::sqrt((double)N));

    std::printf("\n===== dataset: %s  (N=%zu, dim=%zu, K=%zu, Q=%zu) =====\n",
                name, N, DIM, K, Q);

    // Brute oracle: also the ground truth and the recall=1.0 frontier point.
    vdb::BruteIndex oracle(DIM, metric);
    double brute_build = build_ms_of([&] {
        for (size_t i = 0; i < N; ++i) oracle.add(data.data() + i * DIM);
    });
    auto truth = truth_sets(oracle, queries.data(), DIM);
    {
        auto ts = time_queries(oracle, queries.data(), DIM);
        out.push_back({name, "brute", "none", 0, brute_build, 1.0, ts.qps,
                       ts.mean_us, ts.p50_us, ts.p95_us, ts.p99_us});
        std::printf("  brute : build %.1f ms | recall 1.000 | %8.0f QPS | "
                    "p95 %.1f us\n", brute_build, ts.qps, ts.p95_us);
    }

    // IVF: one trained index, sweep nprobe.
    {
        // NOTE: this IVFIndex::train() also populates the inverted lists with all
        // N vectors, so it is the full build. Do NOT also call add() here or every
        // vector is inserted twice (duplicate ids -> recall caps near 0.5).
        vdb::IVFIndex ivf({DIM, nlist, IVF_NPROBE.front(), IVF_KMEANS_ITERS}, metric);
        double build = build_ms_of([&] { ivf.train(data.data(), N); });
        std::printf("  IVF   : build %.1f ms (nlist=%zu)\n", build, nlist);
        for (size_t np : IVF_NPROBE) {
            ivf.set_nprobe(np);
            double rec = recall_at_k(ivf, truth, queries.data(), DIM);
            auto ts = time_queries(ivf, queries.data(), DIM);
            out.push_back({name, "ivf", "nprobe", (long)np, build, rec, ts.qps,
                           ts.mean_us, ts.p50_us, ts.p95_us, ts.p99_us});
            std::printf("      nprobe=%-4zu recall %.3f | %8.0f QPS | p95 %.1f us\n",
                        np, rec, ts.qps, ts.p95_us);
        }
    }

    // HNSW: one built graph, sweep ef (pure search-time knob).
    {
        vdb::HNSWIndex hnsw(
            {DIM, HNSW_M, HNSW_M, 2 * HNSW_M, HNSW_EF_CONSTRUCTION, 0.0f}, metric);
        double build = build_ms_of([&] {
            for (size_t i = 0; i < N; ++i) hnsw.add(data.data() + i * DIM);
        });
        std::printf("  HNSW  : build %.1f ms (M=%zu)\n", build, HNSW_M);
        for (size_t ef : HNSW_EF) {
            hnsw.set_ef_search(ef);
            double rec = recall_at_k(hnsw, truth, queries.data(), DIM);
            auto ts = time_queries(hnsw, queries.data(), DIM);
            out.push_back({name, "hnsw", "ef", (long)ef, build, rec, ts.qps,
                           ts.mean_us, ts.p50_us, ts.p95_us, ts.p99_us});
            std::printf("      ef=%-4zu recall %.3f | %8.0f QPS | p95 %.1f us\n",
                        ef, rec, ts.qps, ts.p95_us);
        }
    }
}

}  // namespace

void run_compare() {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);  // live progress even when piped
    std::printf("\n######## Stage 4: comparative recall-vs-QPS frontier ########\n");
    std::vector<Row> rows;

    run_dataset("uniform",
                random_flat(N, DIM, 1), random_flat(Q, DIM, 2), rows);
    run_dataset("clustered",
                clustered_flat(N, DIM, CLUSTERS, SPREAD, 1),
                clustered_flat(Q, DIM, CLUSTERS, SPREAD, 2), rows);

    write_csv(rows);
    std::printf("plot with:  python3 docs/plot_frontier.py\n");
}
