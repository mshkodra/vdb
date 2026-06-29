#include "bench.h"

#include "brute_index.h"
#include "distance.h"
#include "ivf_index.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <unordered_set>
#include <vector>

// Stage 1 measurement harness. Run with `make bench`. Compare the MIN columns
// across implementation versions to see whether a change actually helped.
//
// Things to A/B with this: L2 with vs without sqrt; vector<vector<float>> vs a
// flat array in BruteIndex; partial_sort vs heap for top-K; `-O2` vs
// `-O3 -march=native`. Change one thing, rerun, compare min.

namespace {

// Row-major pool of n vectors x dim floats, seeded for reproducibility.
std::vector<float> random_flat(size_t n, size_t dim, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    std::vector<float> v(n * dim);
    for (auto& x : v) x = u(rng);
    return v;
}

// Gaussian blobs: nc random centers, each vector = a random center + noise of
// stddev `spread`. Tight, separated clusters give IVF the structure it exploits
// the opposite of random_flat()'s structureless uniform fill.
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

struct Metric {
    const char* name;
    DistanceFn  fn;
};

void bench_distance() {
    std::printf("\n== distance functions  (ns per call; lower is better) ==\n");
    std::printf("%-6s %-8s %-10s %-10s %-10s %-12s\n",
                "dim", "metric", "min", "median", "stddev", "Mdist/s");
    std::printf("%-6s %-8s %-10s %-10s %-10s %-12s\n",
                "---", "------", "---", "------", "------", "-------");

    for (size_t dim : {64u, 128u, 512u, 1024u}) {
        const size_t B = 4096;  // distance calls per measured rep
        auto pool = random_flat(B, dim, /*seed=*/1);
        auto query = random_flat(1, dim, /*seed=*/2);

        Metric metrics[] = {
            {"l2",     vdb::metric_fn(vdb::Metric::L2)},
            {"negIP",  vdb::metric_fn(vdb::Metric::InnerProduct)},
            {"cosine", vdb::metric_fn(vdb::Metric::Cosine)},
        };

        for (auto& m : metrics) {
            auto st = bench::measure([&] {
                float acc = 0.0f;
                for (size_t i = 0; i < B; ++i)
                    acc += m.fn(query.data(), pool.data() + i * dim, dim);
                bench::do_not_optimize(acc);
            });
            const double ns_call = st.min_ns / B;
            const double mdist_s = 1000.0 / ns_call;  // millions of calls / sec
            std::printf("%-6zu %-8s %-10.2f %-10.2f %-10.2f %-12.1f\n",
                        dim, m.name, ns_call, st.median_ns / B,
                        st.stddev_ns / B, mdist_s);
        }
    }
}

void bench_brute() {
    const size_t dim = 128, K = 10, Q = 64;
    std::printf("\n== brute-force search  (dim=%zu, K=%zu, %zu queries) ==\n",
                dim, K, Q);
    std::printf("%-8s %-14s %-14s %-12s\n",
                "N", "us/query(min)", "us/query(med)", "QPS");
    std::printf("%-8s %-14s %-14s %-12s\n",
                "-", "------------", "------------", "---");

    for (size_t N : {1000u, 10000u, 50000u}) {
        auto data = random_flat(N, dim, /*seed=*/1);
        auto queries = random_flat(Q, dim, /*seed=*/2);

        vdb::BruteIndex idx(dim, vdb::metric_fn(vdb::Metric::L2));
        for (size_t i = 0; i < N; ++i) idx.add(data.data() + i * dim);

        auto st = bench::measure([&] {
            size_t hits = 0;
            for (size_t qi = 0; qi < Q; ++qi)
                hits += idx.search(queries.data() + qi * dim, K).size();
            bench::do_not_optimize(hits);
        }, /*reps=*/20, /*warmup=*/3);

        const double us_q_min = (st.min_ns / 1000.0) / Q;
        const double us_q_med = (st.median_ns / 1000.0) / Q;
        std::printf("%-8zu %-14.2f %-14.2f %-12.0f\n",
                    N, us_q_min, us_q_med, 1e6 / us_q_min);
    }
}

// Brute-force top-K ids per query: the recall oracle.
std::vector<std::unordered_set<InternalId>> truth_sets(
    const vdb::BruteIndex& oracle, const float* queries, size_t Q, size_t dim,
    size_t K) {
    std::vector<std::unordered_set<InternalId>> truth(Q);
    for (size_t q = 0; q < Q; ++q) {
        for (auto& [id, d] : oracle.search(queries + q * dim, K))
            truth[q].insert(id);
    }
    return truth;
}

// Mean recall@K: fraction of each query's true top-K that IVF also returns.
double recall_at_k(const vdb::IVFIndex& ivf,
                   const std::vector<std::unordered_set<InternalId>>& truth,
                   const float* queries, size_t Q, size_t dim, size_t K) {
    size_t hits = 0;
    for (size_t q = 0; q < Q; ++q) {
        for (auto& [id, d] : ivf.search(queries + q * dim, K))
            if (truth[q].count(id)) ++hits;
    }
    return static_cast<double>(hits) / (Q * K);
}

// us/query (min) for running all Q searches through an index.
template <typename Index>
double us_per_query(const Index& idx, const float* queries, size_t Q, size_t dim,
                    size_t K) {
    auto st = bench::measure([&] {
        size_t h = 0;
        for (size_t q = 0; q < Q; ++q) h += idx.search(queries + q * dim, K).size();
        bench::do_not_optimize(h);
    }, /*reps=*/20, /*warmup=*/3);
    return (st.min_ns / 1000.0) / Q;
}

void bench_ivf() {
    const size_t dim = 128, K = 10, Q = 100, iters = 25;
    auto metric = vdb::metric_fn(vdb::Metric::L2);

    // --- nprobe sweep: the recall/speed dial (train once, vary nprobe) --------
    {
        const size_t N = 50000, nlist = static_cast<size_t>(std::sqrt((double)N));
        auto data = random_flat(N, dim, 1);
        auto queries = random_flat(Q, dim, 2);

        vdb::BruteIndex oracle(dim, metric);
        for (size_t i = 0; i < N; ++i) oracle.add(data.data() + i * dim);
        auto truth = truth_sets(oracle, queries.data(), Q, dim, K);
        double brute_us = us_per_query(oracle, queries.data(), Q, dim, K);

        vdb::IVFIndex ivf({dim, nlist, 8, iters}, metric);
        ivf.train(data.data(), N);

        std::printf("\n== IVF: nprobe sweep  (N=%zu, dim=%zu, nlist=%zu, K=%zu) ==\n",
                    N, dim, nlist, K);
        std::printf("  brute oracle: %.2f us/query, %.0f QPS\n", brute_us, 1e6 / brute_us);
        std::printf("%-8s %-10s %-14s %-12s %-10s\n",
                    "nprobe", "recall", "us/query(min)", "QPS", "speedup");
        for (size_t np : {1u, 2u, 4u, 8u, 16u, 32u, 64u, (unsigned)nlist}) {
            ivf.set_nprobe(np);
            double rec = recall_at_k(ivf, truth, queries.data(), Q, dim, K);
            double us = us_per_query(ivf, queries.data(), Q, dim, K);
            std::printf("%-8zu %-10.3f %-14.2f %-12.0f %-10.1f\n",
                        np, rec, us, 1e6 / us, brute_us / us);
        }
    }

    // --- nlist sweep: cluster granularity at fixed nprobe ---------------------
    {
        const size_t N = 50000, nprobe = 8;
        auto data = random_flat(N, dim, 1);
        auto queries = random_flat(Q, dim, 2);

        vdb::BruteIndex oracle(dim, metric);
        for (size_t i = 0; i < N; ++i) oracle.add(data.data() + i * dim);
        auto truth = truth_sets(oracle, queries.data(), Q, dim, K);

        std::printf("\n== IVF: nlist sweep  (N=%zu, dim=%zu, nprobe=%zu, K=%zu) ==\n",
                    N, dim, nprobe, K);
        std::printf("%-8s %-12s %-10s %-14s %-12s\n",
                    "nlist", "probe%", "recall", "us/query(min)", "QPS");
        for (size_t nlist : {64u, 128u, 256u, 512u}) {
            vdb::IVFIndex ivf({dim, nlist, nprobe, iters}, metric);
            ivf.train(data.data(), N);
            double rec = recall_at_k(ivf, truth, queries.data(), Q, dim, K);
            double us = us_per_query(ivf, queries.data(), Q, dim, K);
            std::printf("%-8zu %-12.1f %-10.3f %-14.2f %-12.0f\n",
                        nlist, 100.0 * nprobe / nlist, rec, us, 1e6 / us);
        }
    }

    // --- N sweep: how IVF's speedup over brute grows with dataset size --------
    {
        const size_t nprobe = 8;
        std::printf("\n== IVF: N sweep  (dim=%zu, nlist=sqrt(N), nprobe=%zu, K=%zu) ==\n",
                    dim, nprobe, K);
        std::printf("%-8s %-8s %-10s %-14s %-14s %-10s\n",
                    "N", "nlist", "recall", "ivf us/q", "brute us/q", "speedup");
        for (size_t N : {1000u, 10000u, 50000u}) {
            size_t nlist = static_cast<size_t>(std::sqrt((double)N));
            auto data = random_flat(N, dim, 1);
            auto queries = random_flat(Q, dim, 2);

            vdb::BruteIndex oracle(dim, metric);
            for (size_t i = 0; i < N; ++i) oracle.add(data.data() + i * dim);
            auto truth = truth_sets(oracle, queries.data(), Q, dim, K);
            double brute_us = us_per_query(oracle, queries.data(), Q, dim, K);

            vdb::IVFIndex ivf({dim, nlist, nprobe, iters}, metric);
            ivf.train(data.data(), N);
            double rec = recall_at_k(ivf, truth, queries.data(), Q, dim, K);
            double us = us_per_query(ivf, queries.data(), Q, dim, K);
            std::printf("%-8zu %-8zu %-10.3f %-14.2f %-14.2f %-10.1f\n",
                        N, nlist, rec, us, brute_us, brute_us / us);
        }
    }

    // --- dim sweep: dimensionality at fixed N/nprobe --------------------------
    {
        const size_t N = 20000, nprobe = 8;
        size_t nlist = static_cast<size_t>(std::sqrt((double)N));
        std::printf("\n== IVF: dim sweep  (N=%zu, nlist=%zu, nprobe=%zu, K=%zu) ==\n",
                    N, nlist, nprobe, K);
        std::printf("%-8s %-10s %-14s %-14s %-10s\n",
                    "dim", "recall", "ivf us/q", "brute us/q", "speedup");
        for (size_t d : {64u, 128u, 512u}) {
            auto data = random_flat(N, d, 1);
            auto queries = random_flat(Q, d, 2);

            vdb::BruteIndex oracle(d, metric);
            for (size_t i = 0; i < N; ++i) oracle.add(data.data() + i * d);
            auto truth = truth_sets(oracle, queries.data(), Q, d, K);
            double brute_us = us_per_query(oracle, queries.data(), Q, d, K);

            vdb::IVFIndex ivf({d, nlist, nprobe, iters}, metric);
            ivf.train(data.data(), N);
            double rec = recall_at_k(ivf, truth, queries.data(), Q, d, K);
            double us = us_per_query(ivf, queries.data(), Q, d, K);
            std::printf("%-8zu %-10.3f %-14.2f %-14.2f %-10.1f\n",
                        d, rec, us, brute_us, brute_us / us);
        }
    }
}

// Same nprobe sweep as bench_ivf, but on clustered data: the A/B that isolates
// the effect of data structure on recall (uniform vs Gaussian blobs).
void bench_ivf_clustered() {
    const size_t N = 50000, dim = 128, K = 10, Q = 100, iters = 25;
    const size_t nlist = static_cast<size_t>(std::sqrt((double)N));
    const size_t nc = 100;       // true clusters
    const float spread = 0.10f;  // cluster tightness (smaller => more structure)
    auto metric = vdb::metric_fn(vdb::Metric::L2);

    auto data = clustered_flat(N, dim, nc, spread, 1);
    auto queries = clustered_flat(Q, dim, nc, spread, 2);

    vdb::BruteIndex oracle(dim, metric);
    for (size_t i = 0; i < N; ++i) oracle.add(data.data() + i * dim);
    auto truth = truth_sets(oracle, queries.data(), Q, dim, K);

    vdb::IVFIndex ivf({dim, nlist, 8, iters}, metric);
    ivf.train(data.data(), N);

    std::printf("\n== IVF: nprobe sweep on CLUSTERED data "
                "(N=%zu, dim=%zu, nlist=%zu, %zu blobs, spread=%.2f) ==\n",
                N, dim, nlist, nc, spread);
    std::printf("%-8s %-10s %-14s %-12s\n", "nprobe", "recall", "us/query(min)", "QPS");
    for (size_t np : {1u, 2u, 4u, 8u, 16u, 32u, 64u, (unsigned)nlist}) {
        ivf.set_nprobe(np);
        double rec = recall_at_k(ivf, truth, queries.data(), Q, dim, K);
        double us = us_per_query(ivf, queries.data(), Q, dim, K);
        std::printf("%-8zu %-10.3f %-14.2f %-12.0f\n", np, rec, us, 1e6 / us);
    }
}

}  // namespace

int main() {
    std::printf("vdb Stage 1 benchmarks — compare MIN columns across versions\n");
    bench_distance();
    bench_brute();
    bench_ivf();
    bench_ivf_clustered();
    return 0;
}
