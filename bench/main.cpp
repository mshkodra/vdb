#include "bench.h"

#include "brute_index.h"
#include "distance.h"

#include <cstdint>
#include <cstdio>
#include <random>
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

}  // namespace

int main() {
    std::printf("vdb Stage 1 benchmarks — compare MIN columns across versions\n");
    bench_distance();
    bench_brute();
    return 0;
}
