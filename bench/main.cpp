#include <vdb.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <random>
#include <utility>
#include <vector>

namespace {

using clk = std::chrono::steady_clock;

float l2_sq(const float* a, const float* b, size_t d) {
    float s = 0.0f;
    for (size_t i = 0; i < d; ++i) {
        float x = a[i] - b[i];
        s += x * x;
    }
    return s;
}

std::vector<std::vector<float>> random_vectors(size_t n, size_t dim, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    std::vector<std::vector<float>> v(n, std::vector<float>(dim));
    for (auto& vec : v) for (auto& x : vec) x = u(rng);
    return v;
}

// Mixture-of-Gaussians data: simulates the natural clustering you see in real
// embeddings. ANN methods exploit this structure; pure uniform-random data
// is the worst-case for HNSW because distances concentrate near a single value.
std::vector<std::vector<float>> clustered_vectors(size_t n, size_t dim,
                                                  size_t k_clusters, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> center_u(-10.0f, 10.0f);
    std::normal_distribution<float> noise(0.0f, 1.0f);
    std::uniform_int_distribution<size_t> ci(0, k_clusters - 1);

    std::vector<std::vector<float>> centers(k_clusters, std::vector<float>(dim));
    for (auto& c : centers) for (auto& x : c) x = center_u(rng);

    std::vector<std::vector<float>> data(n, std::vector<float>(dim));
    for (auto& v : data) {
        size_t cl = ci(rng);
        for (size_t i = 0; i < dim; ++i) v[i] = centers[cl][i] + noise(rng);
    }
    return data;
}

std::vector<uint32_t> brute_topk(const std::vector<std::vector<float>>& data,
                                 const float* q, size_t dim, size_t K) {
    std::vector<std::pair<float, uint32_t>> scored;
    scored.reserve(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        scored.emplace_back(l2_sq(q, data[i].data(), dim), (uint32_t)i);
    }
    std::partial_sort(scored.begin(), scored.begin() + K, scored.end(),
        [](const auto& a, const auto& b){ return a.first < b.first; });
    std::vector<uint32_t> out;
    out.reserve(K);
    for (size_t i = 0; i < K; ++i) out.push_back(scored[i].second);
    return out;
}

int recall(const std::vector<InternalId>& got, const std::vector<uint32_t>& truth) {
    int hits = 0;
    for (auto g : got) {
        for (auto t : truth) if ((uint32_t)g == t) { ++hits; break; }
    }
    return hits;
}

struct Result {
    size_t N;
    double build_ms;
    double brute_us_per_q;
    double hnsw_us_per_q;
    double speedup;
    double recall_pct;
};

enum class DataKind { Uniform, Clustered };

Result run_one(size_t N, size_t dim, size_t K, size_t ef_search, size_t n_queries,
               DataKind kind = DataKind::Uniform) {
    auto data = (kind == DataKind::Uniform)
        ? random_vectors(N, dim, /*seed=*/1234)
        : clustered_vectors(N, dim, /*k_clusters=*/50, /*seed=*/1234);

    VDBConfig cfg;
    cfg.hnsw.dim = dim;
    cfg.hnsw.M    = 16;
    cfg.hnsw.Mmax = 16;
    cfg.hnsw.Mmax0 = 32;
    cfg.hnsw.ef   = 200;  // efConstruction
    VDB db(cfg);

    auto t0 = clk::now();
    for (const auto& v : data) db.insert(v.data());
    auto t1 = clk::now();
    double build_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    auto queries = (kind == DataKind::Uniform)
        ? random_vectors(n_queries, dim, /*seed=*/9876)
        : clustered_vectors(n_queries, dim, /*k_clusters=*/50, /*seed=*/9876);

    // Brute force
    int total_hits = 0;
    std::vector<std::vector<uint32_t>> truths(n_queries);
    auto tb0 = clk::now();
    for (size_t q = 0; q < n_queries; ++q) {
        truths[q] = brute_topk(data, queries[q].data(), dim, K);
    }
    auto tb1 = clk::now();
    double brute_us =
        std::chrono::duration<double, std::micro>(tb1 - tb0).count() / n_queries;

    // HNSW
    auto th0 = clk::now();
    for (size_t q = 0; q < n_queries; ++q) {
        auto got = db.search(queries[q].data(), K, ef_search);
        total_hits += recall(got, truths[q]);
    }
    auto th1 = clk::now();
    double hnsw_us =
        std::chrono::duration<double, std::micro>(th1 - th0).count() / n_queries;

    return {
        N, build_ms, brute_us, hnsw_us,
        brute_us / hnsw_us,
        100.0 * total_hits / (n_queries * (double)K),
    };
}

}  // namespace

int main() {
    const size_t dim       = 128;
    const size_t K         = 10;
    const size_t ef_search = 100;
    const size_t n_queries = 200;

    std::printf("HNSW vs brute force kNN\n");
    std::printf("dim=%zu  K=%zu  ef_search=%zu  queries=%zu\n\n",
                dim, K, ef_search, n_queries);

    auto print_header = [] {
        std::printf("%-8s %-12s %-14s %-14s %-10s %-10s\n",
                    "N", "Build(ms)", "Brute(us/q)", "HNSW(us/q)", "Speedup", "Recall@10");
        std::printf("%-8s %-12s %-14s %-14s %-10s %-10s\n",
                    "----", "---------", "-----------", "----------", "-------", "---------");
    };
    auto print_row = [](const Result& r) {
        std::printf("%-8zu %-12.1f %-14.2f %-14.2f %-10.1fx %-9.1f%%\n",
                    r.N, r.build_ms, r.brute_us_per_q, r.hnsw_us_per_q,
                    r.speedup, r.recall_pct);
    };

    std::printf("== Uniform random data (curse-of-dim worst case) ==\n");
    print_header();
    std::fflush(stdout);
    for (size_t N : {1000u, 5000u, 20000u}) {
        std::printf("  ... running N=%zu\033[K\r", N);
        std::fflush(stdout);
        auto r = run_one(N, dim, K, ef_search, n_queries, DataKind::Uniform);
        print_row(r);
        std::fflush(stdout);
    }

    std::printf("\n== Clustered data (50 Gaussian clusters, simulates real embeddings) ==\n");
    print_header();
    std::fflush(stdout);
    for (size_t N : {1000u, 5000u, 20000u}) {
        std::printf("  ... running N=%zu\033[K\r", N);
        std::fflush(stdout);
        auto r = run_one(N, dim, K, ef_search, n_queries, DataKind::Clustered);
        print_row(r);
        std::fflush(stdout);
    }

    return 0;
}
