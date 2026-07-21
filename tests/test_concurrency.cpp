// Stage 7, step 1: HNSWIndex must be safe under concurrent add() + search().
// These tests are also the TSan/ASan targets (`make test-tsan`, `make test-asan`):
// the value assertions guard correctness, the sanitizers guard for data races and
// memory errors that a functional test alone cannot see.
#include "hnsw_index.h"

#include "brute_index.h"
#include "distance.h"
#include "test.h"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <random>
#include <thread>
#include <vector>

using namespace vdb;

namespace {

// Flat n*dim buffer of uniform vectors in [-1, 1], seeded for reproducibility.
std::vector<float> gen(size_t n, size_t dim, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    std::vector<float> v(n * dim);
    for (auto& x : v) x = u(rng);
    return v;
}

// Recall of `got` distances against the true top-K distances, matched as a
// multiset within eps. Ids differ between brute and a concurrently-built HNSW
// (insertion order is scrambled), so we compare by distance, not by id.
double recall_by_distance(std::vector<float> truth, std::vector<float> got, size_t K) {
    std::sort(truth.begin(), truth.end());
    if (truth.size() > K) truth.resize(K);
    std::vector<char> used(truth.size(), 0);
    size_t matched = 0;
    for (float d : got) {
        for (size_t i = 0; i < truth.size(); ++i) {
            if (!used[i] && std::abs(truth[i] - d) < 1e-3f) {
                used[i] = 1;
                ++matched;
                break;
            }
        }
    }
    return static_cast<double>(matched) / static_cast<double>(K);
}

// Build an HNSW index over `data` (N x dim) using T threads pulling work items off
// a shared atomic counter.
void build_concurrent(HNSWIndex& h, const std::vector<float>& data, size_t N,
                      size_t dim, size_t T) {
    std::atomic<size_t> next{0};
    auto worker = [&] {
        size_t i;
        while ((i = next.fetch_add(1)) < N) h.add(&data[i * dim]);
    };
    std::vector<std::thread> ts;
    for (size_t t = 0; t < T; ++t) ts.emplace_back(worker);
    for (auto& t : ts) t.join();
}

}  // namespace

// Serial add() must still hand out sequential internal ids — VDB's parallel arrays
// assert exactly this. (Concurrency must not break the single-threaded contract.)
TEST(add_returns_sequential_ids_serial) {
    const size_t N = 200, dim = 8;
    auto data = gen(N, dim, 3);
    HNSWConfig cfg;
    cfg.dim          = dim;
    cfg.max_elements = N;
    HNSWIndex h(cfg, metric_fn(Metric::L2));
    for (size_t i = 0; i < N; ++i) {
        InternalId id = h.add(&data[i * dim]);
        EXPECT(id == static_cast<InternalId>(i));
    }
    EXPECT(h.size() == N);
}

// Past the fixed reserve, add() must throw rather than reallocate (which would
// invalidate lock-free readers).
TEST(max_elements_is_enforced) {
    HNSWConfig cfg;
    cfg.dim          = 4;
    cfg.max_elements = 3;
    HNSWIndex h(cfg, metric_fn(Metric::L2));
    float v[4] = {1, 2, 3, 4};
    h.add(v);
    h.add(v);
    h.add(v);
    bool threw = false;
    try {
        h.add(v);
    } catch (const std::length_error&) {
        threw = true;
    }
    EXPECT(threw);
    EXPECT(h.size() == 3);
}

// Every vector inserted concurrently must be in the graph and findable: a
// self-query returns an exact (distance-0) hit. This proves no insert was lost and
// the graph stayed navigable under concurrent linking.
TEST(concurrent_add_all_present) {
    const size_t N = 4000, dim = 32, T = 8;
    auto data = gen(N, dim, 123);
    HNSWConfig cfg;
    cfg.dim          = dim;
    cfg.M            = 16;
    cfg.ef           = 100;
    cfg.max_elements = N;
    HNSWIndex h(cfg, metric_fn(Metric::L2));
    h.set_ef_search(64);

    build_concurrent(h, data, N, dim, T);

    ASSERT(h.size() == N);
    size_t exact_hits = 0;
    for (size_t i = 0; i < N; ++i) {
        auto r = h.search(&data[i * dim], 1);
        if (!r.empty() && r[0].second == 0.0f) ++exact_hits;  // L2² of a point with itself
    }
    // HNSW is approximate, but a self-query should almost always surface the point.
    EXPECT(exact_hits >= static_cast<size_t>(0.98 * N));
}

// A concurrently-built graph must have essentially the same recall as a
// single-threaded build: concurrency reorders inserts (a different but equally
// valid graph), it must not degrade quality.
TEST(concurrent_recall_matches_serial) {
    const size_t N = 3000, dim = 24, Q = 200, K = 10, T = 8;
    auto data    = gen(N, dim, 7);
    auto queries = gen(Q, dim, 99);

    // Exact oracle.
    BruteIndex brute(dim, metric_fn(Metric::L2));
    for (size_t i = 0; i < N; ++i) brute.add(&data[i * dim]);

    HNSWConfig cfg;
    cfg.dim          = dim;
    cfg.M            = 16;
    cfg.ef           = 200;
    cfg.max_elements = N;

    // Serial reference build.
    HNSWIndex serial(cfg, metric_fn(Metric::L2));
    serial.set_ef_search(100);
    for (size_t i = 0; i < N; ++i) serial.add(&data[i * dim]);

    // Concurrent build (same config, scrambled insert order).
    HNSWIndex concurrent(cfg, metric_fn(Metric::L2));
    concurrent.set_ef_search(100);
    build_concurrent(concurrent, data, N, dim, T);

    double serial_recall = 0.0, concurrent_recall = 0.0;
    for (size_t q = 0; q < Q; ++q) {
        const float* query = &queries[q * dim];

        std::vector<float> truth;
        for (auto& [id, d] : brute.search(query, K)) {
            (void)id;
            truth.push_back(d);
        }

        std::vector<float> sd, cd;
        for (auto& [id, d] : serial.search(query, K)) { (void)id; sd.push_back(d); }
        for (auto& [id, d] : concurrent.search(query, K)) { (void)id; cd.push_back(d); }

        serial_recall     += recall_by_distance(truth, sd, K);
        concurrent_recall += recall_by_distance(truth, cd, K);
    }
    serial_recall     /= Q;
    concurrent_recall /= Q;

    // Absolute floor (this config/data comfortably clears it single-threaded) and a
    // "no meaningful regression vs serial" bound.
    EXPECT(concurrent_recall >= 0.80);
    EXPECT(concurrent_recall >= serial_recall - 0.05);
}

// Readers searching *while* writers add must never crash, tear, or return garbage.
// This is the core data-race target: run it under TSan and ASan.
TEST(concurrent_search_during_add) {
    const size_t N = 3000, dim = 24, W = 4, R = 4;
    auto data = gen(N, dim, 55);
    HNSWConfig cfg;
    cfg.dim          = dim;
    cfg.M            = 16;
    cfg.ef           = 100;
    cfg.max_elements = N;
    HNSWIndex h(cfg, metric_fn(Metric::L2));
    h.set_ef_search(50);

    std::atomic<size_t> next{0};
    std::atomic<bool>   done{false};
    std::atomic<size_t> bad{0};

    auto writer = [&] {
        size_t i;
        while ((i = next.fetch_add(1)) < N) h.add(&data[i * dim]);
    };
    auto reader = [&](unsigned seed) {
        std::mt19937 rng(seed);
        std::uniform_int_distribution<size_t> pick(0, N - 1);
        while (!done.load(std::memory_order_relaxed)) {
            auto r = h.search(&data[pick(rng) * dim], 5);
            float prev = -1.0f;
            for (auto& [id, d] : r) {
                (void)id;
                if (d < 0.0f) ++bad;       // L2² is never negative
                if (d + 1e-6f < prev) ++bad;  // results must be sorted nearest-first
                prev = d;
            }
        }
    };

    std::vector<std::thread> writers, readers;
    for (size_t t = 0; t < W; ++t) writers.emplace_back(writer);
    for (size_t t = 0; t < R; ++t) readers.emplace_back(reader, 1000u + static_cast<unsigned>(t));
    for (auto& t : writers) t.join();
    done.store(true, std::memory_order_relaxed);
    for (auto& t : readers) t.join();

    EXPECT(bad.load() == 0);
    EXPECT(h.size() == N);
}
