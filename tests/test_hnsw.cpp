#include "test.h"

#include <hnswindex.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <utility>
#include <vector>

namespace {

float l2_sq(const float* a, const float* b, size_t d) {
    float s = 0.0f;
    for (size_t i = 0; i < d; ++i) {
        float x = a[i] - b[i];
        s += x * x;
    }
    return s;
}

HNSWConfig small_cfg(size_t dim) {
    HNSWConfig c;
    c.dim   = dim;
    c.M     = 8;
    c.Mmax  = 8;
    c.Mmax0 = 16;
    c.ef    = 50;
    return c;
}

std::vector<std::vector<float>> random_vectors(size_t n, size_t dim, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    std::vector<std::vector<float>> v(n, std::vector<float>(dim));
    for (auto& vec : v) for (auto& x : vec) x = u(rng);
    return v;
}

std::vector<InternalId> brute_topk(const std::vector<std::vector<float>>& data,
                                   const float* q, size_t dim, size_t K) {
    std::vector<std::pair<float, InternalId>> scored;
    scored.reserve(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        scored.emplace_back(l2_sq(q, data[i].data(), dim), (InternalId)i);
    }
    std::partial_sort(scored.begin(), scored.begin() + K, scored.end(),
        [](const auto& a, const auto& b){ return a.first < b.first; });
    std::vector<InternalId> out;
    out.reserve(K);
    for (size_t i = 0; i < K; ++i) out.push_back(scored[i].second);
    return out;
}

int recall(const std::vector<InternalId>& got, const std::vector<InternalId>& truth) {
    int hits = 0;
    for (auto g : got) {
        for (auto t : truth) if (g == t) { ++hits; break; }
    }
    return hits;
}

}  // namespace

TEST(hnsw_empty_search_returns_empty) {
    HNSWIndex idx(small_cfg(4), &l2_sq);
    float q[4] = {0, 0, 0, 0};
    auto r = idx.search(q, 5, 10);
    EXPECT(r.empty());
    EXPECT(idx.size() == 0);
}

TEST(hnsw_single_insert_then_search) {
    HNSWIndex idx(small_cfg(4), &l2_sq);
    float v[4] = {1, 2, 3, 4};
    auto id = idx.insert(v);
    EXPECT(id == 0);
    EXPECT(idx.size() == 1);

    auto r = idx.search(v, 1, 10);
    ASSERT(r.size() == 1);
    EXPECT(r[0] == 0);
}

TEST(hnsw_get_returns_stored_vector) {
    HNSWIndex idx(small_cfg(3), &l2_sq);
    float v[3] = {0.5f, -1.0f, 2.25f};
    auto id = idx.insert(v);
    const float* got = idx.get(id);
    EXPECT(got[0] == 0.5f);
    EXPECT(got[1] == -1.0f);
    EXPECT(got[2] == 2.25f);
}

TEST(hnsw_query_with_training_vector_returns_itself) {
    const size_t dim = 8;
    const size_t N = 500;
    HNSWIndex idx(small_cfg(dim), &l2_sq);
    auto data = random_vectors(N, dim, 7);
    for (const auto& v : data) idx.insert(v.data());

    EXPECT(idx.size() == N);

    auto r = idx.search(data[42].data(), 1, 50);
    ASSERT(!r.empty());
    EXPECT(r[0] == 42);
}

TEST(hnsw_recall_against_brute_force) {
    const size_t dim = 16;
    const size_t N = 1000;
    const size_t K = 10;

    HNSWIndex idx(small_cfg(dim), &l2_sq);
    auto data = random_vectors(N, dim, 11);
    for (const auto& v : data) idx.insert(v.data());

    std::mt19937 rng(99);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);

    int total_hits = 0;
    const int n_queries = 20;
    for (int q = 0; q < n_queries; ++q) {
        std::vector<float> qv(dim);
        for (auto& x : qv) x = u(rng);

        auto got   = idx.search(qv.data(), K, 100);
        auto truth = brute_topk(data, qv.data(), dim, K);
        total_hits += recall(got, truth);
    }

    const int total = n_queries * (int)K;
    // Expect at least 90% recall@10 with ef=100 on this small, random dataset.
    EXPECT(total_hits >= total * 9 / 10);
}

TEST(hnsw_search_K_larger_than_index) {
    HNSWIndex idx(small_cfg(2), &l2_sq);
    float a[2] = {0, 0};
    float b[2] = {1, 1};
    idx.insert(a);
    idx.insert(b);

    float q[2] = {0.1f, 0.1f};
    auto r = idx.search(q, 100, 100);
    EXPECT(r.size() == 2);
}
