#include "test.h"

#include "brute_index.h"

#include <cmath>

namespace {

bool near(float a, float b, float eps = 1e-4f) {
    return std::fabs(a - b) <= eps;
}

DistanceFn l2sq() {
    return [](const float* a, const float* b, size_t d) {
        float s = 0.0f;
        for (size_t i = 0; i < d; ++i) {
            float x = a[i] - b[i];
            s += x * x;
        }
        return s;
    };
}

}

TEST(brute_empty_index) {
    vdb::BruteIndex idx(4, l2sq());
    EXPECT(idx.dim() == 4);
    EXPECT(idx.size() == 0);
    float q[] = {0.0f, 0.0f, 0.0f, 0.0f};
    EXPECT(idx.search(q, 5).empty());
}

TEST(brute_add_returns_sequential_ids) {
    vdb::BruteIndex idx(2, l2sq());
    float a[] = {0.0f, 0.0f};
    float b[] = {1.0f, 1.0f};
    float c[] = {2.0f, 2.0f};
    EXPECT(idx.add(a) == 0u);
    EXPECT(idx.add(b) == 1u);
    EXPECT(idx.add(c) == 2u);
    EXPECT(idx.size() == 3);
}

namespace {
vdb::BruteIndex fixture() {
    vdb::BruteIndex idx(2, l2sq());
    float p0[] = {0.0f, 0.0f};
    float p1[] = {1.0f, 0.0f};
    float p2[] = {0.0f, 2.0f};
    float p3[] = {3.0f, 0.0f};
    idx.add(p0);
    idx.add(p1);
    idx.add(p2);
    idx.add(p3);
    return idx;
}
}

TEST(brute_returns_nearest_first) {
    auto idx = fixture();
    float q[] = {0.0f, 0.0f};
    auto res = idx.search(q, 3);
    ASSERT(res.size() == 3);
    EXPECT(res[0].first == 0u && near(res[0].second, 0.0f));
    EXPECT(res[1].first == 1u && near(res[1].second, 1.0f));
    EXPECT(res[2].first == 2u && near(res[2].second, 4.0f));
}

TEST(brute_results_ordered_by_distance) {
    auto idx = fixture();
    float q[] = {0.5f, 0.5f};
    auto res = idx.search(q, 4);
    ASSERT(res.size() == 4);
    for (size_t i = 1; i < res.size(); ++i) {
        EXPECT(res[i - 1].second <= res[i].second);
    }
}

TEST(brute_self_query_finds_self_at_zero) {
    auto idx = fixture();
    float q[] = {0.0f, 2.0f};
    auto res = idx.search(q, 1);
    ASSERT(res.size() == 1);
    EXPECT(res[0].first == 2u);
    EXPECT(near(res[0].second, 0.0f));
}

TEST(brute_near_miss_query) {
    auto idx = fixture();
    float q[] = {0.9f, 0.0f};
    auto res = idx.search(q, 1);
    ASSERT(res.size() == 1);
    EXPECT(res[0].first == 1u);
    EXPECT(near(res[0].second, 0.01f));
}

TEST(brute_k_larger_than_size_returns_all) {
    auto idx = fixture();
    float q[] = {0.0f, 0.0f};
    auto res = idx.search(q, 100);
    EXPECT(res.size() == 4);
}

TEST(brute_k_zero_returns_empty) {
    auto idx = fixture();
    float q[] = {0.0f, 0.0f};
    EXPECT(idx.search(q, 0).empty());
}
