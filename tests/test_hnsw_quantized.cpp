// int8 scalar quantization for HNSWIndex (HNSWIndex<Dist, Elem>'s Elem parameter,
// see hnsw_index.h's class comment). Elem=float's existing behavior is exercised
// throughout the rest of the suite (test_concurrency.cpp, bench/sift.cpp) and is
// unchanged by this parameter's addition — these tests are scoped to the new
// HNSWIndex<L2Int8, int8_t> path specifically: the "must train before insert"
// contract, that quantization still finds the right neighbourhood on well-
// separated data, and that serialize/deserialize round-trips the calibrated scale.

#include "test.h"

#include "distance.h"
#include "hnsw_index.h"
#include "serialize.h"

#include <random>
#include <stdexcept>
#include <vector>

using namespace vdb;

namespace {

constexpr size_t DIM = 8;  // small, deterministic, fast — not SIFT-scale

// 3 well-separated clusters (centers 10 apart) x `per_cluster` points, low
// Gaussian noise — small enough that quantization noise shouldn't be able to
// confuse which cluster a point belongs to, so exact-cluster-membership is a
// meaningful correctness check, not just "returns *something*."
std::vector<std::vector<float>> clustered_points(int per_cluster, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> noise(0.0f, 0.05f);
    std::vector<std::vector<float>> pts;
    for (int c = 0; c < 3; ++c) {
        for (int p = 0; p < per_cluster; ++p) {
            std::vector<float> v(DIM);
            for (size_t d = 0; d < DIM; ++d) v[d] = static_cast<float>(c) * 10.0f + noise(rng);
            pts.push_back(std::move(v));
        }
    }
    return pts;
}

std::vector<float> flatten(const std::vector<std::vector<float>>& pts) {
    std::vector<float> out;
    out.reserve(pts.size() * DIM);
    for (const auto& v : pts) out.insert(out.end(), v.begin(), v.end());
    return out;
}

}  // namespace

TEST(quantized_hnsw_throws_on_insert_before_train) {
    HNSWIndex<L2Int8, int8_t> h({DIM});
    float v[DIM] = {1, 2, 3, 4, 5, 6, 7, 8};

    bool threw = false;
    try {
        h.add(v);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    EXPECT(threw);
}

TEST(quantized_hnsw_allows_insert_after_train) {
    const auto pts = clustered_points(5, 1);
    const auto flat = flatten(pts);

    HNSWIndex<L2Int8, int8_t> h({DIM});
    h.train(flat.data(), pts.size());
    for (const auto& v : pts) h.add(v.data());  // must not throw
    EXPECT(h.size() == pts.size());
}

TEST(quantized_hnsw_finds_the_right_cluster) {
    const int per_cluster = 5;
    const auto pts  = clustered_points(per_cluster, 2);
    const auto flat = flatten(pts);

    HNSWIndex<L2Int8, int8_t> h({DIM});
    h.train(flat.data(), pts.size());
    for (const auto& v : pts) h.add(v.data());

    // Query at cluster 1's exact center (points [5,10) are cluster 1) — the
    // nearest result must come from that cluster, not 0 or 2 (10 units away,
    // far beyond what int8 quantization noise on a single-digit-magnitude
    // dataset could plausibly confuse).
    std::vector<float> q(DIM, 10.0f);
    const auto results = h.search(q.data(), 1);
    ASSERT(results.size() == 1);
    const InternalId id = results[0].first;
    EXPECT(id >= static_cast<InternalId>(per_cluster) &&
          id < static_cast<InternalId>(2 * per_cluster));
}

TEST(quantized_hnsw_serialize_round_trips_scale_and_results) {
    const auto pts  = clustered_points(4, 3);
    const auto flat = flatten(pts);

    HNSWIndex<L2Int8, int8_t> h({DIM});
    h.train(flat.data(), pts.size());
    for (const auto& v : pts) h.add(v.data());

    std::vector<uint8_t> buf;
    h.serialize(buf);

    HNSWIndex<L2Int8, int8_t> h2({DIM});
    Reader r(buf.data(), buf.size());
    h2.deserialize(r);

    // If scale_ didn't round-trip, this would throw ("train() must run first") or
    // silently quantize against the wrong scale — either way the results below
    // would diverge from h's.
    std::vector<float> q(DIM, 10.0f);
    const auto want = h.search(q.data(), 3);
    const auto got  = h2.search(q.data(), 3);
    EXPECT(got == want);
}

TEST(quantized_hnsw_empty_index_returns_no_results) {
    HNSWIndex<L2Int8, int8_t> h({DIM});
    float q[DIM] = {0, 0, 0, 0, 0, 0, 0, 0};
    EXPECT(h.search(q, 5).empty());
}
