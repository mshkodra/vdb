// PR 11: post-filter — VDB::search/search_hits with a Predicate, built on collect_'s
// generalized over-fetch (docs/plans/PR_STACK.md #11, docs/plans/FILTER_STRATEGY.md
// "Post-filter"). Run the index as usual, then keep only live hits matching the
// predicate too; the over-fetch margin is exact because ResolvedPredicate::allowlist
// is already live-filtered.
//
// PR 12: pre-filter — VDB::search_prefiltered/search_hits_prefiltered, which skip
// index_ entirely and brute-force scan just the allowlist (docs/plans/PR_STACK.md
// #12, docs/plans/FILTER_STRATEGY.md "Pre-filter").

#include "test.h"

#include "vdb.h"

#include <algorithm>
#include <random>
#include <stdexcept>
#include <vector>

using namespace vdb;

namespace {

// category (tag, unindexed postings) | price (int64, indexed) | score (float64, not
// indexed) — one indexed and one non-indexed numeric column, the same pairing
// test_metadata.cpp's resolve_schema() uses, to exercise both Range outcomes here too.
std::vector<AttrSpec> filter_schema() {
    return {{"category", AttrType::Tag},
            {"price", AttrType::Int64, /*indexed=*/true},
            {"score", AttrType::Float64, /*indexed=*/false}};
}

VDBConfig filter_config(IndexKind kind, size_t dim) {
    VDBConfig c;
    c.kind   = kind;
    c.dim    = dim;
    c.metric = Metric::L2;
    c.schema = filter_schema();
    return c;
}

Record row(const std::string& cat, int64_t price, double score) {
    Record r;
    r.attrs = {attr_tag(cat), attr_int(price), attr_float(score)};
    return r;
}

float l2sq(const std::vector<float>& a, const std::vector<float>& b) {
    float s = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        const float d = a[i] - b[i];
        s += d * d;
    }
    return s;
}

// Brute-force oracle over an explicit (ext_id, vector) live+matching set.
std::vector<ExternalId> oracle(
    const std::vector<std::pair<ExternalId, std::vector<float>>>& live,
    const std::vector<float>& q, size_t K) {
    std::vector<std::pair<float, ExternalId>> scored;
    for (const auto& [id, v] : live) scored.emplace_back(l2sq(q, v), id);
    std::sort(scored.begin(), scored.end());
    std::vector<ExternalId> out;
    for (size_t i = 0; i < scored.size() && i < K; ++i) out.push_back(scored[i].second);
    return out;
}

}  // namespace

TEST(post_filter_eq_returns_only_matching_live_hits) {
    VDB db(filter_config(IndexKind::Brute, 2));
    // "shoes" points cluster near the origin (closest to the query); "hats" points are
    // strictly closer still, so a plain search would return hats first — the predicate
    // must be what excludes them, not distance.
    const float q[2] = {0.0f, 0.0f};
    std::vector<std::pair<ExternalId, std::vector<float>>> shoes_live;

    for (int i = 0; i < 3; ++i) {
        const float v[2] = {0.1f * i, 0.1f * i};
        const ExternalId id = db.insert(v, row("hats", 1, 1.0));
        (void)id;
    }
    for (int i = 0; i < 3; ++i) {
        const float v[2] = {1.0f + 0.1f * i, 1.0f + 0.1f * i};
        const ExternalId id = db.insert(v, row("shoes", 1, 1.0));
        shoes_live.push_back({id, {v[0], v[1]}});
    }

    const auto want = oracle(shoes_live, {q[0], q[1]}, 2);
    const auto got  = db.search(q, 2, pred_eq(0, attr_tag("shoes")));
    EXPECT(got == want);
}

TEST(post_filter_excludes_tombstoned_matches_even_though_they_are_in_the_allowlist) {
    VDB db(filter_config(IndexKind::Brute, 2));
    const float q[2] = {0.0f, 0.0f};

    const ExternalId near_match = db.insert(q, row("shoes", 1, 1.0));
    const float far[2] = {5.0f, 5.0f};
    const ExternalId far_match = db.insert(far, row("shoes", 1, 1.0));

    ASSERT(db.remove(near_match));  // still in the Tag postings list, just dead

    const auto got = db.search(q, 5, pred_eq(0, attr_tag("shoes")));
    EXPECT(got == std::vector<ExternalId>{far_match});
}

TEST(post_filter_range_on_indexed_column) {
    VDB db(filter_config(IndexKind::Brute, 2));
    const float q[2] = {0.0f, 0.0f};
    std::vector<std::pair<ExternalId, std::vector<float>>> in_range;

    for (int i = 0; i < 6; ++i) {
        const float v[2] = {static_cast<float>(i), static_cast<float>(i)};
        const int64_t price = (i + 1) * 10;  // 10, 20, ..., 60
        const ExternalId id = db.insert(v, row("x", price, 1.0));
        if (price >= 20 && price <= 40) in_range.push_back({id, {v[0], v[1]}});
    }

    auto want = oracle(in_range, {q[0], q[1]}, 10);
    auto got  = db.search(q, 10, pred_range(1, attr_int(20), attr_int(40)));
    std::sort(want.begin(), want.end());
    std::sort(got.begin(), got.end());
    EXPECT(got == want);
}

TEST(post_filter_unresolved_range_on_non_indexed_column_throws) {
    VDB db(filter_config(IndexKind::Brute, 2));
    const float v[2] = {0.0f, 0.0f};
    db.insert(v, row("x", 1, 5.0));

    bool threw = false;
    try {
        db.search(v, 1, pred_range(2, attr_float(0.0), attr_float(10.0)));  // "score": not indexed
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    EXPECT(threw);

    threw = false;
    try {
        db.search_hits(v, 1, pred_range(2, attr_float(0.0), attr_float(10.0)));
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    EXPECT(threw);
}

TEST(post_filter_search_hits_carries_payload_for_matching_hits_only) {
    VDB db(filter_config(IndexKind::Brute, 2));
    const float near_[2] = {0.0f, 0.0f};
    const float far_[2]  = {9.0f, 9.0f};
    const float q[2]     = {0.1f, 0.1f};

    Record r1 = row("shoes", 1, 1.0);
    r1.payload = {'n'};
    Record r2 = row("hats", 1, 1.0);
    r2.payload = {'f', 'a', 'r'};
    db.insert(near_, r1);   // closer to q but wrong category
    const ExternalId match = db.insert(far_, r2);

    const auto hits = db.search_hits(q, 2, pred_eq(0, attr_tag("hats")));
    ASSERT(hits.size() == 1);
    EXPECT(hits[0].id == match);
    EXPECT(hits[0].payload == std::vector<uint8_t>({'f', 'a', 'r'}));
}

// HNSW: the same predicate-filtered search, over a graph index, with a predicate
// selective enough (1 of 4 well-separated clusters) that the generalized over-fetch
// margin — not just deleted_count_ — has to do real work to surface the true match.
TEST(post_filter_over_fetch_finds_selective_match_through_hnsw) {
    VDB db(filter_config(IndexKind::HNSW, 2));
    std::mt19937 rng(7);
    std::normal_distribution<float> jitter(0.0f, 0.05f);

    std::vector<std::pair<ExternalId, std::vector<float>>> target_live;
    for (int c = 0; c < 4; ++c) {
        for (int p = 0; p < 20; ++p) {
            const float v[2] = {static_cast<float>(c) * 10.0f + jitter(rng),
                                 static_cast<float>(c) * 10.0f + jitter(rng)};
            const ExternalId id = db.insert(v, row(c == 2 ? "target" : "other", 1, 1.0));
            if (c == 2) target_live.push_back({id, {v[0], v[1]}});
        }
    }

    const float q[2] = {20.0f, 20.0f};  // centred on cluster 2 ("target")
    const auto want = oracle(target_live, {q[0], q[1]}, 1);
    const auto got  = db.search(q, 1, pred_eq(0, attr_tag("target")));
    ASSERT(got.size() == 1);
    EXPECT(got == want);
}

TEST(prefilter_eq_returns_only_matching_live_hits_nearest_first) {
    VDB db(filter_config(IndexKind::Brute, 2));
    const float q[2] = {0.0f, 0.0f};
    std::vector<std::pair<ExternalId, std::vector<float>>> shoes_live;

    for (int i = 0; i < 3; ++i) {
        const float v[2] = {0.1f * i, 0.1f * i};
        db.insert(v, row("hats", 1, 1.0));
    }
    for (int i = 0; i < 3; ++i) {
        const float v[2] = {1.0f + 0.1f * i, 1.0f + 0.1f * i};
        const ExternalId id = db.insert(v, row("shoes", 1, 1.0));
        shoes_live.push_back({id, {v[0], v[1]}});
    }

    const auto want = oracle(shoes_live, {q[0], q[1]}, 2);
    const auto got  = db.search_prefiltered(q, 2, pred_eq(0, attr_tag("shoes")));
    EXPECT(got == want);
}

TEST(prefilter_excludes_tombstoned_matches_even_though_they_are_in_the_allowlist) {
    VDB db(filter_config(IndexKind::Brute, 2));
    const float q[2] = {0.0f, 0.0f};

    const ExternalId near_match = db.insert(q, row("shoes", 1, 1.0));
    const float far[2] = {5.0f, 5.0f};
    const ExternalId far_match = db.insert(far, row("shoes", 1, 1.0));

    ASSERT(db.remove(near_match));  // still in the Tag postings list, just dead

    const auto got = db.search_prefiltered(q, 5, pred_eq(0, attr_tag("shoes")));
    EXPECT(got == std::vector<ExternalId>{far_match});
}

TEST(prefilter_range_on_indexed_column) {
    VDB db(filter_config(IndexKind::Brute, 2));
    const float q[2] = {0.0f, 0.0f};
    std::vector<std::pair<ExternalId, std::vector<float>>> in_range;

    for (int i = 0; i < 6; ++i) {
        const float v[2] = {static_cast<float>(i), static_cast<float>(i)};
        const int64_t price = (i + 1) * 10;  // 10, 20, ..., 60
        const ExternalId id = db.insert(v, row("x", price, 1.0));
        if (price >= 20 && price <= 40) in_range.push_back({id, {v[0], v[1]}});
    }

    auto want = oracle(in_range, {q[0], q[1]}, 10);
    auto got  = db.search_prefiltered(q, 10, pred_range(1, attr_int(20), attr_int(40)));
    std::sort(want.begin(), want.end());
    std::sort(got.begin(), got.end());
    EXPECT(got == want);
}

TEST(prefilter_unresolved_range_on_non_indexed_column_throws) {
    VDB db(filter_config(IndexKind::Brute, 2));
    const float v[2] = {0.0f, 0.0f};
    db.insert(v, row("x", 1, 5.0));

    bool threw = false;
    try {
        db.search_prefiltered(v, 1, pred_range(2, attr_float(0.0), attr_float(10.0)));  // "score": not indexed
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    EXPECT(threw);

    threw = false;
    try {
        db.search_hits_prefiltered(v, 1, pred_range(2, attr_float(0.0), attr_float(10.0)));
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    EXPECT(threw);
}

TEST(prefilter_search_hits_carries_payload_for_matching_hits_only) {
    VDB db(filter_config(IndexKind::Brute, 2));
    const float near_[2] = {0.0f, 0.0f};
    const float far_[2]  = {9.0f, 9.0f};
    const float q[2]     = {0.1f, 0.1f};

    Record r1 = row("shoes", 1, 1.0);
    r1.payload = {'n'};
    Record r2 = row("hats", 1, 1.0);
    r2.payload = {'f', 'a', 'r'};
    db.insert(near_, r1);   // closer to q but wrong category
    const ExternalId match = db.insert(far_, r2);

    const auto hits = db.search_hits_prefiltered(q, 2, pred_eq(0, attr_tag("hats")));
    ASSERT(hits.size() == 1);
    EXPECT(hits[0].id == match);
    EXPECT(hits[0].payload == std::vector<uint8_t>({'f', 'a', 'r'}));
}

// Pre-filter skips index_ entirely, so it must be exact even over HNSW, where a
// selective predicate makes plain post-filter/graph traversal recall-lossy. This is
// the case pre-filter is *for* (FILTER_STRATEGY.md: "wins when the matching set is
// small") — same scene as the post-filter HNSW test above, but pre-filter must find
// the true nearest match every time, not just "with high probability".
TEST(prefilter_is_exact_over_hnsw_for_a_selective_predicate) {
    VDB db(filter_config(IndexKind::HNSW, 2));
    std::mt19937 rng(13);
    std::normal_distribution<float> jitter(0.0f, 0.05f);

    std::vector<std::pair<ExternalId, std::vector<float>>> target_live;
    for (int c = 0; c < 4; ++c) {
        for (int p = 0; p < 20; ++p) {
            const float v[2] = {static_cast<float>(c) * 10.0f + jitter(rng),
                                 static_cast<float>(c) * 10.0f + jitter(rng)};
            const ExternalId id = db.insert(v, row(c == 3 ? "target" : "other", 1, 1.0));
            if (c == 3) target_live.push_back({id, {v[0], v[1]}});
        }
    }

    const float q[2] = {30.0f, 30.0f};  // centred on cluster 3 ("target")
    const auto want = oracle(target_live, {q[0], q[1]}, 3);
    const auto got  = db.search_prefiltered(q, 3, pred_eq(0, attr_tag("target")));
    EXPECT(got == want);
}

// search_auto/search_hits_auto (filter_planner's plan_strategy wired into VDB, per
// docs/results/filter_findings.md's Run 3 verdict): both strategies are exact, so
// whichever collect_auto_ picks under the default calibration, the result must match
// the same oracle post-filter/pre-filter already do. Which branch actually fires
// isn't observable from outside VDB (no calibration setter on purpose — the design
// call was "baked-in constants only"), so these lean on the same scenes above to
// exercise the dispatch end-to-end rather than asserting on a hidden strategy pick.

TEST(auto_filter_eq_returns_only_matching_live_hits_nearest_first) {
    VDB db(filter_config(IndexKind::Brute, 2));
    const float q[2] = {0.0f, 0.0f};
    std::vector<std::pair<ExternalId, std::vector<float>>> shoes_live;

    for (int i = 0; i < 3; ++i) {
        const float v[2] = {0.1f * i, 0.1f * i};
        db.insert(v, row("hats", 1, 1.0));
    }
    for (int i = 0; i < 3; ++i) {
        const float v[2] = {1.0f + 0.1f * i, 1.0f + 0.1f * i};
        const ExternalId id = db.insert(v, row("shoes", 1, 1.0));
        shoes_live.push_back({id, {v[0], v[1]}});
    }

    const auto want = oracle(shoes_live, {q[0], q[1]}, 2);
    const auto got  = db.search_auto(q, 2, pred_eq(0, attr_tag("shoes")));
    EXPECT(got == want);
}

TEST(auto_filter_excludes_tombstoned_matches_even_though_they_are_in_the_allowlist) {
    VDB db(filter_config(IndexKind::Brute, 2));
    const float q[2] = {0.0f, 0.0f};

    const ExternalId near_match = db.insert(q, row("shoes", 1, 1.0));
    const float far[2] = {5.0f, 5.0f};
    const ExternalId far_match = db.insert(far, row("shoes", 1, 1.0));

    ASSERT(db.remove(near_match));  // still in the Tag postings list, just dead

    const auto got = db.search_auto(q, 5, pred_eq(0, attr_tag("shoes")));
    EXPECT(got == std::vector<ExternalId>{far_match});
}

TEST(auto_filter_range_on_indexed_column) {
    VDB db(filter_config(IndexKind::Brute, 2));
    const float q[2] = {0.0f, 0.0f};
    std::vector<std::pair<ExternalId, std::vector<float>>> in_range;

    for (int i = 0; i < 6; ++i) {
        const float v[2] = {static_cast<float>(i), static_cast<float>(i)};
        const int64_t price = (i + 1) * 10;  // 10, 20, ..., 60
        const ExternalId id = db.insert(v, row("x", price, 1.0));
        if (price >= 20 && price <= 40) in_range.push_back({id, {v[0], v[1]}});
    }

    auto want = oracle(in_range, {q[0], q[1]}, 10);
    auto got  = db.search_auto(q, 10, pred_range(1, attr_int(20), attr_int(40)));
    std::sort(want.begin(), want.end());
    std::sort(got.begin(), got.end());
    EXPECT(got == want);
}

TEST(auto_filter_unresolved_range_on_non_indexed_column_throws) {
    VDB db(filter_config(IndexKind::Brute, 2));
    const float v[2] = {0.0f, 0.0f};
    db.insert(v, row("x", 1, 5.0));

    bool threw = false;
    try {
        db.search_auto(v, 1, pred_range(2, attr_float(0.0), attr_float(10.0)));  // "score": not indexed
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    EXPECT(threw);

    threw = false;
    try {
        db.search_hits_auto(v, 1, pred_range(2, attr_float(0.0), attr_float(10.0)));
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    EXPECT(threw);
}

TEST(auto_filter_search_hits_carries_payload_for_matching_hits_only) {
    VDB db(filter_config(IndexKind::Brute, 2));
    const float near_[2] = {0.0f, 0.0f};
    const float far_[2]  = {9.0f, 9.0f};
    const float q[2]     = {0.1f, 0.1f};

    Record r1 = row("shoes", 1, 1.0);
    r1.payload = {'n'};
    Record r2 = row("hats", 1, 1.0);
    r2.payload = {'f', 'a', 'r'};
    db.insert(near_, r1);   // closer to q but wrong category
    const ExternalId match = db.insert(far_, r2);

    const auto hits = db.search_hits_auto(q, 2, pred_eq(0, attr_tag("hats")));
    ASSERT(hits.size() == 1);
    EXPECT(hits[0].id == match);
    EXPECT(hits[0].payload == std::vector<uint8_t>({'f', 'a', 'r'}));
}

// Same selective-predicate-over-HNSW scene as the post-filter and pre-filter tests
// above: whichever strategy collect_auto_ picks, it must still find the true nearest
// match exactly, not "with high probability".
TEST(auto_filter_is_exact_over_hnsw_for_a_selective_predicate) {
    VDB db(filter_config(IndexKind::HNSW, 2));
    std::mt19937 rng(17);
    std::normal_distribution<float> jitter(0.0f, 0.05f);

    std::vector<std::pair<ExternalId, std::vector<float>>> target_live;
    for (int c = 0; c < 4; ++c) {
        for (int p = 0; p < 20; ++p) {
            const float v[2] = {static_cast<float>(c) * 10.0f + jitter(rng),
                                 static_cast<float>(c) * 10.0f + jitter(rng)};
            const ExternalId id = db.insert(v, row(c == 1 ? "target" : "other", 1, 1.0));
            if (c == 1) target_live.push_back({id, {v[0], v[1]}});
        }
    }

    const float q[2] = {10.0f, 10.0f};  // centred on cluster 1 ("target")
    const auto want = oracle(target_live, {q[0], q[1]}, 1);
    const auto got  = db.search_auto(q, 1, pred_eq(0, attr_tag("target")));
    ASSERT(got.size() == 1);
    EXPECT(got == want);
}
