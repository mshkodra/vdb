#include "test.h"

#include "vdb.h"

#include <algorithm>
#include <random>
#include <unordered_set>
#include <vector>

using vdb::VDB;
using vdb::VDBConfig;
using vdb::IndexKind;
using vdb::Metric;

namespace {

VDBConfig cfg(IndexKind kind, size_t dim) {
    VDBConfig c;
    c.kind = kind;
    c.dim = dim;
    c.metric = Metric::L2;
    return c;
}

// L2^2 between two raw vectors — the same order the indexes rank by.
float l2sq(const std::vector<float>& a, const std::vector<float>& b) {
    float s = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        const float d = a[i] - b[i];
        s += d * d;
    }
    return s;
}

// Reproducible clustered-ish data: k blobs of `per` points around integer grid
// centres. Well separated so an exact index has an unambiguous top-K.
std::vector<std::vector<float>> make_data(size_t k, size_t per, size_t dim,
                                          unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> jitter(0.0f, 0.15f);
    std::vector<std::vector<float>> out;
    for (size_t c = 0; c < k; ++c) {
        std::vector<float> centre(dim, static_cast<float>(c) * 10.0f);
        for (size_t p = 0; p < per; ++p) {
            std::vector<float> v(dim);
            for (size_t d = 0; d < dim; ++d) v[d] = centre[d] + jitter(rng);
            out.push_back(std::move(v));
        }
    }
    return out;
}

// Brute-force oracle over an explicit (ext_id, vector) live set.
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

// ---------------------------------------------------------------------------
// Identity: external ids are stable, distinct, and translate back correctly.
// ---------------------------------------------------------------------------

TEST(lifecycle_insert_assigns_distinct_ids) {
    VDB db(cfg(IndexKind::Brute, 3));
    auto data = make_data(3, 4, 3, 1);
    std::unordered_set<ExternalId> ids;
    for (const auto& v : data) ids.insert(db.insert(v.data()));
    EXPECT(ids.size() == data.size());     // all distinct
    EXPECT(db.size() == data.size());
    EXPECT(db.deleted_count() == 0);
    for (auto id : ids) EXPECT(db.contains(id));
}

TEST(lifecycle_self_query_returns_self) {
    VDB db(cfg(IndexKind::Brute, 4));
    auto data = make_data(4, 3, 4, 2);
    std::vector<ExternalId> ids;
    for (const auto& v : data) ids.push_back(db.insert(v.data()));
    for (size_t i = 0; i < data.size(); ++i) {
        auto res = db.search(data[i].data(), 1);
        ASSERT(res.size() == 1);
        EXPECT(res[0] == ids[i]);
    }
}

// ---------------------------------------------------------------------------
// Deletes: tombstoned vectors vanish from results and from contains().
// ---------------------------------------------------------------------------

TEST(lifecycle_delete_excludes_from_search) {
    VDB db(cfg(IndexKind::Brute, 2));
    std::vector<std::vector<float>> data = {
        {0.0f, 0.0f}, {1.0f, 0.0f}, {2.0f, 0.0f}, {3.0f, 0.0f}};
    std::vector<ExternalId> ids;
    for (const auto& v : data) ids.push_back(db.insert(v.data()));

    // Nearest to origin is id[0]; delete it and the next-nearest must surface.
    std::vector<float> q = {0.0f, 0.0f};
    ASSERT(db.search(q.data(), 1)[0] == ids[0]);

    EXPECT(db.remove(ids[0]));
    EXPECT(!db.contains(ids[0]));
    EXPECT(db.size() == 3);
    EXPECT(db.deleted_count() == 1);

    auto res = db.search(q.data(), 1);
    ASSERT(res.size() == 1);
    EXPECT(res[0] == ids[1]);              // id[0] is gone, id[1] steps up

    // A deleted id never reappears, even asking for everything.
    auto all = db.search(q.data(), 10);
    EXPECT(std::find(all.begin(), all.end(), ids[0]) == all.end());
    EXPECT(all.size() == 3);
}

TEST(lifecycle_remove_unknown_is_noop) {
    VDB db(cfg(IndexKind::Brute, 2));
    std::vector<float> v = {1.0f, 2.0f};
    auto id = db.insert(v.data());
    EXPECT(!db.remove(id + 999));          // unknown id
    EXPECT(db.remove(id));
    EXPECT(!db.remove(id));                // double-remove is a no-op
    EXPECT(db.size() == 0);
    EXPECT(db.deleted_count() == 1);
}

// Over-fetch correctness: after deleting many, the K live nearest must exactly
// match a brute-force oracle over the surviving set.
TEST(lifecycle_search_matches_oracle_after_deletes) {
    const size_t dim = 8;
    VDB db(cfg(IndexKind::Brute, dim));
    auto data = make_data(5, 20, dim, 7);      // 100 points
    std::vector<ExternalId> ids;
    for (const auto& v : data) ids.push_back(db.insert(v.data()));

    // Delete a scattered ~40% of them.
    std::mt19937 rng(99);
    std::vector<std::pair<ExternalId, std::vector<float>>> live;
    for (size_t i = 0; i < data.size(); ++i) {
        if (rng() % 5 < 2) {                   // ~40%
            EXPECT(db.remove(ids[i]));
        } else {
            live.emplace_back(ids[i], data[i]);
        }
    }
    EXPECT(db.size() == live.size());

    std::vector<float> q(dim, 12.0f);          // near cluster 1
    const size_t K = 10;
    auto got = db.search(q.data(), K);
    auto want = oracle(live, q, K);
    ASSERT(got.size() == want.size());
    for (size_t i = 0; i < want.size(); ++i) EXPECT(got[i] == want[i]);
}

// ---------------------------------------------------------------------------
// Updates: same external id, new location.
// ---------------------------------------------------------------------------

TEST(lifecycle_update_moves_vector_under_stable_id) {
    VDB db(cfg(IndexKind::Brute, 2));
    std::vector<float> a = {0.0f, 0.0f};
    std::vector<float> far = {5.0f, 5.0f};
    auto id = db.insert(a.data());
    auto other = db.insert(far.data());
    (void)other;

    std::vector<float> origin = {0.0f, 0.0f};
    ASSERT(db.search(origin.data(), 1)[0] == id);   // starts near origin

    std::vector<float> b = {100.0f, 100.0f};
    EXPECT(db.update(id, b.data()));
    EXPECT(db.contains(id));
    EXPECT(db.size() == 2);                          // one out, one in
    EXPECT(db.deleted_count() == 1);                 // old location tombstoned

    // Now id is the far one: no longer nearest origin, but nearest to (100,100).
    EXPECT(db.search(origin.data(), 1)[0] != id);
    std::vector<float> q = {100.0f, 100.0f};
    ASSERT(db.search(q.data(), 1).size() == 1);
    EXPECT(db.search(q.data(), 1)[0] == id);
}

TEST(lifecycle_update_unknown_is_noop) {
    VDB db(cfg(IndexKind::Brute, 2));
    std::vector<float> v = {1.0f, 1.0f};
    EXPECT(!db.update(12345, v.data()));
    EXPECT(db.size() == 0);
}

// ---------------------------------------------------------------------------
// Compaction: reclaims tombstone space, preserves external ids and results.
// ---------------------------------------------------------------------------

TEST(lifecycle_compact_preserves_live_ids_and_results) {
    const size_t dim = 6;
    VDB db(cfg(IndexKind::Brute, dim));
    auto data = make_data(4, 15, dim, 21);
    std::vector<ExternalId> ids;
    for (const auto& v : data) ids.push_back(db.insert(v.data()));

    // Remember the live set and a set of query results before compaction.
    std::mt19937 rng(5);
    std::unordered_set<ExternalId> live_ids;
    for (size_t i = 0; i < data.size(); ++i) {
        if (rng() % 2 == 0) db.remove(ids[i]);
        else live_ids.insert(ids[i]);
    }

    std::vector<std::vector<float>> queries = {
        std::vector<float>(dim, 0.0f), std::vector<float>(dim, 10.0f),
        std::vector<float>(dim, 20.0f)};
    std::vector<std::vector<ExternalId>> before;
    for (const auto& q : queries) before.push_back(db.search(q.data(), 5));

    const size_t live_before = db.size();
    db.compact();

    // Space reclaimed, counts intact.
    EXPECT(db.size() == live_before);
    EXPECT(db.deleted_count() == 0);

    // Every live id still resolves; deleted ids stay gone.
    for (auto id : ids) {
        if (live_ids.count(id)) EXPECT(db.contains(id));
        else EXPECT(!db.contains(id));
    }

    // Compaction is recall-neutral: identical results for the same queries.
    for (size_t i = 0; i < queries.size(); ++i) {
        auto after = db.search(queries[i].data(), 5);
        ASSERT(after.size() == before[i].size());
        for (size_t j = 0; j < after.size(); ++j) EXPECT(after[j] == before[i][j]);
    }
}

TEST(lifecycle_insert_after_compact_keeps_unique_ids) {
    VDB db(cfg(IndexKind::Brute, 2));
    std::vector<float> a = {0.0f, 0.0f}, b = {1.0f, 1.0f}, c = {2.0f, 2.0f};
    auto ia = db.insert(a.data());
    auto ib = db.insert(b.data());
    db.remove(ia);
    db.compact();

    // next_ext_id_ must not be rewound by compaction, or we'd collide with ib.
    auto ic = db.insert(c.data());
    EXPECT(ic != ib);
    EXPECT(ic != ia);
    EXPECT(db.contains(ib));
    EXPECT(db.contains(ic));
    EXPECT(db.size() == 2);
}

// Churn: many delete/insert cycles then compact; the live set stays exactly right.
TEST(lifecycle_churn_then_compact) {
    const size_t dim = 4;
    VDB db(cfg(IndexKind::Brute, dim));
    std::mt19937 rng(2024);
    std::normal_distribution<float> g(0.0f, 1.0f);

    std::unordered_set<ExternalId> alive;
    std::vector<ExternalId> alive_list;
    auto make = [&] {
        std::vector<float> v(dim);
        for (auto& x : v) x = g(rng);
        return v;
    };

    for (int step = 0; step < 300; ++step) {
        if (alive.empty() || rng() % 3 != 0) {
            auto v = make();
            auto id = db.insert(v.data());
            alive.insert(id);
            alive_list.push_back(id);
        } else {
            // remove a random live id
            auto idx = rng() % alive_list.size();
            auto id = alive_list[idx];
            if (alive.count(id)) {
                EXPECT(db.remove(id));
                alive.erase(id);
            }
        }
    }

    EXPECT(db.size() == alive.size());
    db.compact();
    EXPECT(db.size() == alive.size());
    EXPECT(db.deleted_count() == 0);
    for (auto id : alive) EXPECT(db.contains(id));
}

// ---------------------------------------------------------------------------
// HNSW: tombstoned nodes stay in the graph (for connectivity) but never appear
// in results. This exercises the "traverse-through-deleted" path.
// ---------------------------------------------------------------------------

TEST(lifecycle_hnsw_deletes_excluded_but_recall_holds) {
    const size_t dim = 16;
    VDB db(cfg(IndexKind::HNSW, dim));
    auto data = make_data(6, 40, dim, 314);     // 240 points, tight clusters
    std::vector<ExternalId> ids;
    for (const auto& v : data) ids.push_back(db.insert(v.data()));

    // Delete half at random.
    std::mt19937 rng(11);
    std::unordered_set<ExternalId> deleted;
    std::vector<std::pair<ExternalId, std::vector<float>>> live;
    for (size_t i = 0; i < data.size(); ++i) {
        if (rng() % 2 == 0) { db.remove(ids[i]); deleted.insert(ids[i]); }
        else live.emplace_back(ids[i], data[i]);
    }

    // Query each cluster centre; deleted ids must never show up, and the top hit
    // should match the exact-live oracle (clusters are well separated).
    size_t hits = 0, total = 0;
    for (size_t c = 0; c < 6; ++c) {
        std::vector<float> q(dim, static_cast<float>(c) * 10.0f);
        auto res = db.search(q.data(), 5);
        for (auto id : res) EXPECT(deleted.count(id) == 0);
        auto want = oracle(live, q, 1);
        if (!res.empty() && !want.empty()) {
            ++total;
            if (res[0] == want[0]) ++hits;
        }
    }
    EXPECT(total > 0);
    EXPECT(hits == total);   // exact top-1 on well-separated clusters
}
