#include "test.h"

#include <vdb.h>

#include <algorithm>
#include <vector>

// Phase 1 — Identity & Lifecycle.
// These tests exercise the external-id indirection, tombstone deletes, and updates.
// They deliberately use well-separated points so nearest-neighbour answers are
// deterministic and don't depend on HNSW's approximate behaviour.

namespace {

VDBConfig cfg2d() {
    VDBConfig cfg;
    cfg.hnsw.dim = 2;
    return cfg;
}

bool contains_id(const std::vector<ExternalId>& v, ExternalId id) {
    return std::find(v.begin(), v.end(), id) != v.end();
}

}  // namespace

TEST(vdb_external_ids_are_stable_and_start_at_one) {
    VDB db(cfg2d());
    float a[2] = {0, 0};
    float b[2] = {1, 1};
    float c[2] = {2, 2};

    EXPECT(db.insert(a) == 1);
    EXPECT(db.insert(b) == 2);
    EXPECT(db.insert(c) == 3);
    EXPECT(db.size() == 3);
}

TEST(vdb_remove_makes_key_disappear) {
    VDB db(cfg2d());
    float v[2] = {5, 5};
    auto id = db.insert(v);

    EXPECT(db.contains(id));
    EXPECT(db.get(id) != nullptr);
    EXPECT(db.size() == 1);

    EXPECT(db.remove(id) == true);

    EXPECT(db.contains(id) == false);
    EXPECT(db.get(id) == nullptr);   // key reads as gone immediately
    EXPECT(db.size() == 0);          // size tracks LIVE vectors only
}

TEST(vdb_remove_unknown_id_returns_false) {
    VDB db(cfg2d());
    EXPECT(db.remove(999) == false);
    float v[2] = {0, 0};
    auto id = db.insert(v);
    EXPECT(db.remove(id) == true);
    EXPECT(db.remove(id) == false);  // double-delete is a no-op
}

TEST(vdb_deleted_node_excluded_from_search_but_graph_stays_navigable) {
    // Five points strung out along the x-axis. Query at x=0 ranks them 1<2<3<4<5.
    VDB db(cfg2d());
    std::vector<ExternalId> ids;
    for (int i = 0; i < 5; ++i) {
        float v[2] = {static_cast<float>(i), 0};
        ids.push_back(db.insert(v));
    }

    float q[2] = {0, 0};

    // Nearest is the x=0 point (ids[0]).
    auto before = db.search(q, 1, 16);
    ASSERT(before.size() == 1);
    EXPECT(before[0] == ids[0]);

    // Delete the nearest. Its graph node is tombstoned, not unlinked, so the search
    // can still traverse *through* it to reach the rest of the graph.
    EXPECT(db.remove(ids[0]) == true);

    auto after = db.search(q, 1, 16);
    ASSERT(after.size() == 1);
    EXPECT(after[0] == ids[1]);             // next-nearest survivor
    EXPECT(!contains_id(after, ids[0]));    // deleted id never resurfaces

    // The remaining four are all still reachable.
    auto all = db.search(q, 10, 16);
    EXPECT(all.size() == 4);
    EXPECT(!contains_id(all, ids[0]));
}

TEST(vdb_search_still_returns_K_live_results_after_deletes) {
    // Delete an interior point and confirm we still get K survivors back (the ef
    // buffer absorbs the tombstone so the result set isn't short-changed).
    VDB db(cfg2d());
    std::vector<ExternalId> ids;
    for (int i = 0; i < 8; ++i) {
        float v[2] = {static_cast<float>(i), 0};
        ids.push_back(db.insert(v));
    }
    EXPECT(db.remove(ids[2]) == true);

    float q[2] = {0, 0};
    auto r = db.search(q, 3, 32);
    ASSERT(r.size() == 3);
    EXPECT(!contains_id(r, ids[2]));
    // Closest three survivors to x=0 are ids[0], ids[1], ids[3].
    EXPECT(r[0] == ids[0]);
    EXPECT(r[1] == ids[1]);
    EXPECT(r[2] == ids[3]);
}

TEST(vdb_update_keeps_same_external_id_and_changes_vector) {
    VDB db(cfg2d());
    float origin[2]    = {0, 0};
    float near_origin[2] = {1, 1};
    float far[2]       = {100, 100};

    auto a = db.insert(origin);        // id 1, at origin
    auto b = db.insert(far);           // id 2, far away
    auto c = db.insert(near_origin);   // id 3, stays near origin
    (void)b;

    // Move id `a` out to (100, 99) — right next to b, far from the origin.
    float moved[2] = {100, 99};
    EXPECT(db.update(a, moved) == true);

    EXPECT(db.size() == 3);            // update is not a net insert
    EXPECT(db.contains(a));            // same external id survives

    const float* got = db.get(a);
    ASSERT(got != nullptr);
    EXPECT(got[0] == 100);
    EXPECT(got[1] == 99);

    // Querying near the OLD location now returns `c`, not `a`: a's old node is a
    // tombstone and its data moved away.
    auto near_old = db.search(origin, 1, 16);
    ASSERT(near_old.size() == 1);
    EXPECT(near_old[0] == c);
    EXPECT(near_old[0] != a);

    // Querying near the NEW location does.
    float near_new[2] = {100, 98};
    auto r = db.search(near_new, 1, 16);
    ASSERT(r.size() == 1);
    EXPECT(r[0] == a);
}

TEST(vdb_update_unknown_id_returns_false) {
    VDB db(cfg2d());
    float v[2] = {1, 1};
    EXPECT(db.update(42, v) == false);
}

TEST(vdb_size_reflects_inserts_and_deletes) {
    VDB db(cfg2d());
    std::vector<ExternalId> ids;
    for (int i = 0; i < 10; ++i) {
        float v[2] = {static_cast<float>(i), 0};
        ids.push_back(db.insert(v));
    }
    EXPECT(db.size() == 10);

    for (int i = 0; i < 4; ++i) db.remove(ids[i]);
    EXPECT(db.size() == 6);

    // Re-inserting yields brand-new external ids; old ones are not recycled.
    float v[2] = {-1, 0};
    auto fresh = db.insert(v);
    EXPECT(fresh == 11);
    EXPECT(db.size() == 7);
}
