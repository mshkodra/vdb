#include "test.h"

#include <vdb.h>

#include <vector>

TEST(vdb_constructs_with_hnsw) {
    VDBConfig cfg;
    cfg.type = IndexType::HNSW;
    cfg.hnsw.dim = 4;

    VDB db(cfg);
    EXPECT(db.size() == 0);
    EXPECT(db.dim()  == 4);
}

TEST(vdb_default_distance_is_l2_squared) {
    // No dist_fn provided — should default to squared L2 and still work.
    VDBConfig cfg;
    cfg.hnsw.dim = 3;
    VDB db(cfg);

    float v[3] = {1, 0, 0};
    auto id = db.insert(v);
    auto r = db.search(v, 1, 10);
    ASSERT(r.size() == 1);
    EXPECT(r[0] == id);
}

TEST(vdb_custom_distance_function_used) {
    int call_count = 0;
    auto counting_l2 = [&call_count](const float* a, const float* b, size_t d) {
        ++call_count;
        float s = 0.0f;
        for (size_t i = 0; i < d; ++i) {
            float x = a[i] - b[i];
            s += x * x;
        }
        return s;
    };

    VDBConfig cfg;
    cfg.hnsw.dim = 2;
    cfg.dist_fn = counting_l2;
    VDB db(cfg);

    float v1[2] = {0, 0};
    float v2[2] = {1, 1};
    db.insert(v1);
    db.insert(v2);
    db.search(v1, 1, 10);

    EXPECT(call_count > 0);
}

TEST(vdb_forwards_get_to_index) {
    VDBConfig cfg;
    cfg.hnsw.dim = 3;
    VDB db(cfg);

    float v[3] = {7, 8, 9};
    auto id = db.insert(v);
    const float* got = db.get(id);
    EXPECT(got[0] == 7);
    EXPECT(got[1] == 8);
    EXPECT(got[2] == 9);
}

TEST(vdb_size_reflects_inserts) {
    VDBConfig cfg;
    cfg.hnsw.dim = 2;
    VDB db(cfg);

    EXPECT(db.size() == 0);
    float v[2] = {0, 0};
    for (int i = 0; i < 25; ++i) {
        v[0] = (float)i;
        db.insert(v);
    }
    EXPECT(db.size() == 25);
}
