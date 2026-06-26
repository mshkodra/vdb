#include "test.h"

#include <vdb.h>

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

// Phase 2 — Durability: WAL + snapshots + recovery.
//
// These tests prove the central guarantee: committed writes survive the database
// object being destroyed and reopened from disk, and a torn/corrupt WAL tail (a
// crash mid-write) is dropped cleanly rather than corrupting recovery.

namespace fs = std::filesystem;

namespace {

// Each test gets a fresh, empty data dir under the build tree.
std::string fresh_dir(const std::string& name) {
    const std::string dir = "build/_durability_" + name;
    std::error_code ec;
    fs::remove_all(dir, ec);
    fs::create_directories(dir);
    return dir;
}

VDBConfig durable_cfg(const std::string& dir,
                      DurabilityPolicy policy = DurabilityPolicy::Always) {
    VDBConfig cfg;
    cfg.hnsw.dim    = 2;
    cfg.data_dir    = dir;
    cfg.durability  = policy;
    return cfg;
}

bool contains_id(const std::vector<ExternalId>& v, ExternalId id) {
    return std::find(v.begin(), v.end(), id) != v.end();
}

// Append a single 2-D vector and return the id it was stored under.
ExternalId insert2d(VDB& db, float x, float y) {
    float v[2] = {x, y};
    return db.insert(v);
}

long file_size(const std::string& path) {
    std::error_code ec;
    const auto s = fs::file_size(path, ec);
    return ec ? -1 : static_cast<long>(s);
}

}  // namespace

TEST(durability_roundtrip_reopen) {
    const std::string dir = fresh_dir("roundtrip");

    std::vector<ExternalId> ids;
    {
        VDB db(durable_cfg(dir));
        for (int i = 0; i < 6; ++i) ids.push_back(insert2d(db, static_cast<float>(i), 0));
        EXPECT(db.size() == 6);
    }  // db destroyed: nothing in memory survives — only the WAL on disk does

    // Reopen from disk. Every vector must be present, searchable, and under the
    // same ExternalId.
    VDB db(durable_cfg(dir));
    ASSERT(db.size() == 6);
    for (int i = 0; i < 6; ++i) {
        const float* got = db.get(ids[i]);
        ASSERT(got != nullptr);
        EXPECT(got[0] == static_cast<float>(i));
        EXPECT(got[1] == 0);
    }

    float q[2] = {0, 0};
    auto r = db.search(q, 1, 16);
    ASSERT(r.size() == 1);
    EXPECT(r[0] == ids[0]);  // nearest to origin is the x=0 point
}

TEST(durability_delete_and_update_persist) {
    const std::string dir = fresh_dir("del_upd");

    ExternalId a, b, c;
    {
        VDB db(durable_cfg(dir));
        a = insert2d(db, 0, 0);
        b = insert2d(db, 1, 0);
        c = insert2d(db, 2, 0);
        EXPECT(db.remove(b) == true);    // tombstone the middle one
        float moved[2] = {9, 9};
        EXPECT(db.update(c, moved) == true);  // move c far away, same id
    }

    VDB db(durable_cfg(dir));
    EXPECT(db.size() == 2);              // a and c live; b gone
    EXPECT(db.contains(a));
    EXPECT(db.contains(b) == false);    // delete survived the restart
    EXPECT(db.contains(c));

    const float* gc = db.get(c);
    ASSERT(gc != nullptr);
    EXPECT(gc[0] == 9);                  // update survived the restart
    EXPECT(gc[1] == 9);

    float q[2] = {0, 0};
    auto r = db.search(q, 5, 16);
    EXPECT(!contains_id(r, b));          // tombstone never resurfaces
}

TEST(durability_next_ext_id_not_recycled_after_reopen) {
    const std::string dir = fresh_dir("ids");

    {
        VDB db(durable_cfg(dir));
        for (int i = 0; i < 5; ++i) insert2d(db, static_cast<float>(i), 0);  // ids 1..5
        EXPECT(db.remove(3) == true);
    }

    // After reopen the minter must resume past 5, never recycling 3 or reusing 1..5.
    VDB db(durable_cfg(dir));
    const ExternalId fresh = insert2d(db, 42, 42);
    EXPECT(fresh == 6);
    EXPECT(db.size() == 5);  // 4 survivors + 1 fresh
}

TEST(durability_torn_tail_is_dropped) {
    // Simulate a crash mid-append: chop the last byte off the WAL. Recovery must
    // drop exactly the torn last record and keep everything before it intact.
    const std::string dir = fresh_dir("torn");
    const std::string wal = (fs::path(dir) / "wal.log").string();

    std::vector<ExternalId> ids;
    {
        VDB db(durable_cfg(dir));
        for (int i = 0; i < 4; ++i) ids.push_back(insert2d(db, static_cast<float>(i), 0));
    }

    // Truncate one byte → the final record is now a torn tail.
    const long sz = file_size(wal);
    ASSERT(sz > 0);
    fs::resize_file(wal, static_cast<uintmax_t>(sz - 1));

    VDB db(durable_cfg(dir));
    EXPECT(db.size() == 3);                      // last insert lost, first three kept
    EXPECT(db.contains(ids[0]));
    EXPECT(db.contains(ids[1]));
    EXPECT(db.contains(ids[2]));
    EXPECT(db.contains(ids[3]) == false);        // the torn tail

    // The log must be usable again: a new insert appends cleanly past the truncation.
    const ExternalId fresh = insert2d(db, 100, 0);
    EXPECT(db.contains(fresh));

    VDB db2(durable_cfg(dir));                    // reopen once more to confirm
    EXPECT(db2.contains(fresh));
    EXPECT(db2.size() == 4);                      // 3 survivors + fresh
}

TEST(durability_crc_detects_corruption) {
    // Flip a byte inside the last record's payload. The CRC check must catch it and
    // treat the record as a corrupt tail (dropped), not silently load bad data.
    const std::string dir = fresh_dir("crc");
    const std::string wal = (fs::path(dir) / "wal.log").string();

    std::vector<ExternalId> ids;
    {
        VDB db(durable_cfg(dir));
        for (int i = 0; i < 3; ++i) ids.push_back(insert2d(db, static_cast<float>(i), 0));
    }

    // Corrupt a byte near the end of the file (inside the last record's payload).
    const long sz = file_size(wal);
    ASSERT(sz > 2);
    {
        std::fstream f(wal, std::ios::binary | std::ios::in | std::ios::out);
        f.seekg(sz - 2);
        char b = 0;
        f.read(&b, 1);
        f.seekp(sz - 2);
        b = static_cast<char>(b ^ 0xFF);  // flip all bits → CRC mismatch
        f.write(&b, 1);
    }

    VDB db(durable_cfg(dir));
    EXPECT(db.size() == 2);                 // last record failed CRC → dropped
    EXPECT(db.contains(ids[0]));
    EXPECT(db.contains(ids[1]));
    EXPECT(db.contains(ids[2]) == false);
}

TEST(durability_snapshot_then_more_ops) {
    // Checkpoint, then keep mutating, then reopen. Recovery = snapshot + WAL tail,
    // so the reopened state must reflect BOTH the pre- and post-checkpoint ops. The
    // checkpoint should also have rotated (shrunk) the WAL.
    const std::string dir = fresh_dir("snapshot");
    const std::string wal = (fs::path(dir) / "wal.log").string();

    std::vector<ExternalId> ids;
    long wal_before_rotate = 0;
    {
        VDB db(durable_cfg(dir));
        for (int i = 0; i < 5; ++i) ids.push_back(insert2d(db, static_cast<float>(i), 0));
        wal_before_rotate = file_size(wal);

        db.checkpoint();                    // fold the 5 inserts into a snapshot

        // WAL was rotated: it should now be much smaller than before (just header +
        // the CHECKPOINT breadcrumb).
        EXPECT(file_size(wal) < wal_before_rotate);

        // More ops land in the fresh WAL, after the snapshot's LSN.
        ids.push_back(insert2d(db, 5, 0));
        EXPECT(db.remove(ids[0]) == true);
    }

    // A snapshot file must exist on disk.
    bool found_snapshot = false;
    for (const auto& e : fs::directory_iterator(dir)) {
        if (e.path().filename().string().rfind("snapshot.", 0) == 0) found_snapshot = true;
    }
    EXPECT(found_snapshot);

    VDB db(durable_cfg(dir));
    EXPECT(db.size() == 5);                 // 6 inserted total, 1 removed
    EXPECT(db.contains(ids[0]) == false);   // pre-checkpoint insert, then removed post
    EXPECT(db.contains(ids[5]));            // post-checkpoint insert survived
    for (int i = 1; i <= 4; ++i) EXPECT(db.contains(ids[i]));

    const float* g = db.get(ids[5]);
    ASSERT(g != nullptr);
    EXPECT(g[0] == 5);
}

TEST(durability_auto_checkpoint_keeps_state_consistent) {
    // With a low threshold, checkpoints fire automatically mid-stream. State must
    // remain correct across the resulting snapshot+WAL rotations and a reopen.
    const std::string dir = fresh_dir("auto_cp");

    VDBConfig cfg = durable_cfg(dir);
    cfg.checkpoint_threshold_ops = 3;  // snapshot every 3 mutations

    std::vector<ExternalId> ids;
    {
        VDB db(cfg);
        for (int i = 0; i < 10; ++i) ids.push_back(insert2d(db, static_cast<float>(i), 0));
        EXPECT(db.size() == 10);
    }

    VDB db(cfg);
    EXPECT(db.size() == 10);
    for (int i = 0; i < 10; ++i) {
        const float* g = db.get(ids[i]);
        ASSERT(g != nullptr);
        EXPECT(g[0] == static_cast<float>(i));
    }
}

TEST(durability_policy_never_roundtrips_on_clean_close) {
    // DurabilityPolicy::Never skips fsync, but write() still reaches the page cache,
    // so a clean restart (no power loss) recovers everything. This pins that the
    // record format / replay path is independent of the fsync policy.
    const std::string dir = fresh_dir("never");

    std::vector<ExternalId> ids;
    {
        VDB db(durable_cfg(dir, DurabilityPolicy::Never));
        for (int i = 0; i < 4; ++i) ids.push_back(insert2d(db, static_cast<float>(i), 0));
    }

    VDB db(durable_cfg(dir, DurabilityPolicy::Never));
    EXPECT(db.size() == 4);
    for (auto id : ids) EXPECT(db.contains(id));
}

TEST(durability_reopen_with_wrong_dim_is_rejected) {
    // The WAL/snapshot headers stamp the dim; opening with a mismatched config must
    // fail loudly rather than silently misread fixed-width vector payloads.
    const std::string dir = fresh_dir("dim_guard");
    {
        VDB db(durable_cfg(dir));   // dim = 2
        insert2d(db, 1, 2);
    }

    bool threw = false;
    try {
        VDBConfig cfg = durable_cfg(dir);
        cfg.hnsw.dim = 3;           // mismatched
        VDB db(cfg);
    } catch (const std::exception&) {
        threw = true;
    }
    EXPECT(threw);
}
