#include "test.h"

#include "crc32c.h"
#include "distance.h"
#include "durable_vdb.h"
#include "snapshot.h"
#include "vdb.h"
#include "wal.h"

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include <unistd.h>

using namespace vdb;

namespace {

// A fresh, empty temp directory for one WAL test, removed if it already exists.
std::string fresh_wal_dir() {
    static int counter = 0;
    auto dir = std::filesystem::temp_directory_path() /
               ("vdb_wal_" + std::to_string(::getpid()) + "_" + std::to_string(counter++));
    std::filesystem::remove_all(dir);
    return dir.string();
}

size_t count_segments(const std::string& dir) {
    size_t n = 0;
    for (const auto& e : std::filesystem::directory_iterator(dir))
        if (e.path().extension() == ".log") ++n;
    return n;
}

WalRecord insert_rec(uint64_t lsn, uint64_t ext_id, std::vector<float> vec) {
    return WalRecord{WalRecordType::Insert, lsn, ext_id, std::move(vec), {}};
}

}  // namespace

TEST(crc32c_known_vector) {
    // The canonical CRC32C check value for the ASCII string "123456789".
    const char* msg = "123456789";
    EXPECT(crc32c(msg, 9) == 0xE3069283u);
}

TEST(crc32c_detects_single_bit_flip) {
    std::vector<uint8_t> a = {0x10, 0x20, 0x30, 0x40};
    std::vector<uint8_t> b = a;
    b[2] ^= 0x01;  // flip one bit
    EXPECT(crc32c(a.data(), a.size()) != crc32c(b.data(), b.size()));
}

TEST(wal_header_round_trip) {
    auto buf = encode_header(/*dim=*/128, /*metric=*/2, /*schema_fp=*/0xABCDEF01u);
    EXPECT(buf.size() == kWalHeaderSize);
    WalHeader h;
    ASSERT(decode_header(buf.data(), buf.size(), h));
    EXPECT(h.magic == kWalMagic);
    EXPECT(h.version == kWalVersion);
    EXPECT(h.dim == 128);
    EXPECT(h.metric == 2);
    EXPECT(h.schema_fp == 0xABCDEF01u);
}

TEST(wal_header_rejects_bad_magic) {
    auto buf = encode_header(4, 0, 0);
    buf[0] ^= 0xFF;  // corrupt the magic
    WalHeader h;
    EXPECT(!decode_header(buf.data(), buf.size(), h));
}

TEST(wal_insert_record_round_trip) {
    WalRecord in{WalRecordType::Insert, /*lsn=*/42, /*ext_id=*/7,
                 {1.0f, 2.0f, 3.0f, 4.0f}};
    auto frame = encode_record(in);

    WalRecord out;
    size_t consumed = 0;
    ASSERT(decode_record(frame.data(), frame.size(), out, consumed) == DecodeStatus::Ok);
    EXPECT(consumed == frame.size());
    EXPECT(out.type == WalRecordType::Insert);
    EXPECT(out.lsn == 42);
    EXPECT(out.ext_id == 7);
    ASSERT(out.vec.size() == 4);
    EXPECT(out.vec[0] == 1.0f);
    EXPECT(out.vec[3] == 4.0f);
}

TEST(wal_delete_record_has_no_vector) {
    WalRecord in{WalRecordType::Delete, /*lsn=*/9, /*ext_id=*/3, {}};
    auto frame = encode_record(in);

    WalRecord out;
    size_t consumed = 0;
    ASSERT(decode_record(frame.data(), frame.size(), out, consumed) == DecodeStatus::Ok);
    EXPECT(out.type == WalRecordType::Delete);
    EXPECT(out.ext_id == 3);
    EXPECT(out.vec.empty());
}

TEST(wal_torn_tail_is_incomplete) {
    WalRecord in{WalRecordType::Insert, 1, 1, {1.0f, 2.0f}};
    auto frame = encode_record(in);

    // Any strict prefix of a frame is a torn tail: not enough bytes to decode.
    WalRecord out;
    size_t consumed = 0;
    for (size_t n = 0; n < frame.size(); ++n) {
        EXPECT(decode_record(frame.data(), n, out, consumed) == DecodeStatus::Incomplete);
    }
    EXPECT(decode_record(frame.data(), frame.size(), out, consumed) == DecodeStatus::Ok);
}

TEST(wal_corruption_is_bad_crc) {
    WalRecord in{WalRecordType::Insert, 5, 5, {9.0f, 8.0f, 7.0f}};
    auto frame = encode_record(in);

    // Flip a byte in the body (after the length+crc header): the recomputed crc
    // no longer matches the stored one.
    frame[10] ^= 0x01;

    WalRecord out;
    size_t consumed = 0;
    EXPECT(decode_record(frame.data(), frame.size(), out, consumed) == DecodeStatus::BadCrc);
    EXPECT(consumed == 0);
}

// ---- Wal (segmented file log) ----------------------------------------------

TEST(wal_round_trip_across_segments) {
    const std::string dir = fresh_wal_dir();
    Wal::Options opts;
    opts.segment_max_bytes = 100;  // tiny: forces rotation every couple records

    {
        Wal w;
        w.open(dir, /*dim=*/2, /*metric=*/0, /*schema_fp=*/0, opts);
        for (uint64_t i = 1; i <= 6; ++i)
            w.append(insert_rec(i, i * 10, {float(i), float(i)}));
        w.sync();
    }

    // Rotation actually happened: more than one segment on disk.
    EXPECT(count_segments(dir) > 1);

    std::vector<uint64_t> lsns;
    {
        Wal w;
        w.open(dir, 2, 0, 0, opts);
        w.replay([&](const WalRecord& r) { lsns.push_back(r.lsn); });
        EXPECT(w.max_lsn() == 6);
    }

    ASSERT(lsns.size() == 6);
    for (uint64_t i = 0; i < 6; ++i) EXPECT(lsns[i] == i + 1);

    std::filesystem::remove_all(dir);
}

TEST(wal_crash_injection_drops_torn_tail) {
    const std::string dir = fresh_wal_dir();

    // Write 5 records in a single (large) segment, then close cleanly.
    {
        Wal w;
        w.open(dir, /*dim=*/2, /*metric=*/0, /*schema_fp=*/0);
        for (uint64_t i = 1; i <= 5; ++i)
            w.append(insert_rec(i, i, {float(i), 0.0f}));
        w.sync();
    }

    // Simulate a torn final write: chop 3 bytes off the end of the active segment.
    std::string seg;
    for (const auto& e : std::filesystem::directory_iterator(dir))
        if (e.path().extension() == ".log") seg = e.path().string();
    ASSERT(!seg.empty());
    const auto sz = std::filesystem::file_size(seg);
    std::filesystem::resize_file(seg, sz - 3);

    // Recovery must drop exactly the torn last record and keep the first four.
    std::vector<uint64_t> lsns;
    {
        Wal w;
        w.open(dir, 2, 0, 0);
        w.replay([&](const WalRecord& r) { lsns.push_back(r.lsn); });
        ASSERT(lsns.size() == 4);
        EXPECT(lsns.back() == 4);

        // The truncation left a clean append boundary: a new write survives replay.
        w.append(insert_rec(5, 5, {5.0f, 0.0f}));
        w.sync();
    }

    std::vector<uint64_t> after;
    {
        Wal w;
        w.open(dir, 2, 0, 0);
        w.replay([&](const WalRecord& r) { after.push_back(r.lsn); });
    }
    ASSERT(after.size() == 5);
    EXPECT(after.back() == 5);

    std::filesystem::remove_all(dir);
}

TEST(wal_truncate_reclaims_covered_segments) {
    const std::string dir = fresh_wal_dir();
    Wal::Options opts;
    opts.segment_max_bytes = 100;  // ~2 records per segment for dim=2 inserts

    {
        Wal w;
        w.open(dir, /*dim=*/2, /*metric=*/0, /*schema_fp=*/0, opts);
        for (uint64_t i = 1; i <= 6; ++i)
            w.append(insert_rec(i, i, {float(i), float(i)}));
        w.sync();
        const size_t before = count_segments(dir);
        EXPECT(before >= 3);

        // A snapshot covered up to lsn 2 → the first sealed segment is reclaimable.
        w.truncate(2);
        EXPECT(count_segments(dir) < before);
    }

    // Records still on disk are exactly the suffix past the reclaimed segment.
    std::vector<uint64_t> lsns;
    {
        Wal w;
        w.open(dir, 2, 0, 0, opts);
        w.replay([&](const WalRecord& r) { lsns.push_back(r.lsn); });
    }
    ASSERT(!lsns.empty());
    EXPECT(lsns.front() == 3);  // 1 and 2 lived in the reclaimed segment
    EXPECT(lsns.back() == 6);

    std::filesystem::remove_all(dir);
}

// ---- Snapshots --------------------------------------------------------------

TEST(snapshot_brute_round_trip) {
    const std::string dir = fresh_wal_dir();
    std::filesystem::create_directories(dir);
    const std::string path = dir + "/snap";

    VDBConfig cfg;
    cfg.kind = IndexKind::Brute;
    cfg.dim = 3;
    cfg.metric = Metric::L2;

    const float q[3] = {0.1f, 0.2f, 0.3f};
    std::vector<ExternalId> ids;
    std::vector<ExternalId> want;
    {
        VDB db(cfg);
        for (int i = 0; i < 20; ++i) {
            const float v[3] = {float(i), float((i * 2) % 7), float(i % 5)};
            ids.push_back(db.insert(v));
        }
        db.remove(ids[3]);
        db.remove(ids[7]);
        want = db.search(q, 5);
        save_snapshot(db, path, 123);
    }

    VDB db2(cfg);
    EXPECT(load_snapshot(db2, path) == 123);
    EXPECT(db2.size() == 18);
    EXPECT(!db2.contains(ids[3]));
    EXPECT(db2.contains(ids[0]));

    const auto got = db2.search(q, 5);
    ASSERT(got.size() == want.size());
    for (size_t i = 0; i < got.size(); ++i) EXPECT(got[i] == want[i]);

    std::filesystem::remove_all(dir);
}

TEST(snapshot_hnsw_preserves_graph) {
    const std::string dir = fresh_wal_dir();
    std::filesystem::create_directories(dir);
    const std::string path = dir + "/snap";

    VDBConfig cfg;
    cfg.kind = IndexKind::HNSW;
    cfg.dim = 8;
    cfg.metric = Metric::L2;

    auto gen = [](int i, int d) { return float((i * 31 + d * 7) % 97) * 0.1f; };
    std::vector<float> q(8);
    for (int d = 0; d < 8; ++d) q[d] = gen(3, d);

    std::vector<ExternalId> ids;
    std::vector<ExternalId> want;
    {
        VDB db(cfg);
        for (int i = 0; i < 60; ++i) {
            std::vector<float> v(8);
            for (int d = 0; d < 8; ++d) v[d] = gen(i, d);
            ids.push_back(db.insert(v.data()));
        }
        db.remove(ids[10]);
        want = db.search(q.data(), 5);
        save_snapshot(db, path, 7);
    }

    VDB db2(cfg);
    EXPECT(load_snapshot(db2, path) == 7);
    EXPECT(db2.size() == 59);

    // The graph is restored byte-for-byte, so search returns identical results.
    const auto got = db2.search(q.data(), 5);
    ASSERT(got.size() == want.size());
    for (size_t i = 0; i < got.size(); ++i) EXPECT(got[i] == want[i]);

    std::filesystem::remove_all(dir);
}

TEST(snapshot_rejects_config_mismatch) {
    const std::string dir = fresh_wal_dir();
    std::filesystem::create_directories(dir);
    const std::string path = dir + "/snap";

    VDBConfig cfg;
    cfg.kind = IndexKind::Brute;
    cfg.dim = 3;
    cfg.metric = Metric::L2;
    {
        VDB db(cfg);
        const float v[3] = {1.0f, 2.0f, 3.0f};
        db.insert(v);
        save_snapshot(db, path, 1);
    }

    VDBConfig bad = cfg;
    bad.dim = 4;
    VDB db2(bad);
    bool threw = false;
    try {
        load_snapshot(db2, path);
    } catch (...) {
        threw = true;
    }
    EXPECT(threw);

    std::filesystem::remove_all(dir);
}

// ---- DurableVDB (WAL + snapshots wired together) ----------------------------

TEST(durable_round_trip_recovers_state) {
    const std::string dir = fresh_wal_dir();
    VDBConfig cfg;
    cfg.kind = IndexKind::HNSW;
    cfg.dim = 6;
    cfg.metric = Metric::L2;
    DurableVDB::Options opts;
    opts.policy = DurableVDB::Policy::PerOpSync;

    auto gen = [](int i, int d) { return float((i * 13 + d * 5) % 89) * 0.1f; };
    std::vector<float> q(6);
    for (int d = 0; d < 6; ++d) q[d] = gen(2, d);

    std::vector<ExternalId> ids;
    std::vector<ExternalId> want;
    {
        DurableVDB db(cfg, dir, opts);
        for (int i = 0; i < 40; ++i) {
            std::vector<float> v(6);
            for (int d = 0; d < 6; ++d) v[d] = gen(i, d);
            ids.push_back(db.insert(v.data()));
        }
        db.remove(ids[5]);
        db.remove(ids[9]);
        want = db.search(q.data(), 5);
    }  // dropped without a clean checkpoint; PerOpSync leaves the WAL fully durable

    DurableVDB db2(cfg, dir, opts);
    EXPECT(db2.size() == 38);
    EXPECT(!db2.contains(ids[5]));
    EXPECT(db2.contains(ids[0]));

    const auto got = db2.search(q.data(), 5);
    ASSERT(got.size() == want.size());
    for (size_t i = 0; i < got.size(); ++i) EXPECT(got[i] == want[i]);

    std::filesystem::remove_all(dir);
}

TEST(durable_recovers_across_checkpoint) {
    const std::string dir = fresh_wal_dir();
    VDBConfig cfg;
    cfg.kind = IndexKind::Brute;
    cfg.dim = 4;
    cfg.metric = Metric::L2;
    DurableVDB::Options opts;
    opts.policy = DurableVDB::Policy::PerOpSync;
    opts.checkpoint_ops = 10;  // force checkpoints partway through

    const float q[4] = {0.5f, 0.5f, 0.5f, 0.5f};
    std::vector<ExternalId> ids;
    std::vector<ExternalId> want;
    {
        DurableVDB db(cfg, dir, opts);
        for (int i = 0; i < 25; ++i) {
            const float v[4] = {float(i), float(i % 3), float(i % 7), float(i % 2)};
            ids.push_back(db.insert(v));
        }
        db.remove(ids[2]);
        want = db.search(q, 5);
    }
    EXPECT(std::filesystem::exists(dir + "/snapshot"));  // a checkpoint ran

    DurableVDB db2(cfg, dir, opts);
    EXPECT(db2.size() == 24);
    EXPECT(!db2.contains(ids[2]));

    const auto got = db2.search(q, 5);
    ASSERT(got.size() == want.size());
    for (size_t i = 0; i < got.size(); ++i) EXPECT(got[i] == want[i]);

    std::filesystem::remove_all(dir);
}

TEST(durable_periodic_recovers_after_close) {
    const std::string dir = fresh_wal_dir();
    VDBConfig cfg;
    cfg.kind = IndexKind::Brute;
    cfg.dim = 3;
    cfg.metric = Metric::L2;
    DurableVDB::Options opts;
    opts.policy = DurableVDB::Policy::Periodic;
    opts.flush_interval_ms = 100000;  // never flush mid-run; only the destructor does

    std::vector<ExternalId> ids;
    {
        DurableVDB db(cfg, dir, opts);
        for (int i = 0; i < 5; ++i) {
            const float v[3] = {float(i), 0.0f, 0.0f};
            ids.push_back(db.insert(v));
        }
    }  // destructor flushes the deferred writes

    DurableVDB db2(cfg, dir, opts);
    EXPECT(db2.size() == 5);
    EXPECT(db2.contains(ids[4]));

    std::filesystem::remove_all(dir);
}
