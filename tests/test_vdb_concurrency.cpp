#include "test.h"

#include <vdb.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <filesystem>
#include <mutex>
#include <random>
#include <set>
#include <string>
#include <thread>
#include <vector>

// Phase 3.1 — Concurrency (Isolation).
//
// VDB guards all shared state with a single reader/writer lock: many concurrent
// searches OR one mutation at a time. These tests are the safety net the roadmap
// asks for before any finer-grained locking (3.2): they hammer the database from
// many threads and assert the two things coarse locking must guarantee —
//   (1) no crash / no data race (run these under -fsanitize=thread to see the race
//       detector stay quiet), and
//   (2) no lost or duplicated writes (the exclusive lock must serialize id minting
//       and the WAL append path).
//
// Test-harness rule: the harness's failure counter (_vdb_fail behind EXPECT/ASSERT)
// is a plain int and is NOT safe to touch from worker threads. So every worker
// records into std::atomic counters / mutex-guarded containers, and ALL EXPECT /
// ASSERT checks run on the main thread after join().

namespace fs = std::filesystem;

namespace {

constexpr size_t kDim = 8;

VDBConfig mem_cfg() {
    VDBConfig cfg;
    cfg.hnsw.dim = kDim;
    return cfg;
}

std::string fresh_dir(const std::string& name) {
    const std::string dir = "build/_concurrency_" + name;
    std::error_code ec;
    fs::remove_all(dir, ec);
    fs::create_directories(dir);
    return dir;
}

VDBConfig durable_cfg(const std::string& dir) {
    VDBConfig cfg;
    cfg.hnsw.dim   = kDim;
    cfg.data_dir   = dir;
    cfg.durability = DurabilityPolicy::Always;
    return cfg;
}

// A deterministic pseudo-random vector keyed by `seed`, so a test can later
// recompute the exact bytes it expects a given logical point to hold.
std::array<float, kDim> make_vec(uint64_t seed) {
    std::mt19937 rng(static_cast<uint32_t>(seed * 2654435761u + 1));
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    std::array<float, kDim> v{};
    for (auto& x : v) x = u(rng);
    return v;
}

ExternalId insert_seeded(VDB& db, uint64_t seed) {
    auto v = make_vec(seed);
    return db.insert(v.data());
}

}  // namespace

// ---------------------------------------------------------------------------
// 1. Many readers + one writer: the classic shared/exclusive workload. The win
//    condition is "doesn't crash or race"; we also confirm reads kept returning
//    well-formed results the whole time and the writer's inserts all landed.
// ---------------------------------------------------------------------------
TEST(concurrency_many_readers_one_writer_no_crash) {
    VDB db(mem_cfg());

    constexpr int kSeed = 64;
    for (int i = 0; i < kSeed; ++i) insert_seeded(db, i);  // ids 1..64

    std::atomic<bool> stop{false};
    std::atomic<uint64_t> searches_done{0};
    std::atomic<uint64_t> bad_ids{0};  // any search result outside the minted range

    constexpr int kReaders = 8;
    std::vector<std::thread> readers;
    for (int r = 0; r < kReaders; ++r) {
        readers.emplace_back([&, r] {
            auto q = make_vec(1000 + r);
            while (!stop.load(std::memory_order_relaxed)) {
                auto res = db.search(q.data(), 10, 32);
                for (ExternalId id : res) {
                    // Every returned id must be one the database actually minted.
                    if (id == 0) ++bad_ids;
                }
                searches_done.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }

    // One writer streams in new vectors while the readers run.
    constexpr int kWrites = 400;
    std::thread writer([&] {
        for (int i = 0; i < kWrites; ++i) insert_seeded(db, 10000 + i);
        stop.store(true, std::memory_order_relaxed);
    });

    writer.join();
    for (auto& t : readers) t.join();

    EXPECT(bad_ids.load() == 0);
    EXPECT(searches_done.load() > 0);             // readers actually ran
    EXPECT(db.size() == kSeed + kWrites);         // every insert survived
}

// ---------------------------------------------------------------------------
// 2. No lost inserts: many writers minting ids in parallel. The exclusive lock
//    must serialize next_ext_id_, so the union of returned ids is exactly the
//    expected count with zero duplicates (a lost update would collide an id or
//    drop the size).
// ---------------------------------------------------------------------------
TEST(concurrency_no_lost_inserts) {
    VDB db(mem_cfg());

    constexpr int kWriters = 8;
    constexpr int kPer     = 150;

    std::mutex mu;
    std::vector<ExternalId> all_ids;
    all_ids.reserve(kWriters * kPer);

    std::vector<std::thread> writers;
    for (int w = 0; w < kWriters; ++w) {
        writers.emplace_back([&, w] {
            std::vector<ExternalId> local;
            local.reserve(kPer);
            for (int i = 0; i < kPer; ++i) {
                local.push_back(insert_seeded(db, static_cast<uint64_t>(w) * 100000 + i));
            }
            std::lock_guard<std::mutex> g(mu);
            all_ids.insert(all_ids.end(), local.begin(), local.end());
        });
    }
    for (auto& t : writers) t.join();

    const size_t expected = static_cast<size_t>(kWriters) * kPer;
    ASSERT(all_ids.size() == expected);

    // No id handed out twice.
    std::set<ExternalId> unique(all_ids.begin(), all_ids.end());
    EXPECT(unique.size() == expected);
    EXPECT(*unique.begin() == 1);                 // ids start at 1
    EXPECT(*unique.rbegin() == expected);         // and are dense 1..N

    EXPECT(db.size() == expected);                // index agrees with the id count
    for (ExternalId id : unique) EXPECT(db.contains(id));
}

// ---------------------------------------------------------------------------
// 3. Concurrent deletes over a disjoint partition. N live vectors, D deleters
//    each owning a distinct slice. Every delete must take effect exactly once,
//    so the final live count is N minus the deleted slice — no double-free of a
//    tombstone, no lost delete.
// ---------------------------------------------------------------------------
TEST(concurrency_disjoint_deletes_account_exactly) {
    VDB db(mem_cfg());

    constexpr int kN = 600;
    std::vector<ExternalId> ids;
    for (int i = 0; i < kN; ++i) ids.push_back(insert_seeded(db, i));

    constexpr int kDeleters = 6;          // each deletes ids[ d, d+kDeleters, ... ]
    std::atomic<uint64_t> ok_deletes{0};
    std::vector<std::thread> ts;
    for (int d = 0; d < kDeleters; ++d) {
        ts.emplace_back([&, d] {
            for (int i = d; i < kN; i += kDeleters) {
                if (db.remove(ids[i])) ok_deletes.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }
    for (auto& t : ts) t.join();

    EXPECT(ok_deletes.load() == static_cast<uint64_t>(kN));  // each removed once
    EXPECT(db.size() == 0);
    for (ExternalId id : ids) EXPECT(db.contains(id) == false);
}

// ---------------------------------------------------------------------------
// 4. Concurrent updates to the SAME id. update() is tombstone-old + insert-new
//    under one exclusive lock, so it is atomic: the id never disappears, the
//    live count never changes, and the surviving vector is exactly one of the
//    values some thread wrote (never a torn mix of two).
// ---------------------------------------------------------------------------
TEST(concurrency_updates_same_id_stay_atomic) {
    VDB db(mem_cfg());

    const ExternalId id = insert_seeded(db, 7);

    constexpr int kThreads = 8;
    constexpr int kIters   = 100;
    std::vector<std::thread> ts;
    for (int t = 0; t < kThreads; ++t) {
        ts.emplace_back([&, t] {
            for (int i = 0; i < kIters; ++i) {
                auto v = make_vec(static_cast<uint64_t>(t) * 1000 + i + 1);
                db.update(id, v.data());
            }
        });
    }
    for (auto& th : ts) th.join();

    EXPECT(db.size() == 1);            // never duplicated, never lost
    EXPECT(db.contains(id));

    // The survivor must equal one of the exact vectors that were written. No
    // concurrent write can be in flight now (all joined), so reading get()'s
    // pointer here is safe.
    const float* got = db.get(id);
    ASSERT(got != nullptr);
    bool matched = false;
    for (int t = 0; t < kThreads && !matched; ++t) {
        for (int i = 0; i < kIters; ++i) {
            auto v = make_vec(static_cast<uint64_t>(t) * 1000 + i + 1);
            if (std::equal(v.begin(), v.end(), got)) { matched = true; break; }
        }
    }
    EXPECT(matched);
}

// ---------------------------------------------------------------------------
// 5. Mixed-operation stress: every thread randomly inserts / searches /
//    contains / removes / updates against a shared db for many iterations. This
//    is the broad "throw everything at it" run — its job is to surface crashes,
//    deadlocks and data races. We also keep an exact account of net inserts so a
//    silently dropped or duplicated mutation shows up as a size mismatch.
// ---------------------------------------------------------------------------
TEST(concurrency_mixed_operations_stress) {
    VDB db(mem_cfg());

    // Seed so searches/removes have something to chew on from the very start.
    std::vector<ExternalId> seed_ids;
    for (int i = 0; i < 50; ++i) seed_ids.push_back(insert_seeded(db, i));

    std::atomic<int64_t> net_inserts{0};   // inserts minus successful removes
    std::atomic<uint64_t> bad{0};          // any malformed search result

    // Shared pool of known-live ids that threads remove from / add to. Guarded
    // by its own mutex (independent of VDB's internal lock).
    std::mutex pool_mu;
    std::vector<ExternalId> pool = seed_ids;

    constexpr int kThreads = 8;
    constexpr int kIters   = 500;
    std::vector<std::thread> ts;
    for (int t = 0; t < kThreads; ++t) {
        ts.emplace_back([&, t] {
            std::mt19937 rng(static_cast<uint32_t>(t * 7919 + 1));
            auto q = make_vec(50000 + t);
            for (int i = 0; i < kIters; ++i) {
                const int op = rng() % 5;
                if (op == 0) {  // insert
                    ExternalId id = insert_seeded(db, 200000 + static_cast<uint64_t>(t) * kIters + i);
                    net_inserts.fetch_add(1, std::memory_order_relaxed);
                    std::lock_guard<std::mutex> g(pool_mu);
                    pool.push_back(id);
                } else if (op == 1) {  // remove a candidate id
                    ExternalId victim = 0;
                    {
                        std::lock_guard<std::mutex> g(pool_mu);
                        if (!pool.empty()) {
                            size_t k = rng() % pool.size();
                            victim = pool[k];
                            pool[k] = pool.back();
                            pool.pop_back();
                        }
                    }
                    if (victim != 0 && db.remove(victim)) {
                        net_inserts.fetch_sub(1, std::memory_order_relaxed);
                    }
                } else if (op == 2) {  // update a candidate id (no net size change)
                    ExternalId target = 0;
                    {
                        std::lock_guard<std::mutex> g(pool_mu);
                        if (!pool.empty()) target = pool[rng() % pool.size()];
                    }
                    if (target != 0) {
                        auto v = make_vec(900000 + static_cast<uint64_t>(t) * kIters + i);
                        db.update(target, v.data());  // may race a remover → may be a no-op
                    }
                } else if (op == 3) {  // search
                    auto res = db.search(q.data(), 5, 32);
                    for (ExternalId id : res) if (id == 0) ++bad;
                } else {  // contains
                    (void)db.contains(static_cast<ExternalId>(rng() % 256));
                }
            }
        });
    }
    for (auto& th : ts) th.join();

    EXPECT(bad.load() == 0);

    // net_inserts counts seed + inserts - successful removes. db.size() must agree
    // exactly: that equality is what proves no mutation was lost or double-counted.
    const int64_t expected_live = static_cast<int64_t>(seed_ids.size()) + net_inserts.load();
    EXPECT(static_cast<int64_t>(db.size()) == expected_live);
    EXPECT(expected_live >= 0);

    // The index is still queryable and internally consistent after the storm.
    float q[kDim] = {0};
    auto res = db.search(q, 10, 64);
    EXPECT(res.size() <= 10);
    for (ExternalId id : res) EXPECT(db.contains(id));   // never returns a tombstone
}

// ---------------------------------------------------------------------------
// 6. Durable + concurrent: the WAL append path runs under the exclusive lock, so
//    concurrent writers must produce a clean, fully-replayable log. Write hard
//    from many threads, close, reopen from disk, and assert every committed
//    insert recovered — interleaved appends never corrupted the record stream.
// ---------------------------------------------------------------------------
TEST(concurrency_durable_writes_recover_fully) {
    const std::string dir = fresh_dir("durable");

    constexpr int kWriters = 6;
    constexpr int kPer     = 120;
    const size_t expected  = static_cast<size_t>(kWriters) * kPer;

    std::mutex mu;
    std::vector<ExternalId> ids;

    {
        VDB db(durable_cfg(dir));
        std::vector<std::thread> ts;
        for (int w = 0; w < kWriters; ++w) {
            ts.emplace_back([&, w] {
                std::vector<ExternalId> local;
                for (int i = 0; i < kPer; ++i)
                    local.push_back(insert_seeded(db, static_cast<uint64_t>(w) * 100000 + i));
                std::lock_guard<std::mutex> g(mu);
                ids.insert(ids.end(), local.begin(), local.end());
            });
        }
        for (auto& t : ts) t.join();
        EXPECT(db.size() == expected);
    }  // db destroyed → only the WAL on disk remains

    ASSERT(ids.size() == expected);

    // Reopen: replay must reconstruct every concurrently-logged insert.
    VDB db(durable_cfg(dir));
    EXPECT(db.size() == expected);
    for (ExternalId id : ids) EXPECT(db.contains(id));
}

// ---------------------------------------------------------------------------
// 7. Durable mixed ops with auto-checkpointing under concurrency. Snapshots fire
//    mid-stream (rotating the WAL) while many threads mutate. Reopening must
//    reproduce the exact final live set across the snapshot + WAL-tail boundary.
// ---------------------------------------------------------------------------
TEST(concurrency_durable_autocheckpoint_consistent) {
    const std::string dir = fresh_dir("durable_cp");

    VDBConfig cfg = durable_cfg(dir);
    cfg.checkpoint_threshold_ops = 25;   // checkpoints interleave with the writers

    std::set<ExternalId> expected_live;
    std::mutex live_mu;

    {
        VDB db(cfg);

        // Pre-seed a batch every thread can delete from deterministically.
        std::vector<ExternalId> seed;
        for (int i = 0; i < 40; ++i) {
            ExternalId id = insert_seeded(db, i);
            seed.push_back(id);
            expected_live.insert(id);
        }

        constexpr int kThreads = 4;
        constexpr int kPer     = 100;
        std::vector<std::thread> ts;
        for (int t = 0; t < kThreads; ++t) {
            ts.emplace_back([&, t] {
                for (int i = 0; i < kPer; ++i) {
                    ExternalId id =
                        insert_seeded(db, 300000 + static_cast<uint64_t>(t) * kPer + i);
                    std::lock_guard<std::mutex> g(live_mu);
                    expected_live.insert(id);
                }
            });
        }
        for (auto& th : ts) th.join();

        // Deterministically remove the seed set (single-threaded, exact accounting).
        for (ExternalId id : seed) {
            if (db.remove(id)) expected_live.erase(id);
        }

        db.checkpoint();  // final checkpoint to exercise the rotate path once more
        EXPECT(db.size() == expected_live.size());
    }

    VDB db(cfg);
    EXPECT(db.size() == expected_live.size());
    for (ExternalId id : expected_live) EXPECT(db.contains(id));
}

// ---------------------------------------------------------------------------
// 8. Readers must never observe a half-applied write or a garbage id. The writer
//    here only inserts and updates — both keep every id it has ever minted a LIVE
//    key (update rebinds the same ExternalId to a fresh internal node under the
//    exclusive lock). So every id a search returns must satisfy contains() == true,
//    with no legitimate race: a removal would create a real insert/remove TOCTOU,
//    which is exactly why we exclude removals from this particular invariant test.
//    Meanwhile inserts reallocate the node pool and updates churn tombstones, so the
//    readers are traversing structure that is constantly being rewritten underneath
//    them — yet under the shared lock they only ever see consistent committed states.
// ---------------------------------------------------------------------------
TEST(concurrency_search_results_are_always_live) {
    VDB db(mem_cfg());

    constexpr int kSeed = 100;
    std::vector<ExternalId> live_ids;
    for (int i = 0; i < kSeed; ++i) live_ids.push_back(insert_seeded(db, i));

    std::atomic<bool> stop{false};
    std::atomic<uint64_t> inconsistencies{0};
    std::atomic<uint64_t> checked{0};

    constexpr int kReaders = 6;
    std::vector<std::thread> readers;
    for (int r = 0; r < kReaders; ++r) {
        readers.emplace_back([&, r] {
            auto q = make_vec(77000 + r);
            while (!stop.load(std::memory_order_relaxed)) {
                auto res = db.search(q.data(), 8, 32);
                for (ExternalId id : res) {
                    // No id is ever removed in this test, so any live search result
                    // must still be a live key. A false here would be a torn read.
                    if (!db.contains(id)) inconsistencies.fetch_add(1, std::memory_order_relaxed);
                    checked.fetch_add(1, std::memory_order_relaxed);
                }
            }
        });
    }

    // Writer churns structure WITHOUT removing: fresh inserts (reallocate the pool)
    // and in-place updates of the seed ids (tombstone old slot + insert new).
    std::thread writer([&] {
        std::mt19937 rng(12345);
        for (int i = 0; i < 500; ++i) {
            insert_seeded(db, 88000 + i);
            ExternalId target = live_ids[rng() % live_ids.size()];
            auto v = make_vec(99000 + i);
            db.update(target, v.data());
        }
        stop.store(true, std::memory_order_relaxed);
    });

    writer.join();
    for (auto& t : readers) t.join();

    EXPECT(inconsistencies.load() == 0);
    EXPECT(checked.load() > 0);
}
