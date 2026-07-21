#pragma once
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

#include "vdb.h"
#include "wal.h"

namespace vdb {

// A crash-safe VDB: composes a pure in-memory VDB with a write-ahead log and
// periodic snapshots. Every mutation is logged before it is applied; on startup
// the state is rebuilt from the newest snapshot plus the WAL tail.
//
// Durability policy:
//   PerOpSync - the write is durable before it is applied (strict visibility); the
//               fsync is batched across concurrent writers (group commit).
//   Periodic  - apply first and fsync at most every flush_interval_ms (fast;
//               a crash loses the last unflushed window).
//
// Concurrency (Stage 7 step 3). Many writers and readers are safe. A mutation runs
// in three parts: a serialised WAL *prefix* under `write_mutex_` (mint the ext id,
// assign the lsn, append the record — this one section fixes WAL-order = lsn-order
// = ext-id-order), then durability, then apply into the (thread-safe) VDB. Under
// PerOpSync the prefix is followed by group_commit_(): one fsync, batched by a
// leader/follower handshake, makes many writers durable at once. Readers just
// forward to VDB. checkpoint() blocks new prefixes and waits for in-flight applies
// to drain, so the snapshot it serialises is consistent.
class DurableVDB {
public:
    enum class Policy { PerOpSync, Periodic };

    struct Options {
        Policy       policy            = Policy::Periodic;
        uint64_t     flush_interval_ms = 1000;
        uint64_t     checkpoint_ops    = 10000;  // ops between automatic checkpoints
        Wal::Options wal;
    };

    DurableVDB(VDBConfig cfg, const std::string& data_dir, Options opts);
    DurableVDB(VDBConfig cfg, const std::string& data_dir)
        : DurableVDB(cfg, data_dir, Options{}) {}
    ~DurableVDB();

    DurableVDB(const DurableVDB&) = delete;
    DurableVDB& operator=(const DurableVDB&) = delete;

    ExternalId insert(const float* vec);
    bool       remove(ExternalId id);
    bool       update(ExternalId id, const float* vec);

    bool                    contains(ExternalId id) const { return db_.contains(id); }
    std::vector<ExternalId> search(const float* query, size_t K) const {
        return db_.search(query, K);
    }

    void checkpoint();  // snapshot now and reclaim the covered WAL
    void sync();        // fsync the WAL now

    size_t size() const { return db_.size(); }
    size_t deleted_count() const { return db_.deleted_count(); }
    size_t dim() const { return db_.dim(); }

private:
    void recover_();
    void apply_(const WalRecord& r);
    void group_commit_(uint64_t lsn);  // PerOpSync: wait until lsn is durable (batched)
    void maybe_flush_();               // Periodic: lazy time-based fsync
    void finish_op_();                 // drop the in-flight count; wake a waiting checkpoint

    VDBConfig   config_;
    std::string data_dir_;
    std::string wal_dir_;
    std::string snapshot_path_;
    Options     opts_;
    VDB         db_;
    Wal         wal_;

    // next_lsn_/snapshot_id_/ops_since_checkpoint_ are touched only under write_mutex_.
    uint64_t next_lsn_             = 1;
    uint64_t snapshot_id_          = 0;
    uint64_t ops_since_checkpoint_ = 0;

    std::mutex write_mutex_;  // serialises the WAL prefix (reserve id, lsn, append)

    // Group-commit / quiesce state, all guarded by commit_mutex_.
    std::mutex              commit_mutex_;
    std::condition_variable commit_cv_;
    uint64_t                durable_lsn_ = 0;      // highest lsn known on stable storage
    bool                    syncing_     = false;  // a leader is fsyncing right now
    uint64_t                in_flight_   = 0;      // prefixed-but-not-yet-applied ops
    std::chrono::steady_clock::time_point last_sync_;
};

}
