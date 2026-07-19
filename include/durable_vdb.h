#pragma once
#include <chrono>
#include <cstdint>
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
//   PerOpSync - fsync the log before returning (safe; durable-before-apply).
//   Periodic  - apply first and fsync at most every flush_interval_ms (fast;
//               a crash loses the last unflushed window).
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
    void log_(WalRecord&& r);
    void maybe_flush_();
    void maybe_checkpoint_();

    VDBConfig   config_;
    std::string data_dir_;
    std::string wal_dir_;
    std::string snapshot_path_;
    Options     opts_;
    VDB         db_;
    Wal         wal_;

    uint64_t next_lsn_             = 1;
    uint64_t snapshot_id_          = 0;
    uint64_t ops_since_checkpoint_ = 0;

    std::chrono::steady_clock::time_point last_sync_;
};

}
