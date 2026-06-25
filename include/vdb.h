#pragma once

#include <hnswindex.h>
#include <wal.h>

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

enum class IndexType {
    HNSW,
};

struct VDBConfig {
    IndexType  type = IndexType::HNSW;
    HNSWConfig hnsw;          // used when type == HNSW
    DistanceFn dist_fn;       // optional; defaults to squared L2 if empty

    // ---- Durability (Phase 2) ----
    // data_dir empty  → pure in-memory database (Phase 1 behaviour, no disk I/O).
    // data_dir set    → durable: a WAL + periodic snapshots live under this dir,
    //                   and the constructor recovers prior state from them.
    std::string      data_dir;
    DurabilityPolicy durability        = DurabilityPolicy::Always;
    uint32_t         fsync_interval_ms = 1000;  // used when durability == EveryMs
    uint8_t          metric            = 0;      // 0 = L2; stamped into file headers

    // Auto-checkpoint after this many mutating ops (0 = never auto-checkpoint;
    // callers can still invoke VDB::checkpoint() explicitly).
    uint64_t checkpoint_threshold_ops = 0;
};

// VDB is the database layer that owns the *logical identity* of every vector.
//
// The HNSW index underneath addresses nodes by InternalId — a physical offset into
// its node pool. Those offsets are an implementation detail: they are reused, they
// shift under compaction, and they leak the storage layout. VDB therefore never
// exposes them. Instead it hands callers a stable ExternalId and keeps a two-way
// map between the two. This layer of indirection is what makes delete, update, and
// (later) compaction possible without invalidating references callers already hold.
class VDB {
public:
    explicit VDB(VDBConfig cfg);

    // Insert a vector; returns a fresh, stable ExternalId (>= 1).
    ExternalId insert(const float* vec);

    // K nearest live neighbours of `query`, returned as ExternalIds, closest first.
    std::vector<ExternalId> search(const float* query, size_t K, size_t ef) const;

    // Pointer to the stored vector for `id`, or nullptr if `id` is absent/deleted.
    const float* get(ExternalId id) const;

    // Tombstone the vector behind `id`. Returns false if `id` is not present.
    bool remove(ExternalId id);

    // Replace the vector behind `id` while keeping the SAME ExternalId. Implemented
    // as tombstone-old + insert-new (HNSW links are load-bearing and cannot be
    // edited in place). Returns false if `id` is not present.
    bool update(ExternalId id, const float* vec);

    bool   contains(ExternalId id) const;
    size_t size() const;   // number of live (non-deleted) vectors
    size_t dim()  const;

    // Write a full snapshot and rotate (truncate) the WAL. No-op for an in-memory
    // database. Callers rarely need this — it also runs automatically once
    // checkpoint_threshold_ops mutating operations have accumulated.
    void checkpoint();

    // True when backed by a WAL + snapshots on disk.
    bool durable() const { return durable_; }

private:
    VDBConfig                                  config_;
    std::unique_ptr<HNSWIndex>                 hnsw_;

    // ext_to_int_ holds only LIVE keys: removing an entry makes the key read as gone
    // immediately. int_to_ext_ is indexed by InternalId and is append-only (it must
    // keep mapping tombstoned internal nodes back to the ExternalId they last held,
    // because search may still surface them before they are filtered out).
    std::unordered_map<ExternalId, InternalId> ext_to_int_;
    std::vector<ExternalId>                    int_to_ext_;
    ExternalId                                 next_ext_id_ = 1;

    // ---- durability state (only meaningful when durable_) ----
    std::unique_ptr<Wal> wal_;
    bool                 durable_              = false;
    uint64_t             current_lsn_          = 0;  // last durably-logged LSN
    uint64_t             snapshot_seq_         = 0;  // id of the newest snapshot
    uint64_t             ops_since_checkpoint_ = 0;

    void bind(InternalId internal, ExternalId ext);

    // In-memory state transitions, shared by the live write path and WAL replay.
    // They take an explicit ExternalId because replay must reconstruct the exact
    // same identities that were logged — minting fresh ids would diverge from disk.
    void do_insert(ExternalId ext, const float* vec);
    void do_remove(ExternalId ext);
    void do_update(ExternalId ext, const float* vec);
    void apply(const WalRecord& rec);

    // log-before-apply: append rec to the WAL (assigning the next LSN) and fsync
    // per policy, BEFORE the matching do_* mutates memory. No-op when !durable_.
    void log(WalRecordType type, ExternalId ext, const float* vec);

    void recover();              // load snapshot + replay WAL on open
    void maybe_auto_checkpoint();
};
