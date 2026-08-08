#pragma once
#include <cstdint>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "distance.h"
#include "index.h"
#include "metadata.h"

namespace vdb {

enum class IndexKind { Brute, IVF, HNSW };

struct VDBConfig {
    IndexKind kind   = IndexKind::HNSW;
    size_t    dim    = 0;
    Metric    metric = Metric::L2;
    // Declared once and fixed for the life of the database. Persisted by fingerprint
    // in the WAL and snapshot headers and checked on open, exactly like dim/metric.
    // Empty means "no filterable attributes"; payload still works.
    std::vector<AttrSpec> schema;
};

// A search result carrying its opaque payload. The payload is *copied* rather than
// handed back as a span into payload_: the shared lock is released when search
// returns, and a later compact() moves every row, so a span would dangle. Callers who
// don't want the copy use search(), which returns bare ids.
struct Hit {
    ExternalId           id;
    float                dist;
    std::vector<uint8_t> payload;
};

// VDB is the database layer sitting on top of a pure Index. The index speaks
// InternalId (an array offset that is unstable across compaction); VDB hands
// callers a stable ExternalId and translates at the boundary. That indirection
// is what makes deletes, updates, and compaction possible without breaking a
// caller's references.
//
// Deletes are tombstones: the internal node stays in the index (its links are
// load-bearing for HNSW connectivity, and shrinking an IVF list mid-flight is
// expensive) but is excluded from results. compact() reclaims the dead space by
// rebuilding the index from live vectors, remapping internal ids while keeping
// every external id stable.
//
// Thread-safety (Stage 7). VDB is safe for many concurrent readers and writers.
// `mu_` (a shared_mutex) guards the identity maps and parallel arrays: readers take
// it shared, writers exclusive. A writer holds it exclusive only for two brief
// phases — allocate (reserve the id, size the arrays) and publish (make the node
// visible) — and does the expensive graph link *between* them with `mu_` released,
// relying on the index's own per-node locks. So many inserts link in parallel and a
// half-inserted node is never observed: allocate marks it deleted_=true (pending)
// and publish flips it live. Concurrent mutations must target *distinct* external
// ids (typical multi-writer partitioning); same-id races are not serialised here.
class VDB {
public:
    explicit VDB(VDBConfig cfg);

    friend void     save_snapshot(const VDB& db, const std::string& path, uint64_t lsn);
    friend uint64_t load_snapshot(VDB& db, const std::string& path);

    // Insert a new vector; returns its freshly minted, stable ExternalId. The
    // metadata overload attaches attributes (positional against config.schema) and
    // an opaque payload; the bare overload inserts an all-null row.
    ExternalId insert(const float* vec);
    ExternalId insert(const float* vec, const Record& meta);

    // Two-step insert for the durable layer (Stage 7 step 3), so it can mint the id
    // in its serialised WAL prefix and apply it after the fsync. reserve_id() hands
    // out the next id; insert_reserved() runs the allocate→link→publish insert under
    // that pre-assigned id (and keeps next_ext_id_ ahead of it, so the replay path —
    // which calls insert_reserved directly with logged ids — stays consistent).
    ExternalId reserve_id();
    void       insert_reserved(ExternalId ext, const float* vec);
    void       insert_reserved(ExternalId ext, const float* vec, const Record& meta);

    // Tombstone the vector behind `id`. Returns false if `id` is unknown.
    bool remove(ExternalId id);

    // Tombstone the old vector and insert `vec` under the *same* ExternalId.
    // Returns false if `id` is unknown. The bare overload carries the existing
    // metadata row forward onto the replacement node; the metadata overload replaces
    // both the vector and the row.
    bool update(ExternalId id, const float* vec);
    bool update(ExternalId id, const float* vec, const Record& meta);

    // Replace the metadata row for `id` without touching the vector. O(1) and no
    // graph work at all — the whole reason metadata lives outside the index rather
    // than being folded into the node. Returns false if `id` is unknown.
    bool set_metadata(ExternalId id, const Record& meta);

    // Read back the attributes and payload for `id`. False if `id` is unknown.
    bool get_metadata(ExternalId id, Record& out) const;

    // True iff `id` refers to a live (non-tombstoned) vector.
    bool contains(ExternalId id) const;

    // Top-K live neighbours, nearest first, as external ids. Tombstoned hits are
    // skipped; the index is over-queried to compensate for them.
    std::vector<ExternalId> search(const float* query, size_t K) const;

    // Same search, but each hit carries its distance and a copy of its payload.
    std::vector<Hit> search_hits(const float* query, size_t K) const;

    // Post-filtered search: run the index as usual, then keep only live hits that
    // also match `pred`. `pred` is resolved once (MetadataStore::resolve) before the
    // hot loop, not re-dispatched per candidate. Throws std::invalid_argument if
    // `pred` doesn't resolve to an allowlist (a range on a non-`indexed` column) —
    // that per-candidate case isn't supported yet.
    std::vector<ExternalId> search(const float* query, size_t K, const Predicate& pred) const;
    std::vector<Hit> search_hits(const float* query, size_t K, const Predicate& pred) const;

    // Rebuild the index from live vectors only, reclaiming tombstoned space.
    // External ids are preserved; internal ids are renumbered.
    void compact();

    size_t size() const {                                  // live vectors only
        std::shared_lock<std::shared_mutex> lk(mu_);
        return live_count_;
    }
    size_t deleted_count() const {                         // outstanding tombstones
        std::shared_lock<std::shared_mutex> lk(mu_);
        return deleted_count_;
    }
    size_t dim() const { return config_.dim; }             // immutable; no lock

    // Schema fingerprint, for the durability layer's header validation.
    uint64_t schema_fingerprint() const { return meta_.fingerprint(); }

    // Exact count of live rows where attribute `attr` holds `code` (a dictionary
    // code for Tag, 0/1 for Bool) — the read side of MetadataStore's incrementally
    // maintained counts, for selectivity estimation.
    uint32_t attr_count(size_t attr, uint32_t code) const {
        std::shared_lock<std::shared_mutex> lk(mu_);
        return meta_.count(attr, code);
    }

    // The ExternalId the next insert() will mint. Lets a durable wrapper log an
    // insert record before applying it. Not synchronised: the durable layer calls
    // this inside its own serialised write path (Stage 6/7 step 3).
    ExternalId peek_next_id() const { return next_ext_id_; }

private:
    VDBConfig              config_;
    std::unique_ptr<Index> index_;

    // Guards the identity maps + parallel arrays below. mutable so the read paths
    // (search/contains/size) can take it shared from const methods.
    mutable std::shared_mutex mu_;

    // Identity maps. Internal ids index the parallel arrays below; they always
    // match the offsets the index hands back from add() (both append-only).
    std::unordered_map<ExternalId, InternalId> ext_to_int_;
    std::vector<ExternalId>                    int_to_ext_;  // by internal id
    std::vector<bool>                          deleted_;     // by internal id
    std::vector<std::vector<float>>            vectors_;      // by internal id (raw copy)

    // Filterable columns + opaque payloads, also by internal id — the same keyspace
    // as deleted_/vectors_ above, so a predicate and the tombstone check are one
    // indexed load each in the same loop.
    MetadataStore meta_;

    ExternalId next_ext_id_   = 0;
    size_t     live_count_    = 0;
    size_t     deleted_count_ = 0;

    // Append `vec` as a fresh *live* internal node (single-phase add), keeping the
    // parallel arrays in step. Not synchronised — the caller must hold mu_ exclusive
    // (or be single-threaded, as in compact()).
    InternalId append_(ExternalId ext, const float* vec);

    // Shared body of the two update() overloads; null `meta` carries the old row over.
    bool update_(ExternalId id, const float* vec, const Record* meta);

    // Shared body of the four search paths: run the index, drop tombstones (and,
    // if `pred` is given, non-matches), and hand each surviving (internal id,
    // distance) to `emit` until K have been taken. `pred` must already be resolved
    // (MetadataStore::resolve) — collect_ never dispatches on predicate kind itself.
    // Caller must hold mu_ (shared is enough).
    template <class Emit>
    void collect_(const float* query, size_t K, Emit&& emit, const ResolvedPredicate* pred = nullptr) const;

    static std::unique_ptr<Index> make_index_(const VDBConfig& cfg);
};

}
