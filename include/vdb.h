#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "distance.h"
#include "index.h"

namespace vdb {

enum class IndexKind { Brute, IVF, HNSW };

struct VDBConfig {
    IndexKind kind   = IndexKind::HNSW;
    size_t    dim    = 0;
    Metric    metric = Metric::L2;
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
class VDB {
public:
    explicit VDB(VDBConfig cfg);

    friend void     save_snapshot(const VDB& db, const std::string& path, uint64_t lsn);
    friend uint64_t load_snapshot(VDB& db, const std::string& path);

    // Insert a new vector; returns its freshly minted, stable ExternalId.
    ExternalId insert(const float* vec);

    // Tombstone the vector behind `id`. Returns false if `id` is unknown.
    bool remove(ExternalId id);

    // Tombstone the old vector and insert `vec` under the *same* ExternalId.
    // Returns false if `id` is unknown.
    bool update(ExternalId id, const float* vec);

    // True iff `id` refers to a live (non-tombstoned) vector.
    bool contains(ExternalId id) const;

    // Top-K live neighbours, nearest first, as external ids. Tombstoned hits are
    // skipped; the index is over-queried to compensate for them.
    std::vector<ExternalId> search(const float* query, size_t K) const;

    // Rebuild the index from live vectors only, reclaiming tombstoned space.
    // External ids are preserved; internal ids are renumbered.
    void compact();

    size_t size() const { return live_count_; }           // live vectors only
    size_t deleted_count() const { return deleted_count_; } // outstanding tombstones
    size_t dim() const { return config_.dim; }

private:
    VDBConfig              config_;
    std::unique_ptr<Index> index_;

    // Identity maps. Internal ids index the parallel arrays below; they always
    // match the offsets the index hands back from add() (both append-only).
    std::unordered_map<ExternalId, InternalId> ext_to_int_;
    std::vector<ExternalId>                    int_to_ext_;  // by internal id
    std::vector<bool>                          deleted_;     // by internal id
    std::vector<std::vector<float>>            vectors_;      // by internal id (raw copy)

    ExternalId next_ext_id_   = 0;
    size_t     live_count_    = 0;
    size_t     deleted_count_ = 0;

    // Append `vec` as a fresh internal node, keeping the parallel arrays in step.
    InternalId append_(ExternalId ext, const float* vec);

    static std::unique_ptr<Index> make_index_(const VDBConfig& cfg, DistanceFn dist_fn);
};

}
