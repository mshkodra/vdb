#pragma once

#include <hnswindex.h>

#include <memory>
#include <unordered_map>
#include <vector>

enum class IndexType {
    HNSW,
};

struct VDBConfig {
    IndexType  type = IndexType::HNSW;
    HNSWConfig hnsw;          // used when type == HNSW
    DistanceFn dist_fn;       // optional; defaults to squared L2 if empty
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

    void bind(InternalId internal, ExternalId ext);
};
