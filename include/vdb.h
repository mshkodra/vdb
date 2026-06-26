#pragma once
#include <memory>
#include <vector>

#include "distance.h"
#include "index.h"

namespace vdb {

// Top-level database. Owns a chosen Index plus the ExternalId <-> InternalId
// mapping, deletes, metadata, and (later) durability + concurrency. The Index
// stays a pure ANN structure; the DB layer is where "database" behaviour lives.
// See ROADMAP Stage 5 onward.
enum class IndexKind { Brute, IVF, HNSW };

struct VDBConfig {
    IndexKind kind   = IndexKind::HNSW;
    size_t    dim    = 0;
    Metric    metric = Metric::L2;
    // Index-specific configs are filled in by the builder in vdb.cpp for now.
};

class VDB {
public:
    explicit VDB(VDBConfig cfg);

    // Returns the stable ExternalId for the inserted vector.
    ExternalId insert(const float* vec);

    // Returns ExternalIds of the K nearest neighbours, closest first.
    std::vector<ExternalId> search(const float* query, size_t K) const;

    size_t size() const;
    size_t dim() const;

private:
    VDBConfig              config_;
    std::unique_ptr<Index> index_;

    // TODO Stage 5: ExternalId <-> InternalId maps, tombstones, metadata.
};

}  // namespace vdb
