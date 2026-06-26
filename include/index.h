#pragma once
#include <cstddef>
#include <utility>
#include <vector>

#include "vdb_types.h"

namespace vdb {

// Abstract index interface shared by every index implementation (HNSW, IVF, and a
// brute-force baseline). This is a DESIGN DECISION worth owning — see ROADMAP
// Stage 0 "Design decisions". Notably:
//   * train() exists because IVF must learn centroids before it can add vectors,
//     while HNSW and brute force don't need it (default no-op).
//   * search() returns (InternalId, distance) pairs, closest first.
class Index {
public:
    virtual ~Index() = default;

    // Optional offline training step over a representative sample of `n` vectors
    // laid out row-major (n * dim floats). No-op for indexes that don't need it.
    virtual void train(const float* /*data*/, size_t /*n*/) {}

    // Insert one vector; returns its InternalId. Requires the index be trained
    // first if it needs training.
    virtual InternalId add(const float* vec) = 0;

    // Return up to K nearest neighbours to `query`, closest first.
    virtual std::vector<std::pair<InternalId, float>> search(const float* query,
                                                             size_t K) const = 0;

    virtual size_t size() const = 0;
    virtual size_t dim() const = 0;
};

}  // namespace vdb
