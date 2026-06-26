#include "ivf_index.h"

#include <utility>

namespace vdb {

IVFIndex::IVFIndex(IVFConfig cfg, DistanceFn dist_fn)
    : config_(cfg), dist_fn_(std::move(dist_fn)) {}

void IVFIndex::train(const float* /*data*/, size_t /*n*/) {
    // TODO Stage 2: run k-means to fill centroids_ (nlist x dim), set trained_.
    trained_ = true;
}

InternalId IVFIndex::add(const float* /*vec*/) {
    // TODO Stage 2: find nearest centroid, append to its inverted list.
    return 0;
}

std::vector<std::pair<InternalId, float>> IVFIndex::search(const float* /*query*/,
                                                           size_t /*K*/) const {
    // TODO Stage 2: pick nprobe nearest centroids, scan their lists, top-K.
    return {};
}

size_t IVFIndex::size() const { return vectors_.size(); }
size_t IVFIndex::dim() const { return config_.dim; }

}  // namespace vdb
