#include "hnsw_index.h"

#include <cmath>
#include <random>
#include <utility>

namespace vdb {

HNSWIndex::HNSWIndex(HNSWConfig cfg, DistanceFn dist_fn)
    : config_(cfg), dist_fn_(std::move(dist_fn)), rng_(std::random_device{}()) {
    if (config_.mL <= 0.0f) {
        config_.mL = 1.0f / std::log(static_cast<float>(config_.M));
    }
}

InternalId HNSWIndex::add(const float* /*vec*/) {
    // TODO Stage 3: sample layer, greedy-descent entry, connect neighbours.
    return 0;
}

std::vector<std::pair<InternalId, float>> HNSWIndex::search(const float* /*query*/,
                                                            size_t /*K*/) const {
    // TODO Stage 3: greedy descent through layers, ef-search at layer 0.
    return {};
}

size_t HNSWIndex::size() const { return nodes_.size(); }
size_t HNSWIndex::dim() const { return config_.dim; }

}  // namespace vdb
