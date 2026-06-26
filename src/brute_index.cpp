#include "brute_index.h"

#include <utility>

namespace vdb {

BruteIndex::BruteIndex(size_t dim, DistanceFn dist_fn)
    : dim_(dim), dist_fn_(std::move(dist_fn)) {}

InternalId BruteIndex::add(const float* /*vec*/) {
    // TODO Stage 1: copy vec into data_, return its index.
    return 0;
}

std::vector<std::pair<InternalId, float>> BruteIndex::search(const float* /*query*/,
                                                             size_t /*K*/) const {
    // TODO Stage 1: score every stored vector, partial_sort, return top-K.
    return {};
}

size_t BruteIndex::size() const { return data_.size(); }
size_t BruteIndex::dim() const { return dim_; }

}  // namespace vdb
