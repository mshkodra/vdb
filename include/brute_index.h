#pragma once
#include <vector>

#include "index.h"

namespace vdb {

// Exact, linear-scan index. Slow but always correct — it is the GROUND TRUTH
// oracle you measure HNSW/IVF recall against (ROADMAP Stage 1). Implement this
// first; everything else is judged relative to it.
class BruteIndex : public Index {
public:
    BruteIndex(size_t dim, DistanceFn dist_fn);

    InternalId add(const float* vec) override;
    std::vector<std::pair<InternalId, float>> search(const float* query,
                                                     size_t K) const override;
    size_t size() const override;
    size_t dim() const override;

private:
    size_t                          dim_;
    DistanceFn                      dist_fn_;
    std::vector<std::vector<float>> data_;  // TODO Stage 1: implement
};

}  // namespace vdb
