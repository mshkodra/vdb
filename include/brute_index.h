#pragma once
#include <vector>

#include "index.h"

namespace vdb {

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
    std::vector<std::vector<float>> data_;
};

}  // namespace vdb
