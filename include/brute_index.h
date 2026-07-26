#pragma once
#include <vector>

#include "index.h"

namespace vdb {

template <class Dist>
class BruteIndex : public Index {
public:
    explicit BruteIndex(size_t dim);

    InternalId add(const float* vec) override;
    std::vector<std::pair<InternalId, float>> search(const float* query,
                                                     size_t K) const override;
    size_t size() const override;
    size_t dim() const override;

    void serialize(std::vector<uint8_t>& out) const override;
    void deserialize(Reader& r) override;

private:
    size_t                          dim_;
    Dist                            dist_;
    std::vector<std::vector<float>> data_;
};

}
