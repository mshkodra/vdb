#pragma once
#include <vector>

#include "index.h"

namespace vdb {

struct IVFConfig {
    size_t dim;
    size_t nlist  = 100;
    size_t nprobe = 8;
    size_t kmeans_iters = 25;
};

template <class Dist>
class IVFIndex : public Index {
public:
    explicit IVFIndex(IVFConfig cfg);

    void train(const float* data, size_t n) override;

    InternalId add(const float* vec) override;
    std::vector<std::pair<InternalId, float>> search(const float* query,
                                                     size_t K) const override;

    void set_nprobe(size_t np) { config_.nprobe = np; }
    size_t size() const override;
    size_t dim() const override;

    void serialize(std::vector<uint8_t>& out) const override;
    void deserialize(Reader& r) override;

private:
    IVFConfig   config_;
    Dist        dist_;
    bool        trained_ = false;

    std::vector<std::vector<float>>             centroids_;
    std::vector<std::vector<InternalId>>        inverted_lists_;
    std::vector<std::vector<float>>             vectors_;

    int nearest_centroid(const float* v) const;
};

}
