#pragma once
#include <random>
#include <vector>

#include "index.h"

namespace vdb {

struct HNSWConfig {
    size_t dim;
    size_t M     = 16;
    size_t Mmax  = 16;
    size_t Mmax0 = 32;
    size_t ef    = 200;
    float  mL    = 0.0f;
};

class HNSWIndex : public Index {
public:
    HNSWIndex(HNSWConfig cfg, DistanceFn dist_fn);

    InternalId add(const float* vec) override;
    std::vector<std::pair<InternalId, float>> search(const float* query,
                                                     size_t K) const override;
    size_t size() const override;
    size_t dim() const override;

    void set_ef_search(size_t ef) { ef_search_ = ef; }

private:
    struct Node {
        std::vector<float>                    data;
        std::vector<std::vector<InternalId>>  neighbours;
    };

    HNSWConfig  config_;
    DistanceFn  dist_fn_;
    InternalId  entry_point_ = 0;
    int         max_layer_   = -1;
    size_t      ef_search_   = 50;
    mutable std::mt19937 rng_;

    std::vector<Node> nodes_;

    int sample_layer() const;
    std::vector<InternalId> search_layer(const float* q, InternalId ep, int layer_number) const;

    std::vector<InternalId> select_nearest(const float* q,
                                           std::vector<InternalId> cands,
                                           size_t M) const;

    template <typename Container, typename Compare>
    InternalId extremum_in(const float* q, const Container& container, Compare better) const {
        float best_dist = 0.0f;
        InternalId best_id = entry_point_;
        bool first = true;
        for (InternalId id : container) {
            float d = dist_fn_(q, nodes_[id].data.data(), config_.dim);
            if (first || better(d, best_dist)) {
                best_dist = d;
                best_id = id;
                first = false;
            }
        }
        return best_id;
    }
};

}
