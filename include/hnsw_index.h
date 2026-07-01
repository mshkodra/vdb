#pragma once
#include <random>
#include <vector>

#include "index.h"

namespace vdb {

// Hierarchical Navigable Small World graph index (HNSW) — ROADMAP Stage 3.
//
// Idea: a multi-layer proximity graph. Upper layers are sparse "express lanes"
// for long hops; lower layers are dense for fine-grained search. A query does a
// greedy descent from a single entry point down to layer 0. Built on the NSW /
// skip-list intuition (Malkov & Yashunin 2018).
struct HNSWConfig {
    size_t dim;
    size_t M     = 16;   // connections per node per layer
    size_t Mmax  = 16;   // max connections at layers > 0
    size_t Mmax0 = 32;   // max connections at layer 0 (usually 2*M)
    size_t ef    = 200;  // efConstruction — candidate width while building
    float  mL    = 0.0f; // level-generation factor; set to 1/ln(M) in ctor
};

class HNSWIndex : public Index {
public:
    HNSWIndex(HNSWConfig cfg, DistanceFn dist_fn);

    InternalId add(const float* vec) override;
    std::vector<std::pair<InternalId, float>> search(const float* query,
                                                     size_t K) const override;
    size_t size() const override;
    size_t dim() const override;

    // Search width knob, set per-query later. For now search() can use ef = K.
    void set_ef_search(size_t ef) { ef_search_ = ef; }

private:
    struct Node {
        std::vector<float>                    data;
        std::vector<std::vector<InternalId>>  neighbours;  // per layer
    };

    HNSWConfig  config_;
    DistanceFn  dist_fn_;
    InternalId  entry_point_ = 0;
    int         max_layer_   = -1;
    size_t      ef_search_   = 50;
    mutable std::mt19937 rng_;

    std::vector<Node> nodes_;  // TODO Stage 3

    int sample_layer() const;
    std::vector<InternalId> search_layer(const float* q, int layer_number) const;
    // std::vector<InternalId> select_neighbors() const;

    // Return the id in `container` that is the extreme (by `better`) w.r.t. its
    // distance to `q`. Pass std::less<>{} for nearest, std::greater<>{} for
    // furthest. Generic over the container: works for a set, a vector, anything
    // iterable of InternalId. Returns entry_point_ for an empty container.
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

}  // namespace vdb
