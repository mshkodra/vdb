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

    // int sample_layer() const;
    // std::vector<InternalId> search_layer(...) const;
    // std::vector<InternalId> select_neighbors(...) const;
};

}  // namespace vdb
