#include "hnsw_index.h"

#include <cmath>
#include <functional>
#include <random>
#include <utility>
#include <unordered_set>

namespace vdb {

HNSWIndex::HNSWIndex(HNSWConfig cfg, DistanceFn dist_fn)
    : config_(cfg), dist_fn_(std::move(dist_fn)), rng_(std::random_device{}()) {
    if (config_.mL <= 0.0f) {
        config_.mL = 1.0f / std::log(static_cast<float>(config_.M));
    }
}


int HNSWIndex::sample_layer() const {
    std::uniform_real_distribution<double> u(0.0, 1.0);
    double r = -std::log(1.0 - u(rng_)) * config_.mL;
    return static_cast<int>(r);
}


std::vector<InternalId> HNSWIndex::search_layer(const float* q, int layer_number) const {
    std::unordered_set<InternalId> visited = {entry_point_};
    std::unordered_set<InternalId> candidates = {entry_point_};

    std::unordered_set<InternalId> found_neighbours = {entry_point_};

    while (!candidates.empty()) {
        InternalId c = extremum_in(q, candidates, std::less<>{});
        candidates.erase(c);


        InternalId f = extremum_in(q, found_neighbours, std::greater<>{});

        float dcq = dist_fn_(q, nodes_[c].data.data(), config_.dim);
        float dfq = dist_fn_(q, nodes_[f].data.data(), config_.dim);
        if(dcq > dfq) break;

        for(InternalId id : nodes_[c].neighbours[layer_number]) {
            if(visited.find(id) != visited.end()) continue;

            visited.insert(id);
            f = extremum_in(q, found_neighbours, std::greater<>{});
            dfq = dist_fn_(q, nodes_[f].data.data(), config_.dim);


            if(dist_fn_(nodes_[id].data.data(), q, config_.dim) >= dfq && found_neighbours.size() >= config_.ef) continue;

            candidates.insert(id);
            found_neighbours.insert(id);

            if(found_neighbours.size() > config_.ef) found_neighbours.erase(f);
        }
    }

    return std::vector<InternalId>(found_neighbours.begin(), found_neighbours.end());
}

InternalId HNSWIndex::add(const float* vec) {

    std::vector<InternalId> w;

    int ep_layer = 0;
    int l = sample_layer();

    for(int i = ep_layer; i <= l + 1; i++) {
        // search layer
    }
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
