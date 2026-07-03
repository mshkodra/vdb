#include "hnsw_index.h"

#include <algorithm>
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

std::vector<InternalId> HNSWIndex::search_layer(const float* q, InternalId ep, int layer_number) const {
    std::unordered_set<InternalId> visited = {ep};
    std::unordered_set<InternalId> candidates = {ep};

    std::unordered_set<InternalId> found_neighbours = {ep};

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

std::vector<InternalId> HNSWIndex::select_nearest(const float* q,
                                                  std::vector<InternalId> cands,
                                                  size_t M) const {
    if (cands.size() <= M) return cands;
    std::partial_sort(cands.begin(), cands.begin() + M, cands.end(),
                      [&](InternalId a, InternalId b) {
                          return dist_fn_(q, nodes_[a].data.data(), config_.dim) <
                                 dist_fn_(q, nodes_[b].data.data(), config_.dim);
                      });
    cands.resize(M);
    return cands;
}

InternalId HNSWIndex::add(const float* vec) {
    const InternalId id = static_cast<InternalId>(nodes_.size());
    const int l = sample_layer();

    Node node;
    node.data.assign(vec, vec + config_.dim);
    node.neighbours.resize(l + 1);
    nodes_.push_back(std::move(node));

    if (max_layer_ < 0) {
        entry_point_ = id;
        max_layer_ = l;
        return id;
    }

    const int L = max_layer_;
    InternalId ep = entry_point_;

    for (int lc = L; lc > l; lc--) {
        std::vector<InternalId> W = search_layer(vec, ep, lc);
        ep = extremum_in(vec, W, std::less<>{});
    }

    for (int lc = std::min(L, l); lc >= 0; lc--) {
        std::vector<InternalId> W = search_layer(vec, ep, lc);
        std::vector<InternalId> neighbours = select_nearest(vec, W, config_.M);

        const size_t Mmax = (lc == 0) ? config_.Mmax0 : config_.Mmax;

        nodes_[id].neighbours[lc] = neighbours;

        for (InternalId e : neighbours) {
            auto& e_conn = nodes_[e].neighbours[lc];
            e_conn.push_back(id);
            if (e_conn.size() > Mmax) {
                e_conn = select_nearest(nodes_[e].data.data(), e_conn, Mmax);
            }
        }

        ep = extremum_in(vec, W, std::less<>{});
    }

    if (l > L) {
        entry_point_ = id;
        max_layer_ = l;
    }

    return id;
}

std::vector<std::pair<InternalId, float>> HNSWIndex::search(const float* query,
                                                            size_t K) const {
    if (nodes_.empty()) return {};

    InternalId ep = entry_point_;
    for (int lc = max_layer_; lc > 0; lc--) {
        std::vector<InternalId> W = search_layer(query, ep, lc);
        ep = extremum_in(query, W, std::less<>{});
    }

    std::vector<InternalId> W = search_layer(query, ep, 0);

    std::vector<std::pair<InternalId, float>> results;
    results.reserve(W.size());
    for (InternalId id : W) {
        results.emplace_back(id, dist_fn_(query, nodes_[id].data.data(), config_.dim));
    }
    std::sort(results.begin(), results.end(),
              [](const auto& a, const auto& b) { return a.second < b.second; });
    if (results.size() > K) results.resize(K);
    return results;
}

size_t HNSWIndex::size() const { return nodes_.size(); }
size_t HNSWIndex::dim() const { return config_.dim; }

}
