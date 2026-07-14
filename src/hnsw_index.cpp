#include "hnsw_index.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <queue>
#include <random>
#include <utility>
#include <unordered_set>

namespace vdb {

HNSWIndex::HNSWIndex(HNSWConfig cfg, DistanceFn dist_fn)
    : config_(cfg), dist_fn_(std::move(dist_fn)), rng_(cfg.seed) {
    if (config_.mL <= 0.0f) {
        config_.mL = 1.0f / std::log(static_cast<float>(config_.M));
    }
}

int HNSWIndex::sample_layer() const {
    std::uniform_real_distribution<double> u(0.0, 1.0);
    double r = -std::log(1.0 - u(rng_)) * config_.mL;
    return static_cast<int>(r);
}

std::vector<InternalId> HNSWIndex::search_layer(const float* q, InternalId ep,
                                                size_t ef, int layer_number) const {
    using DI = std::pair<float, InternalId>;  // (distance to q, node id)

    std::unordered_set<InternalId> visited{ep};
    // candidates: min-heap on distance (nearest to expand next on top).
    std::priority_queue<DI, std::vector<DI>, std::greater<>> candidates;
    // W: max-heap on distance (current worst result on top, so we can evict it).
    std::priority_queue<DI> W;

    const float d_ep = dist_fn_(q, nodes_[ep].data.data(), config_.dim);
    candidates.emplace(d_ep, ep);
    W.emplace(d_ep, ep);

    while (!candidates.empty()) {
        auto [dc, c] = candidates.top();
        candidates.pop();
        if (dc > W.top().first) break;  // nearest candidate farther than worst result

        for (InternalId id : nodes_[c].neighbours[layer_number]) {
            if (!visited.insert(id).second) continue;

            const float d = dist_fn_(q, nodes_[id].data.data(), config_.dim);
            if (W.size() < ef || d < W.top().first) {
                candidates.emplace(d, id);
                W.emplace(d, id);
                if (W.size() > ef) W.pop();
            }
        }
    }

    std::vector<InternalId> result;
    result.reserve(W.size());
    while (!W.empty()) {
        result.push_back(W.top().second);
        W.pop();
    }
    return result;
}

// Malkov Alg. 4: keep a candidate only if it is closer to the base q than to any
// already-selected neighbour. This spreads links across *directions* instead of
// piling them onto the M nearest (which, in a tight cluster, all point inward and
// leave the cluster poorly bridged). keepPrunedConnections tops the result back up
// to M from the discarded set so connectivity is preserved.
std::vector<InternalId> HNSWIndex::select_neighbors(const float* q,
                                                    std::vector<InternalId> cands,
                                                    size_t M) const {
    if (cands.size() <= M) return cands;

    // Process candidates nearest-to-q first.
    std::sort(cands.begin(), cands.end(), [&](InternalId a, InternalId b) {
        return dist_fn_(q, nodes_[a].data.data(), config_.dim) <
               dist_fn_(q, nodes_[b].data.data(), config_.dim);
    });

    std::vector<InternalId> result;
    std::vector<InternalId> discarded;
    result.reserve(M);
    for (InternalId e : cands) {
        if (result.size() >= M) break;
        const float d_eq = dist_fn_(q, nodes_[e].data.data(), config_.dim);
        bool closer_to_q = true;
        for (InternalId r : result) {
            const float d_er = dist_fn_(nodes_[e].data.data(),
                                        nodes_[r].data.data(), config_.dim);
            if (d_er < d_eq) {  // e sits nearer an existing neighbour than to q
                closer_to_q = false;
                break;
            }
        }
        (closer_to_q ? result : discarded).push_back(e);
    }

    for (InternalId e : discarded) {  // keepPrunedConnections
        if (result.size() >= M) break;
        result.push_back(e);
    }
    return result;
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
        std::vector<InternalId> W = search_layer(vec, ep, 1, lc);
        ep = extremum_in(vec, W, std::less<>{});
    }

    for (int lc = std::min(L, l); lc >= 0; lc--) {
        std::vector<InternalId> W = search_layer(vec, ep, config_.ef, lc);
        std::vector<InternalId> neighbours = select_neighbors(vec, W, config_.M);

        const size_t Mmax = (lc == 0) ? config_.Mmax0 : config_.Mmax;

        nodes_[id].neighbours[lc] = neighbours;

        for (InternalId e : neighbours) {
            auto& e_conn = nodes_[e].neighbours[lc];
            e_conn.push_back(id);
            if (e_conn.size() > Mmax) {
                e_conn = select_neighbors(nodes_[e].data.data(), e_conn, Mmax);
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
        std::vector<InternalId> W = search_layer(query, ep, 1, lc);
        ep = extremum_in(query, W, std::less<>{});
    }

    std::vector<InternalId> W = search_layer(query, ep, std::max(ef_search_, K), 0);

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
