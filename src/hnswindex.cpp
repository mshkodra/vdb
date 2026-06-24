#include <hnswindex.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <unordered_set>
#include <utility>

HNSWIndex::Node::Node(const float* vec, size_t dim, int max_layer, size_t M, size_t Mmax0) {
    data.assign(vec, vec + dim);
    neighbours.resize(max_layer + 1);
    neighbours[0].reserve(Mmax0);

    for (int l = 1; l <= max_layer; l++) neighbours[l].reserve(M);
}

HNSWIndex::HNSWIndex(HNSWConfig cfg, DistanceFn dist_fn)
    : config_(cfg),
      dist_fn_(std::move(dist_fn)),
      rng_(std::random_device{}()) {
    if (config_.mL <= 0.0f) {
        config_.mL = 1.0f / std::log(static_cast<float>(config_.M));
    }
}

int HNSWIndex::sample_layer() const {
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    float r;
    do { r = u(rng_); } while (r <= 0.0f);
    return static_cast<int>(std::floor(-std::log(r) * config_.mL));
}

float HNSWIndex::distance(InternalId a, InternalId b) const {
    return dist_fn_(node_pool_[a].data.data(),
                    node_pool_[b].data.data(),
                    config_.dim);
}

float HNSWIndex::distance(const float* q, InternalId b) const {
    return dist_fn_(q, node_pool_[b].data.data(), config_.dim);
}

const float* HNSWIndex::get(InternalId id) const {
    return node_pool_[id].data.data();
}

void HNSWIndex::mark_deleted(InternalId id) {
    node_pool_[id].deleted = true;
}

bool HNSWIndex::is_deleted(InternalId id) const {
    return node_pool_[id].deleted;
}

std::vector<InternalId> HNSWIndex::search_layer(const float* q,
                                                const std::vector<InternalId>& eps,
                                                size_t ef, int layer) const {
    using Pair = std::pair<float, InternalId>;

    std::unordered_set<InternalId> visited;
    visited.reserve(ef * 4);

    // candidates: min-heap by distance (smallest on top)
    std::priority_queue<Pair, std::vector<Pair>, std::greater<Pair>> candidates;
    // W: max-heap by distance (largest on top) — current best-of-ef
    std::priority_queue<Pair> W;

    for (InternalId ep : eps) {
        if (!visited.insert(ep).second) continue;
        const float d = distance(q, ep);
        candidates.emplace(d, ep);
        W.emplace(d, ep);
    }
    while (W.size() > ef) W.pop();

    while (!candidates.empty()) {
        auto [d_c, c] = candidates.top();
        candidates.pop();

        if (W.empty()) break;
        const float d_f = W.top().first;
        if (d_c > d_f) break;

        const Node& cnode = node_pool_[c];
        if (layer >= static_cast<int>(cnode.neighbours.size())) continue;

        for (InternalId e : cnode.neighbours[layer]) {
            if (!visited.insert(e).second) continue;

            const float d_e = distance(q, e);
            const float d_top = W.empty() ? std::numeric_limits<float>::infinity()
                                          : W.top().first;
            if (W.size() < ef || d_e < d_top) {
                candidates.emplace(d_e, e);
                W.emplace(d_e, e);
                if (W.size() > ef) W.pop();
            }
        }
    }

    // Drain max-heap into a vector (largest-first), then reverse to closest-first.
    std::vector<Pair> tmp;
    tmp.reserve(W.size());
    while (!W.empty()) {
        tmp.push_back(W.top());
        W.pop();
    }
    std::reverse(tmp.begin(), tmp.end());

    std::vector<InternalId> ids;
    ids.reserve(tmp.size());
    for (const auto& p : tmp) ids.push_back(p.second);
    return ids;
}

std::vector<InternalId> HNSWIndex::select_neighbors(const float* q,
                                                    const std::vector<InternalId>& candidates,
                                                    size_t M, int /*layer*/) const {
    // HNSW paper Algorithm 4: heuristic neighbor selection with keepPrunedConnections.
    // Add a candidate e to R only if e is closer to q than to any r already in R —
    // this keeps neighbors spread across directions instead of clustering. Pruned
    // candidates are appended to fill any remaining slots so connectivity is preserved.
    std::vector<std::pair<float, InternalId>> sorted_cands;
    sorted_cands.reserve(candidates.size());
    for (InternalId c : candidates) {
        sorted_cands.emplace_back(distance(q, c), c);
    }
    std::sort(sorted_cands.begin(), sorted_cands.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    std::vector<InternalId> R;
    R.reserve(M);
    std::vector<std::pair<float, InternalId>> discarded;

    for (const auto& [d_eq, e] : sorted_cands) {
        if (R.size() >= M) break;

        bool keep = true;
        const float* e_data = node_pool_[e].data.data();
        for (InternalId r : R) {
            const float d_er = dist_fn_(e_data,
                                        node_pool_[r].data.data(),
                                        config_.dim);
            if (d_er < d_eq) { keep = false; break; }
        }

        if (keep) R.push_back(e);
        else      discarded.emplace_back(d_eq, e);
    }

    // keepPrunedConnections: fill any remaining slots with the closest discarded.
    for (const auto& [_, e] : discarded) {
        if (R.size() >= M) break;
        R.push_back(e);
    }

    return R;
}

InternalId HNSWIndex::insert(const float* vec) {
    const int L = sample_layer();
    const InternalId new_id = static_cast<InternalId>(node_pool_.size());
    node_pool_.emplace_back(vec, config_.dim, L, config_.M, config_.Mmax0);

    if (max_layer_ == -1) {
        entry_point_ = new_id;
        max_layer_   = L;
        return new_id;
    }

    std::vector<InternalId> eps = {entry_point_};
    const int top = max_layer_;

    // Greedy descent through layers above the new node's top layer.
    for (int layer = top; layer > L; --layer) {
        auto W = search_layer(vec, eps, /*ef=*/1, layer);
        if (!W.empty()) eps = {W.front()};
    }

    // Insert at every layer from min(L, top) down to 0.
    const int start = std::min(L, top);
    for (int layer = start; layer >= 0; --layer) {
        auto W = search_layer(vec, eps, config_.ef, layer);
        auto neighbors = select_neighbors(vec, W, config_.M, layer);

        const size_t Mmax_layer = (layer == 0) ? config_.Mmax0 : config_.Mmax;

        for (InternalId n : neighbors) {
            node_pool_[new_id].neighbours[layer].push_back(n);
            node_pool_[n].neighbours[layer].push_back(new_id);

            auto& n_neighbors = node_pool_[n].neighbours[layer];
            if (n_neighbors.size() > Mmax_layer) {
                const float* n_data = node_pool_[n].data.data();
                auto pruned = select_neighbors(n_data, n_neighbors, Mmax_layer, layer);
                n_neighbors = std::move(pruned);
            }
        }

        eps = std::move(W);  // carry the full result set as entry points for next layer
    }

    if (L > max_layer_) {
        max_layer_   = L;
        entry_point_ = new_id;
    }

    return new_id;
}

std::vector<InternalId> HNSWIndex::search(const float* query, size_t K, size_t ef) const {
    if (max_layer_ == -1 || node_pool_.empty()) return {};

    std::vector<InternalId> eps = {entry_point_};
    for (int layer = max_layer_; layer > 0; --layer) {
        auto W = search_layer(query, eps, /*ef=*/1, layer);
        if (!W.empty()) eps = {W.front()};
    }

    // Collect ef >= K candidates ordered closest-first, then drop tombstoned nodes
    // and keep the K nearest survivors. Crucially, the traversal above descended
    // *through* deleted nodes — only the final result set excludes them. The ef
    // buffer is what lets us still return K live results when some are tombstoned.
    auto W = search_layer(query, eps, std::max(ef, K), 0);

    std::vector<InternalId> result;
    result.reserve(std::min(W.size(), K));
    for (InternalId id : W) {
        if (node_pool_[id].deleted) continue;
        result.push_back(id);
        if (result.size() == K) break;
    }
    return result;
}
