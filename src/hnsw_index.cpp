#include "hnsw_index.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <queue>
#include <random>
#include <stdexcept>
#include <utility>
#include <unordered_set>

namespace vdb {

HNSWIndex::HNSWIndex(HNSWConfig cfg, DistanceFn dist_fn)
    : config_(cfg), dist_fn_(std::move(dist_fn)), rng_(cfg.seed) {
    if (config_.mL <= 0.0f) {
        config_.mL = 1.0f / std::log(static_cast<float>(config_.M));
    }
    // Reserve once so the store never reallocates: existing nodes keep their
    // address and their immutable `data` stays put for lock-free reads while a
    // concurrent writer appends into a fresh slot.
    nodes_.reserve(config_.max_elements);
}

int HNSWIndex::sample_layer_() const {
    std::uniform_real_distribution<double> u(0.0, 1.0);
    double r = -std::log(1.0 - u(rng_)) * config_.mL;
    return static_cast<int>(r);
}

InternalId HNSWIndex::closest_(const float* q,
                               const std::vector<InternalId>& cands) const {
    float      best_dist = dist_fn_(q, nodes_[cands[0]].data.data(), config_.dim);
    InternalId best_id   = cands[0];
    for (size_t i = 1; i < cands.size(); ++i) {
        const float d = dist_fn_(q, nodes_[cands[i]].data.data(), config_.dim);
        if (d < best_dist) {
            best_dist = d;
            best_id   = cands[i];
        }
    }
    return best_id;
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

        // B1 reader policy: copy c's neighbour list under c's lock, then release
        // and walk the copy. The outer `neighbours` vector is sized once at
        // allocate and never resized, so reading its size lock-free is safe; only
        // the inner list is mutable, so it is copied under the lock.
        std::vector<InternalId> nbrs;
        {
            std::lock_guard<std::mutex> g(*nodes_[c].lock);
            if (layer_number < static_cast<int>(nodes_[c].neighbours.size()))
                nbrs = nodes_[c].neighbours[layer_number];
        }

        for (InternalId id : nbrs) {
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

std::pair<InternalId, int> HNSWIndex::allocate_node_(const float* vec) {
    std::lock_guard<std::mutex> g(grow_mutex_);
    if (nodes_.size() >= config_.max_elements)
        throw std::length_error("HNSWIndex: max_elements exceeded");

    const InternalId id = static_cast<InternalId>(nodes_.size());
    const int        l  = sample_layer_();

    // emplace into reserved capacity: constructs in place, no reallocation.
    nodes_.emplace_back();
    Node& n = nodes_.back();
    n.data.assign(vec, vec + config_.dim);
    n.neighbours.resize(l + 1);  // sized once here; never resized again
    n.lock = std::make_unique<std::mutex>();
    return {id, l};
}

void HNSWIndex::link_node_(InternalId id) {
    const float* vec = nodes_[id].data.data();          // stable, immutable
    const int    l   = static_cast<int>(nodes_[id].neighbours.size()) - 1;

    // Snapshot the entry point / max layer. If the graph is empty this node
    // becomes the entry with no links to make.
    InternalId ep;
    int        L;
    {
        std::lock_guard<std::mutex> g(entry_mutex_);
        if (max_layer_ < 0) {
            entry_point_ = id;
            max_layer_   = l;
            return;
        }
        ep = entry_point_;
        L  = max_layer_;
    }

    // Greedy descent (ef=1) through the layers above this node's top layer.
    for (int lc = L; lc > l; --lc) {
        std::vector<InternalId> W = search_layer(vec, ep, 1, lc);
        if (!W.empty()) ep = closest_(vec, W);
    }

    // Wire in at every layer from min(L, l) down to 0.
    for (int lc = std::min(L, l); lc >= 0; --lc) {
        std::vector<InternalId> W          = search_layer(vec, ep, config_.ef, lc);
        std::vector<InternalId> neighbours = select_neighbors(vec, W, config_.M);
        const size_t            Mmax       = (lc == 0) ? config_.Mmax0 : config_.Mmax;

        // Set this node's own outgoing links (under its own lock).
        {
            std::lock_guard<std::mutex> g(*nodes_[id].lock);
            nodes_[id].neighbours[lc] = neighbours;
        }

        // Add the back-link on each neighbour, pruning if it overflows. Only the
        // neighbour's lock is held here (one lock at a time → no deadlock); the
        // prune reads other nodes' immutable `data`, never their locks.
        for (InternalId e : neighbours) {
            std::lock_guard<std::mutex> g(*nodes_[e].lock);
            auto& e_conn = nodes_[e].neighbours[lc];
            e_conn.push_back(id);
            if (e_conn.size() > Mmax)
                e_conn = select_neighbors(nodes_[e].data.data(), e_conn, Mmax);
        }

        if (!W.empty()) ep = closest_(vec, W);
    }

    // Promote to entry point if taller than the current max (re-check under lock).
    if (l > L) {
        std::lock_guard<std::mutex> g(entry_mutex_);
        if (l > max_layer_) {
            max_layer_   = l;
            entry_point_ = id;
        }
    }
}

InternalId HNSWIndex::add(const float* vec) {
    auto [id, l] = allocate_node_(vec);
    (void)l;
    link_node_(id);
    return id;
}

std::vector<std::pair<InternalId, float>> HNSWIndex::search(const float* query,
                                                            size_t K) const {
    InternalId ep;
    int        L;
    {
        std::lock_guard<std::mutex> g(entry_mutex_);
        if (max_layer_ < 0) return {};  // empty graph
        ep = entry_point_;
        L  = max_layer_;
    }

    for (int lc = L; lc > 0; --lc) {
        std::vector<InternalId> W = search_layer(query, ep, 1, lc);
        if (!W.empty()) ep = closest_(query, W);
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

size_t HNSWIndex::size() const {
    std::lock_guard<std::mutex> g(grow_mutex_);
    return nodes_.size();
}

size_t HNSWIndex::dim() const { return config_.dim; }

void HNSWIndex::serialize(std::vector<uint8_t>& out) const {
    put<uint32_t>(out, entry_point_);
    put<int32_t>(out, max_layer_);
    put<uint64_t>(out, ef_search_);
    put<uint64_t>(out, nodes_.size());
    for (const auto& node : nodes_) {
        put_floats(out, node.data);
        put<uint64_t>(out, node.neighbours.size());
        for (const auto& layer : node.neighbours) {
            put<uint64_t>(out, layer.size());
            for (InternalId id : layer) put<uint32_t>(out, id);
        }
    }
}

void HNSWIndex::deserialize(Reader& r) {
    entry_point_     = r.get<uint32_t>();
    max_layer_       = r.get<int32_t>();
    ef_search_       = r.get<uint64_t>();
    const uint64_t n = r.get<uint64_t>();
    if (n > config_.max_elements)
        throw std::length_error("HNSWIndex::deserialize: node count exceeds max_elements");

    nodes_.clear();
    nodes_.reserve(config_.max_elements);
    for (uint64_t i = 0; i < n; ++i) {
        nodes_.emplace_back();
        Node& node = nodes_.back();
        node.data  = r.get_floats();
        const uint64_t nl = r.get<uint64_t>();
        node.neighbours.resize(nl);
        for (uint64_t l = 0; l < nl; ++l) {
            const uint64_t m = r.get<uint64_t>();
            node.neighbours[l].resize(m);
            for (uint64_t j = 0; j < m; ++j) node.neighbours[l][j] = r.get<uint32_t>();
        }
        node.lock = std::make_unique<std::mutex>();
    }
}

}
