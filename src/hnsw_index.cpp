#include "hnsw_index.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <queue>
#include <random>
#include <stdexcept>
#include <utility>
#include <unordered_set>

#include "distance.h"

namespace vdb {

template <class Dist, class Elem>
HNSWIndex<Dist, Elem>::HNSWIndex(HNSWConfig cfg)
    : config_(cfg), rng_(cfg.seed) {
    if (config_.mL <= 0.0f) {
        config_.mL = 1.0f / std::log(static_cast<float>(config_.M));
    }
    // Reserve once so the store never reallocates: existing nodes keep their
    // address and their immutable `data` stays put for lock-free reads while a
    // concurrent writer appends.
    nodes_.reserve(config_.max_elements);
}

template <class Dist, class Elem>
void HNSWIndex<Dist, Elem>::train(const float* data, size_t n) {
    if constexpr (!std::is_same_v<Elem, float>) {
        float max_abs = 0.0f;
        const size_t total = n * config_.dim;
        for (size_t i = 0; i < total; ++i) max_abs = std::max(max_abs, std::fabs(data[i]));
        // Degenerate all-zero corpus guard: scale must stay positive (it's a
        // divisor in quantize_()).
        scale_   = max_abs > 0.0f ? max_abs / 127.0f : 1.0f;
        trained_ = true;
    }
}

template <class Dist, class Elem>
std::vector<Elem> HNSWIndex<Dist, Elem>::quantize_(const float* v) const {
    if constexpr (std::is_same_v<Elem, float>) {
        return std::vector<Elem>(v, v + config_.dim);
    } else {
        std::vector<Elem> out(config_.dim);
        for (size_t i = 0; i < config_.dim; ++i) {
            const float scaled  = v[i] / scale_;
            const float clamped = std::clamp(scaled, -127.0f, 127.0f);
            out[i] = static_cast<Elem>(std::lround(clamped));
        }
        return out;
    }
}

template <class Dist, class Elem>
int HNSWIndex<Dist, Elem>::sample_layer_() const {
    std::uniform_real_distribution<double> u(0.0, 1.0);
    double r = -std::log(1.0 - u(rng_)) * config_.mL;
    return static_cast<int>(r);
}

template <class Dist, class Elem>
InternalId HNSWIndex<Dist, Elem>::closest_(const Elem* q,
                                          const std::vector<InternalId>& cands) const {
    float      best_dist = static_cast<float>(dist_(q, nodes_[cands[0]].data.data(), config_.dim));
    InternalId best_id   = cands[0];
    for (size_t i = 1; i < cands.size(); ++i) {
        const float d = static_cast<float>(dist_(q, nodes_[cands[i]].data.data(), config_.dim));
        if (d < best_dist) {
            best_dist = d;
            best_id   = cands[i];
        }
    }
    return best_id;
}

template <class Dist, class Elem>
std::vector<InternalId> HNSWIndex<Dist, Elem>::search_layer(const Elem* q, InternalId ep,
                                                            size_t ef, int layer_number) const {
    using DI = std::pair<float, InternalId>;  // (distance to q, node id)

    std::unordered_set<InternalId> visited{ep};
    // candidates: min-heap on distance (nearest to expand next on top).
    std::priority_queue<DI, std::vector<DI>, std::greater<>> candidates;
    // W: max-heap on distance (current worst result on top, so we can evict it).
    std::priority_queue<DI> W;

    const float d_ep = static_cast<float>(dist_(q, nodes_[ep].data.data(), config_.dim));
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

            const float d = static_cast<float>(dist_(q, nodes_[id].data.data(), config_.dim));
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
template <class Dist, class Elem>
std::vector<InternalId> HNSWIndex<Dist, Elem>::select_neighbors(const Elem* q,
                                                                std::vector<InternalId> cands,
                                                                size_t M) const {
    if (cands.size() <= M) return cands;

    // Process candidates nearest-to-q first.
    std::sort(cands.begin(), cands.end(), [&](InternalId a, InternalId b) {
        return dist_(q, nodes_[a].data.data(), config_.dim) <
               dist_(q, nodes_[b].data.data(), config_.dim);
    });

    std::vector<InternalId> result;
    std::vector<InternalId> discarded;
    result.reserve(M);
    for (InternalId e : cands) {
        if (result.size() >= M) break;
        const float d_eq = static_cast<float>(dist_(q, nodes_[e].data.data(), config_.dim));
        bool closer_to_q = true;
        for (InternalId r : result) {
            const float d_er = static_cast<float>(
                dist_(nodes_[e].data.data(), nodes_[r].data.data(), config_.dim));
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

template <class Dist, class Elem>
std::pair<InternalId, int> HNSWIndex<Dist, Elem>::allocate_node_(const float* vec) {
    if constexpr (!std::is_same_v<Elem, float>) {
        if (!trained_)
            throw std::runtime_error(
                "HNSWIndex<Dist, int8_t>::allocate: train() must run before the first "
                "insert (needs a calibrated quantization scale — see the class doc comment)");
    }
    std::lock_guard<std::mutex> g(grow_mutex_);
    if (nodes_.size() >= config_.max_elements)
        throw std::length_error("HNSWIndex: max_elements exceeded");

    const InternalId id = static_cast<InternalId>(nodes_.size());
    const int        l  = sample_layer_();

    // emplace into reserved capacity: constructs in place, no reallocation.
    nodes_.emplace_back();
    Node& n = nodes_.back();
    n.data  = quantize_(vec);
    n.neighbours.resize(l + 1);  // sized once here; never resized again
    n.lock = std::make_unique<std::mutex>();
    return {id, l};
}

template <class Dist, class Elem>
void HNSWIndex<Dist, Elem>::link_node_(InternalId id) {
    const Elem* vec = nodes_[id].data.data();            // stable, immutable
    const int   l   = static_cast<int>(nodes_[id].neighbours.size()) - 1;

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

template <class Dist, class Elem>
InternalId HNSWIndex<Dist, Elem>::allocate(const float* vec) { return allocate_node_(vec).first; }

template <class Dist, class Elem>
void HNSWIndex<Dist, Elem>::link(InternalId id) { link_node_(id); }

template <class Dist, class Elem>
InternalId HNSWIndex<Dist, Elem>::add(const float* vec) {
    const InternalId id = allocate(vec);
    link(id);
    return id;
}

template <class Dist, class Elem>
std::vector<std::pair<InternalId, float>> HNSWIndex<Dist, Elem>::search(const float* query,
                                                                        size_t K) const {
    InternalId ep;
    int        L;
    {
        std::lock_guard<std::mutex> g(entry_mutex_);
        if (max_layer_ < 0) return {};  // empty graph
        ep = entry_point_;
        L  = max_layer_;
    }

    // Query crosses into the index's stored representation exactly once here —
    // everything downstream (search_layer, closest_) operates on the quantized
    // buffer, never re-touching `query` itself.
    const std::vector<Elem> q  = quantize_(query);
    const Elem*              qp = q.data();

    for (int lc = L; lc > 0; --lc) {
        std::vector<InternalId> W = search_layer(qp, ep, 1, lc);
        if (!W.empty()) ep = closest_(qp, W);
    }

    std::vector<InternalId> W = search_layer(qp, ep, std::max(ef_search_, K), 0);

    std::vector<std::pair<InternalId, float>> results;
    results.reserve(W.size());
    for (InternalId id : W) {
        // scale_*scale_ rescales the kernel's raw (unscaled) squared distance back
        // to a magnitude comparable with plain float L2 — see distance.h's L2Int8
        // comment. A no-op for Elem=float (scale_ is always 1.0 there): one
        // multiply, not a second code path.
        const float raw = static_cast<float>(dist_(qp, nodes_[id].data.data(), config_.dim));
        results.emplace_back(id, raw * scale_ * scale_);
    }
    std::sort(results.begin(), results.end(),
              [](const auto& a, const auto& b) { return a.second < b.second; });
    if (results.size() > K) results.resize(K);
    return results;
}

template <class Dist, class Elem>
size_t HNSWIndex<Dist, Elem>::size() const {
    std::lock_guard<std::mutex> g(grow_mutex_);
    return nodes_.size();
}

template <class Dist, class Elem>
size_t HNSWIndex<Dist, Elem>::dim() const { return config_.dim; }

template <class Dist, class Elem>
void HNSWIndex<Dist, Elem>::serialize(std::vector<uint8_t>& out) const {
    put<uint32_t>(out, entry_point_);
    put<int32_t>(out, max_layer_);
    put<uint64_t>(out, ef_search_);
    // Elem=float's wire format is untouched byte-for-byte by design — scale_/
    // trained_ are only meaningful for a quantized index, and existing cached
    // snapshots (e.g. data/sift/cache/hnsw.snap) predate this field entirely.
    if constexpr (!std::is_same_v<Elem, float>) {
        put<float>(out, scale_);
        put<uint8_t>(out, trained_ ? 1 : 0);
    }
    put<uint64_t>(out, nodes_.size());
    for (const auto& node : nodes_) {
        if constexpr (std::is_same_v<Elem, float>) {
            put_floats(out, node.data);
        } else {
            put<uint64_t>(out, node.data.size());
            const auto* p = reinterpret_cast<const uint8_t*>(node.data.data());
            out.insert(out.end(), p, p + node.data.size() * sizeof(Elem));
        }
        put<uint64_t>(out, node.neighbours.size());
        for (const auto& layer : node.neighbours) {
            put<uint64_t>(out, layer.size());
            for (InternalId id : layer) put<uint32_t>(out, id);
        }
    }
}

template <class Dist, class Elem>
void HNSWIndex<Dist, Elem>::deserialize(Reader& r) {
    entry_point_     = r.get<uint32_t>();
    max_layer_       = r.get<int32_t>();
    ef_search_       = r.get<uint64_t>();
    if constexpr (!std::is_same_v<Elem, float>) {
        scale_   = r.get<float>();
        trained_ = r.get<uint8_t>() != 0;
    }
    const uint64_t n = r.get<uint64_t>();
    if (n > config_.max_elements)
        throw std::length_error("HNSWIndex::deserialize: node count exceeds max_elements");

    nodes_.clear();
    nodes_.reserve(config_.max_elements);
    for (uint64_t i = 0; i < n; ++i) {
        nodes_.emplace_back();
        Node& node = nodes_.back();
        if constexpr (std::is_same_v<Elem, float>) {
            node.data = r.get_floats();
        } else {
            const uint64_t m = r.get<uint64_t>();
            node.data.resize(m);
            for (uint64_t j = 0; j < m; ++j) node.data[j] = r.get<Elem>();
        }
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

// One materialized index type per metric functor (see brute_index.cpp) — plus the
// int8-quantized L2 variant this file adds.
template class HNSWIndex<L2>;
template class HNSWIndex<InnerProduct>;
template class HNSWIndex<Cosine>;
template class HNSWIndex<L2Int8, int8_t>;

}
