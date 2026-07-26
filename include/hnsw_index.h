#pragma once
#include <memory>
#include <mutex>
#include <random>
#include <utility>
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
    unsigned seed = 42;  // level sampler RNG seed; fixed for reproducible builds
    // Fixed capacity. The node store is reserved to this once and never
    // reallocates, so concurrent readers can index existing nodes lock-free while
    // a writer appends. add() throws std::length_error past this. (Stage 7.)
    size_t max_elements = 1u << 20;
};

// Thread-safety (Stage 7). Concurrent add() and search() are safe:
//   - Node storage is reserved to `max_elements` and never reallocates, so an
//     existing node's address and immutable `data` are stable for lock-free reads.
//   - Each node carries its own mutex guarding its (mutable) neighbour lists.
//     Writers edit a neighbour list under that node's lock; readers copy it under
//     the same lock (the "B1" reader policy), then walk the copy unlocked.
//   - `grow_mutex_` serialises the append into `nodes_` (and the level RNG).
//   - `entry_mutex_` guards the entry point and max layer.
// At most one node lock is ever held at a time, so the graph edit is deadlock-free
// without a lock-ordering rule.
template <class Dist>
class HNSWIndex : public Index {
public:
    explicit HNSWIndex(HNSWConfig cfg);

    InternalId add(const float* vec) override;
    InternalId allocate(const float* vec) override;  // serial: reserve slot + store vector
    void       link(InternalId id) override;         // concurrent: wire into the graph
    std::vector<std::pair<InternalId, float>> search(const float* query,
                                                     size_t K) const override;
    size_t size() const override;
    size_t dim() const override;

    void serialize(std::vector<uint8_t>& out) const override;
    void deserialize(Reader& r) override;

    void set_ef_search(size_t ef) { ef_search_ = ef; }

private:
    struct Node {
        std::vector<float>                   data;        // immutable after allocate
        std::vector<std::vector<InternalId>> neighbours;  // outer sized once; inner mutable
        std::unique_ptr<std::mutex>          lock;        // guards `neighbours`
    };

    HNSWConfig  config_;
    Dist        dist_;
    InternalId  entry_point_ = 0;
    int         max_layer_   = -1;
    size_t      ef_search_   = 50;
    mutable std::mt19937 rng_;  // touched only under grow_mutex_

    std::vector<Node> nodes_;  // reserved to max_elements; never reallocates

    mutable std::mutex  grow_mutex_;   // serialises append into nodes_ + rng_; read by size()
    mutable std::mutex  entry_mutex_;  // guards entry_point_ + max_layer_

    int sample_layer_() const;  // caller must hold grow_mutex_

    // Serial half of add(): reserve the slot, store the vector, size the neighbour
    // arrays, sample the layer. Returns (internal id, top layer).
    std::pair<InternalId, int> allocate_node_(const float* vec);

    // Parallel half of add(): navigate the graph and wire the node in, under
    // per-node locks. Reads its own vector from nodes_[id].data.
    void link_node_(InternalId id);

    std::vector<InternalId> search_layer(const float* q, InternalId ep, size_t ef,
                                         int layer_number) const;

    std::vector<InternalId> select_neighbors(const float* q,
                                             std::vector<InternalId> cands,
                                             size_t M) const;

    // Nearest of `cands` to q (cands must be non-empty).
    InternalId closest_(const float* q, const std::vector<InternalId>& cands) const;
};

}
