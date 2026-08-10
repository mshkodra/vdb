#pragma once
#include <memory>
#include <mutex>
#include <random>
#include <type_traits>
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
//
// `Elem` (default float): what a node's stored vector is made of. This exists for
// scalar quantization — HNSWIndex<L2Int8, int8_t> stores int8_t in memory instead
// of float, cutting a node's vector footprint 4x, while every insert/search
// entry point still takes/returns `float*`/`float` per the Index interface (the
// float<->Elem conversion happens once per call, at quantize_()). For Elem=float
// this is all a no-op identity path — zero behavior change from before this
// parameter existed, which is why every existing call site (HNSWIndex<L2>, relying
// on the default) compiles and runs unchanged.
//
// Quantization is symmetric and dataset-wide, not per-vector: train() computes one
// scale = max(|value|) / 127 across a calibration batch, so every stored vector and
// every query share the same affine mapping. That's what makes int8-space squared
// L2 an exact rescaling of true squared L2 (see distance.h's L2Int8 comment) —
// per-vector scales would break that property. Consequence a caller must accept:
// train() must run before any insert when Elem != float (add()/allocate() throw
// otherwise), and an insert whose values fall outside the calibrated range gets
// clipped, not re-calibrated — recalibrating would require requantizing every
// already-stored vector.
template <class Dist, class Elem = float>
class HNSWIndex : public Index {
public:
    explicit HNSWIndex(HNSWConfig cfg);

    // Elem != float only: computes the single dataset-wide quantization scale from
    // `data` (n vectors of dim config_.dim, row-major) — must be called before the
    // first add()/allocate(). A no-op for Elem=float (matches the Index base
    // class's default train(), just made explicit here rather than inherited, so
    // the "must train first" contract for Elem=int8_t is enforced in one place).
    void train(const float* data, size_t n) override;

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
        std::vector<Elem>                    data;        // immutable after allocate
        std::vector<std::vector<InternalId>> neighbours;  // outer sized once; inner mutable
        std::unique_ptr<std::mutex>          lock;        // guards `neighbours`
    };

    HNSWConfig  config_;
    Dist        dist_;
    InternalId  entry_point_ = 0;
    int         max_layer_   = -1;
    size_t      ef_search_   = 50;
    mutable std::mt19937 rng_;  // touched only under grow_mutex_

    // Elem != float only (see the class comment for why quantization is dataset-
    // wide, not per-vector): scale_ maps a raw float value to its int8 code via
    // round(v / scale_), clamped to [-127, 127]. trained_ gates add()/allocate()
    // until train() has actually run. Both are always present (even for Elem=float,
    // where they're inert — scale_ stays 1.0, trained_ is never checked) so the
    // rest of the class needs no separate Elem=float/!=float code path outside
    // train()/quantize_()/serialize()/deserialize() themselves.
    float scale_   = 1.0f;
    bool  trained_ = false;

    std::vector<Node> nodes_;  // reserved to max_elements; never reallocates

    mutable std::mutex  grow_mutex_;   // serialises append into nodes_ + rng_; read by size()
    mutable std::mutex  entry_mutex_;  // guards entry_point_ + max_layer_

    int sample_layer_() const;  // caller must hold grow_mutex_

    // float -> Elem for one vector. Identity copy when Elem=float; symmetric
    // quantization (round(v/scale_), clamped) otherwise. The one place a float
    // vector crosses into the index's stored representation — every internal
    // method after this point (search_layer, select_neighbors, closest_,
    // link_node_) operates purely on Elem, never float.
    std::vector<Elem> quantize_(const float* v) const;

    // Serial half of add(): reserve the slot, store the vector, size the neighbour
    // arrays, sample the layer. Returns (internal id, top layer).
    std::pair<InternalId, int> allocate_node_(const float* vec);

    // Parallel half of add(): navigate the graph and wire the node in, under
    // per-node locks. Reads its own vector from nodes_[id].data.
    void link_node_(InternalId id);

    std::vector<InternalId> search_layer(const Elem* q, InternalId ep, size_t ef,
                                         int layer_number) const;

    std::vector<InternalId> select_neighbors(const Elem* q,
                                             std::vector<InternalId> cands,
                                             size_t M) const;

    // Nearest of `cands` to q (cands must be non-empty).
    InternalId closest_(const Elem* q, const std::vector<InternalId>& cands) const;
};

}
