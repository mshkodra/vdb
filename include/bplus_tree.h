#pragma once
#include <cstdint>
#include <vector>

#include "vdb_types.h"

namespace vdb {

struct BTreeConfig {
    // Max entries per node before it splits. Wide enough that a node spans several
    // cache lines — each hop down the tree is a likely cache/TLB miss, so fewer,
    // wider hops beat many narrow ones — narrow enough that an in-node binary search
    // and a split's O(node_capacity) shift stay cheap. A runtime field rather than a
    // compile-time constant, so it's benchmarkable later without a recompile (same
    // reasoning as HNSWConfig::Mmax0).
    size_t node_capacity = 128;
};

// A B+-tree keyed on a pre-transformed sortable uint64_t (see sortable_bits.h) ->
// InternalId, for one numeric (Int64/Float64) column's secondary index. This PR is
// point lookup and insert-with-splitting only: single-threaded (concurrency is a
// later stage, B-link/Lehman-Yao) and insert-only (delete/merge-on-underflow is a
// later stage too).
//
// It's a B **+** -tree, not a plain B-tree: only leaves hold values (InternalId).
// Internal nodes hold nothing but separator keys and child pointers, purely to route
// a descent — the payoff (not yet built here) is that leaves can later be threaded
// into a linked list for a cheap sequential range scan, which a plain B-tree's
// values-everywhere layout can't do.
//
// Duplicate keys are expected and cheap: many rows can share one numeric value, so
// each (key, id) pair is its own leaf entry rather than one key mapping to a list of
// ids — every leaf slot is the same fixed width (one uint64_t + one InternalId),
// which is what keeps a split, and the shift within a node, O(node_capacity) instead
// of depending on how skewed the data is.
//
// find() only searches the single leaf the key's descent lands on. If enough rows
// share a key that they spilled across a split into the next leaf, find() alone
// won't see the ones that landed on the other side — closing that gap is exactly
// what leaf chaining (next_leaf, a later stage) is for.
class BPlusTree {
public:
    explicit BPlusTree(BTreeConfig cfg = {});

    void insert(uint64_t key, InternalId id);

    // Every id stored under exactly `key` that lives in the one leaf the descent
    // reaches — see the class comment's caveat about duplicates spanning a split.
    std::vector<InternalId> find(uint64_t key) const;

    size_t size() const { return size_; }

    // Number of levels from root to leaf, inclusive (1 for a tree that's still a
    // single leaf). Read-only diagnostic for tests to confirm a split actually grew
    // the tree, not part of the query path.
    size_t height() const;

private:
    using NodeId = uint32_t;

    struct Node {
        bool is_leaf = true;
        std::vector<uint64_t> keys;  // always sorted

        // Leaf only. values[i] is the InternalId for keys[i] — parallel arrays, not
        // vector<pair<uint64_t, InternalId>>: a packed {uint64_t, uint32_t} pair
        // pads to 16 bytes under alignment, two parallel arrays pack to 12.
        std::vector<InternalId> values;

        // Internal only. children.size() == keys.size() + 1. children[i] roots the
        // subtree holding keys < keys[i]; children[keys.size()] holds keys >=
        // keys.back(). (A key exactly equal to a separator always routes right —
        // this is what makes the routing rule a single upper_bound, with no special
        // case for ties, even though ties are common with duplicate-heavy columns.)
        std::vector<NodeId> children;
    };

    // Node pool: every node reference is an index into `nodes_`, never a raw
    // pointer. A raw `Node*`/`Node&` taken before a split's new-node allocation
    // would dangle the moment that push_back reallocates the vector; an index
    // stays valid regardless of where the buffer moves. split_leaf_/split_internal_
    // re-fetch their references after allocating the sibling for exactly this
    // reason — see the .cpp.
    std::vector<Node> nodes_;
    NodeId             root_;
    size_t             size_ = 0;
    BTreeConfig        cfg_;

    NodeId new_leaf_();
    NodeId new_internal_();

    // Which child of internal node `n` a descent for `key` takes.
    size_t child_index_(const Node& n, uint64_t key) const;

    struct SplitResult {
        NodeId   right;      // the new right sibling
        uint64_t separator;  // key to insert into the parent, routing to `right`
    };
    // Splits an overfull leaf: the new right leaf keeps its own copy of its first
    // key as the separator (a leaf must retain every entry it holds).
    SplitResult split_leaf_(NodeId id);
    // Splits an overfull internal node: the middle key is promoted to the parent
    // and removed from both halves — an internal separator exists only to route,
    // so once it's promoted it has no reason to also stay behind.
    SplitResult split_internal_(NodeId id);
};

}
