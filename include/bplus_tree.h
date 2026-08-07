#pragma once
#include <algorithm>
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
// InternalId, for one numeric (Int64/Float64) column's secondary index. Single-
// threaded — concurrency (B-link/Lehman-Yao) is a later stage.
//
// It's a B **+** -tree, not a plain B-tree: only leaves hold values (InternalId).
// Internal nodes hold nothing but separator keys and child pointers, purely to route
// a descent. The payoff: leaves are threaded left-to-right into a linked list
// (`next_leaf`), so once a descent finds the starting point, a range scan is a cheap
// sequential walk — no re-descending the tree per key, and no visiting internal
// nodes at all after the first one. A plain B-tree, with values scattered through
// internal nodes too, can't do this: there'd be no single sorted sequence to walk.
//
// Duplicate keys are expected and cheap: many rows can share one numeric value, so
// each (key, id) pair is its own leaf entry rather than one key mapping to a list of
// ids — every leaf slot is the same fixed width (one uint64_t + one InternalId),
// which is what keeps a split, and the shift within a node, O(node_capacity) instead
// of depending on how skewed the data is. The leaf chain is also what makes this
// exact, even when a run of duplicates spills across a split: find() and range()
// both keep walking next_leaf while matches keep coming, instead of stopping at
// whichever leaf the initial descent happened to land on.
class BPlusTree {
public:
    explicit BPlusTree(BTreeConfig cfg = {});

    void insert(uint64_t key, InternalId id);

    // Removes the one entry matching both `key` and `id` exactly (duplicates mean
    // key alone doesn't identify a row — an update or a delete needs to remove a
    // *specific* row's entry, not an arbitrary one sharing its value). Returns
    // false, no change, if that exact pair isn't present. Rebalances (borrow from a
    // sibling, or merge with one) if the removal leaves a node under half full —
    // the mirror image of insert()'s split, and not a simple inverse of it: see
    // rebalance_after_erase_'s comment for why.
    bool remove(uint64_t key, InternalId id);

    // Every id stored under exactly `key`, across as many leaves as it takes.
    std::vector<InternalId> find(uint64_t key) const;

    // Every (key, id) with lo <= key <= hi, in ascending key order, via emit(key,
    // id) — a callback rather than a materialized vector so a caller that only
    // wants a count, or wants to stream straight into a bitset, doesn't pay for an
    // allocation it won't use. Same shape as VDB::collect_'s Emit parameter.
    template <class Emit>
    void range(uint64_t lo, uint64_t hi, Emit&& emit) const {
        if (lo > hi) return;
        NodeId cur = descend_to_leaf_(lo);
        while (cur != kNoNext) {
            const Node& leaf = nodes_[cur];
            auto it = std::lower_bound(leaf.keys.begin(), leaf.keys.end(), lo);
            bool exceeded_hi = false;
            for (; it != leaf.keys.end(); ++it) {
                if (*it > hi) { exceeded_hi = true; break; }
                emit(*it, leaf.values[static_cast<size_t>(it - leaf.keys.begin())]);
            }
            if (exceeded_hi) break;  // this leaf had a key past hi: the range is done
            cur = leaf.next_leaf;    // this leaf ran out without exceeding hi: keep walking
        }
    }

    size_t size() const { return size_; }

    // Number of levels from root to leaf, inclusive (1 for a tree that's still a
    // single leaf). Read-only diagnostic for tests to confirm a split actually grew
    // the tree, not part of the query path.
    size_t height() const;

private:
    using NodeId = uint32_t;
    // Sentinel for "no right sibling" (the rightmost leaf) — all-bits-set, so it
    // can never collide with a real index into nodes_.
    static constexpr NodeId kNoNext = static_cast<NodeId>(-1);

    struct Node {
        bool is_leaf = true;
        std::vector<uint64_t> keys;  // always sorted

        // Leaf only. values[i] is the InternalId for keys[i] — parallel arrays, not
        // vector<pair<uint64_t, InternalId>>: a packed {uint64_t, uint32_t} pair
        // pads to 16 bytes under alignment, two parallel arrays pack to 12.
        std::vector<InternalId> values;
        // Leaf only: this leaf's right neighbor, kNoNext for the rightmost leaf —
        // the thread that turns "find the starting leaf" into "then just walk" for
        // range()/find(), with no internal-node traversal after the first descent.
        NodeId next_leaf = kNoNext;

        // Internal only. children.size() == keys.size() + 1. children[i] roots the
        // subtree holding keys <= keys[i]; children[keys.size()] holds keys >
        // keys.back(). A key exactly equal to a separator routes *left* — that's
        // not an arbitrary tie-break, it's required: split_leaf_ sets a leaf's
        // separator to its own max key, so a query for that exact value has to
        // land on that leaf first, then range()/find()'s forward-only leaf-chain
        // walk can pick up any more matches that spilled into the next leaf. Route
        // ties right instead and a duplicate run split down the middle strands its
        // left portion permanently unreachable — no error, just silently absent.
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

    // Walks from the root down to the leaf that would hold `key` — the read-only
    // half of insert()'s descent, shared by find()/range(), neither of which needs
    // the path-back-up bookkeeping insert() does.
    NodeId descend_to_leaf_(uint64_t key) const;

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

    // --- delete / rebalance ------------------------------------------------
    // No allocation happens anywhere below (no new_leaf_/new_internal_ calls), so
    // unlike split_leaf_/split_internal_, these never need to re-fetch a Node&
    // after taking it — nodes_ never grows during a remove().

    // `child`'s index within `parent.children` (linear scan; children.size() <=
    // node_capacity + 1, so this is cheap — no parent-index is cached on the child
    // the way it would be with actual parent pointers).
    size_t child_position_(const Node& parent, NodeId child) const;

    // Moves (path, cur) to the leaf immediately after cur in left-to-right order,
    // keeping path a valid ancestor chain for the new cur. remove() needs this
    // because the leaf a normal descent reaches isn't necessarily the one holding
    // the specific id being removed — the same duplicates-spanning-a-split
    // situation find()/range() handle by walking next_leaf, except here the walk
    // also has to keep the path a later rebalance depends on in sync, which
    // next_leaf alone doesn't give you. False if cur was already the last leaf.
    bool advance_to_next_leaf_(std::vector<NodeId>& path, NodeId& cur) const;

    // After an entry has been erased from the leaf `node` (found via `path`,
    // node's ancestors root-first), restores the >= half-full invariant if `node`
    // dropped below it — borrowing one entry from a sibling that can spare it, or
    // merging with one that can't, cascading up through `path` if a merge leaves
    // an ancestor underfull too, and collapsing the root by one level if it merges
    // down to a single child.
    //
    // This is not simply "insert()'s split in reverse": a split only ever needs to
    // know about the *one* node overflowing, but fixing an underflow needs a
    // sibling — which means going through the parent, since (unlike leaves) an
    // internal node keeps no sibling pointer of its own. That's also why this
    // takes the full path rather than working the way split_leaf_/split_internal_
    // do (operate on one node, hand the caller a result to place one level up):
    // every level here needs its parent in hand to find, and rewrite, a sibling.
    void rebalance_after_erase_(std::vector<NodeId>& path, NodeId node);

    // Steals one entry from a sibling that has more than the minimum, so `node`
    // (parent.children[ci]) reaches the minimum without cutting into the sibling's
    // own floor. Only the separator in `parent` moves for a leaf-level borrow (the
    // stolen entry becomes node's new min or max); an internal-level borrow
    // rotates through `parent` — the old separator descends into node, the
    // sibling's outermost key ascends to replace it — because an internal
    // separator isn't itself owned by any child the way a leaf entry is.
    void borrow_from_right_(Node& parent, size_t ci);
    void borrow_from_left_(Node& parent, size_t ci);

    // Absorbs parent.children[ci + 1] into parent.children[ci] and removes the
    // pair (separator, right child) from parent — the only option once neither
    // sibling can spare an entry. For an internal merge, the separator between
    // them is *pulled down* into the merged node (mirroring split_internal_'s
    // promotion in reverse): it's the only thing that used to distinguish "reached
    // via the left child" from "reached via the right child", and the merged node
    // still needs that distinction internally now that it has both children.
    void merge_with_right_(Node& parent, size_t ci);
};

}
