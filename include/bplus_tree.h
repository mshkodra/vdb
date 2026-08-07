#pragma once
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <utility>
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

    // Node arena is reserved to this once, at construction, and never grows past
    // it — insert() throws std::length_error rather than exceed it. Same tradeoff
    // HNSWConfig::max_elements already makes in this codebase, and for the same
    // reason: a concurrent reader can hold a reference into the arena (in this
    // case, a locked node) only if the arena never reallocates out from under it.
    size_t max_nodes = 1u << 20;
};

// A B+-tree keyed on a pre-transformed sortable uint64_t (see sortable_bits.h) ->
// InternalId, for one numeric (Int64/Float64) column's secondary index.
//
// It's a B **+** -tree, not a plain B-tree: only leaves hold values (InternalId).
// Internal nodes hold nothing but separator keys and child pointers, purely to route
// a descent. The payoff: every node (leaf or internal) is threaded to its right
// neighbor at the same level (`right_link`), so once a descent finds a starting
// point, walking forward is a cheap sequential hop — no re-descending the tree per
// key, and (past the first node) no lock ever held on more than one node at once.
//
// Duplicate keys are expected and cheap: many rows can share one numeric value, so
// each (key, id) pair is its own leaf entry rather than one key mapping to a list of
// ids — every leaf slot is the same fixed width (one uint64_t + one InternalId),
// which is what keeps a split, and the shift within a node, O(node_capacity) instead
// of depending on how skewed the data is. The leaf chain is also what makes this
// exact, even when a run of duplicates spills across a split: find() and range()
// both keep walking right_link while matches keep coming, instead of stopping at
// whichever leaf the initial descent happened to land on.
//
// Thread-safety: insert(), find(), and range() are safe to call concurrently with
// each other, following Lehman & Yao's B-link protocol — the "hardest, most novel
// part" of this whole structure, so it's worth spelling out precisely:
//
//   - Every node carries its own high_key (the largest value its subtree can
//     legitimately hold; nullopt means "unbounded", true only for whichever node is
//     currently rightmost at its level) and right_link (its right sibling, kNoNext
//     if none). A descent that lands on a node checks the target against high_key
//     first, *before* looking at this node's own children/values: if the target
//     exceeds it, this node has been split by a concurrent writer and part of what
//     it used to cover now lives in right_link — so move there and re-check, rather
//     than trusting a possibly-stale idea of "this subtree covers my target".
//   - A writer's split is two separate, independently-safe steps. Step one: under
//     the splitting node's own lock, shrink it, set its high_key to the new
//     separator, and set its right_link to the new sibling — then release that
//     lock. Step two, entirely separate: splice (separator, new sibling) into the
//     parent, possibly cascading the same two-step process up a level. Between
//     steps one and two, the new sibling is reachable *only* via right_link, not
//     yet via any parent — and that's fine, because it's exactly what a concurrent
//     reader's high_key check is for.
//   - Finding "the parent" for step two is never a fresh search by value from the
//     root: with duplicate-heavy data, more than one node can share the exact same
//     separator, so a value-only search has no reliable way to tell which one is
//     actually the right parent, and can walk into a completely unrelated part of
//     the tree (this corrupted the tree during testing before insert() was
//     rewritten to avoid it). Instead, every node carries a parent_hint, set once
//     when it's first spliced into a parent and never updated after. Finding a
//     node's current parent means following that hint and correcting it for
//     staleness with the same high_key check move_right_ always uses: did *this
//     one, already-identified* node split since the hint was set, and if so, which
//     side did our node land on. That's unambiguous (unlike "where in the whole
//     tree does this value belong") because it's about one specific node's own
//     split, never a search among possibly-many candidates sharing a value. A node
//     with no hint yet (kNoNext) either is the current root, or is about to be
//     given one by whichever writer is mid-way through growing a new root over
//     it — insert() distinguishes the two by checking root_ itself.
//   - Every node has its own mutex, and at most one is ever held at a time: a
//     descent locks a child and releases the parent before looking at the child, a
//     rightward hop locks the neighbor and releases the current node. No ancestor
//     lock is ever held while waiting on a descendant's — the actual mechanical
//     payoff of the whole protocol, and what makes it deadlock-free without a
//     lock-ordering rule. (Matches HNSWIndex's own "at most one node lock held at a
//     time" policy, just with a self-correcting traversal instead of an
//     approximate one.)
//
// remove() is *not* part of this: concurrent delete for a B-link tree is a
// materially harder, far-less-established problem than concurrent insert (Lehman &
// Yao's original paper is an insert protocol), so it's out of scope here. Calling
// remove() concurrently with insert()/find()/range() is undefined — the same
// "quiesce first" caveat VDB::compact() already documents for the vector index.
class BPlusTree {
public:
    explicit BPlusTree(BTreeConfig cfg = {});

    void insert(uint64_t key, InternalId id);

    // Not safe to call concurrently with insert()/find()/range() — see the class
    // comment. Removes the one entry matching both `key` and `id` exactly
    // (duplicates mean key alone doesn't identify a row — an update or a delete
    // needs to remove a *specific* row's entry, not an arbitrary one sharing its
    // value). Returns false, no change, if that exact pair isn't present.
    // Rebalances (borrow from a sibling, or merge with one) if the removal leaves a
    // node under half full — the mirror image of insert()'s split, and not a
    // simple inverse of it: see rebalance_after_erase_'s comment for why.
    bool remove(uint64_t key, InternalId id);

    // Every id stored under exactly `key`, across as many leaves as it takes.
    std::vector<InternalId> find(uint64_t key) const;

    // Every (key, id) with lo <= key <= hi, in ascending key order, via emit(key,
    // id) — a callback rather than a materialized vector so a caller that only
    // wants a count, or wants to stream straight into a bitset, doesn't pay for an
    // allocation it won't use. Same shape as VDB::collect_'s Emit parameter.
    //
    // Each leaf's matching entries are copied out *before* emit() is called for any
    // of them, so no node lock is ever held while running caller code — the same
    // "copy under the lock, then work the copy unlocked" policy HNSWIndex's readers
    // already use, and for the same reason: holding a lock across arbitrary
    // callback work would turn one slow caller into contention for every writer.
    template <class Emit>
    void range(uint64_t lo, uint64_t hi, Emit&& emit) const {
        if (lo > hi) return;
        NodeId cur;
        std::unique_lock<std::mutex> lock;
        std::tie(cur, lock) = lock_root_();
        move_right_(cur, lock, lo);
        descend_locked_(cur, lock, lo);  // now at the leaf that would hold lo

        std::vector<std::pair<uint64_t, InternalId>> batch;
        for (;;) {
            move_right_(cur, lock, lo);
            const Node& leaf = nodes_[cur];
            auto it = std::lower_bound(leaf.keys.begin(), leaf.keys.end(), lo);
            bool exceeded_hi = false;
            batch.clear();
            for (; it != leaf.keys.end(); ++it) {
                if (*it > hi) { exceeded_hi = true; break; }
                batch.emplace_back(*it, leaf.values[static_cast<size_t>(it - leaf.keys.begin())]);
            }
            const NodeId next = leaf.right_link;
            lock.unlock();
            for (auto& [k, id] : batch) emit(k, id);
            if (exceeded_hi || next == kNoNext) return;
            cur = next;
            lock = std::unique_lock<std::mutex>(*nodes_[cur].mutex);
        }
    }

    size_t size() const { return size_.load(std::memory_order_relaxed); }

    // Number of levels from root to leaf, inclusive (1 for a tree that's still a
    // single leaf). Read-only diagnostic for tests to confirm a split actually grew
    // the tree, not part of the query path — and, unlike insert/find/range, *not*
    // safe to call while a writer might be concurrently touching the tree: it walks
    // children[0] node-to-node without taking any lock, meant for quiescent
    // inspection after threads have joined, the way tests use it.
    size_t height() const;

private:
    using NodeId = uint32_t;
    // Sentinel for "no right sibling" (whichever node is rightmost at its level) —
    // all-bits-set, so it can never collide with a real index into nodes_.
    static constexpr NodeId kNoNext = static_cast<NodeId>(-1);

    struct Node {
        bool is_leaf = true;
        std::vector<uint64_t> keys;  // always sorted

        // Leaf only. values[i] is the InternalId for keys[i] — parallel arrays, not
        // vector<pair<uint64_t, InternalId>>: a packed {uint64_t, uint32_t} pair
        // pads to 16 bytes under alignment, two parallel arrays pack to 12.
        std::vector<InternalId> values;

        // Internal only. children.size() == keys.size() + 1. children[i] roots the
        // subtree holding keys <= keys[i]; children[keys.size()] holds keys >
        // keys.back(). A key exactly equal to a separator routes *left* — that's
        // not an arbitrary tie-break, it's required: split_leaf_ sets a node's
        // separator to its own max key, so a query for that exact value has to
        // land on that node first, then range()/find()'s forward-only right_link
        // walk can pick up any more matches that spilled into the next node. Route
        // ties right instead and a duplicate run split down the middle strands its
        // left portion permanently unreachable — no error, just silently absent.
        std::vector<NodeId> children;

        // Both leaf and internal: this node's right sibling at the same level
        // (kNoNext if it's currently the rightmost), and the largest key its
        // subtree can legitimately hold (nullopt if unbounded, true only for the
        // rightmost node at this level). Together these are the B-link protocol's
        // entire self-correction mechanism — see the class comment.
        NodeId right_link = kNoNext;
        std::optional<uint64_t> high_key;

        // Both leaf and internal: a *hint* at this node's current parent, kNoNext
        // if it has none (true only for the actual root). Set exactly once, the
        // moment this node is first spliced into some parent — never updated
        // after, so it can go stale the same way a remembered ancestor can (that
        // parent later splits) but never in a way move_right_ can't fix, for the
        // same reason a remembered ancestor is safe to correct that way: this is
        // always "did *this one, already-identified* node move", never "which of
        // possibly-several nodes with the same value is the right one" — the
        // question a value-only search can't answer under duplicate-heavy data.
        // Guarded by this node's own mutex, like every other field here.
        NodeId parent_hint = kNoNext;

        // unique_ptr, not a plain std::mutex member, purely so Node stays movable
        // enough for the vector machinery even though it's never actually moved at
        // runtime (nodes_ is reserved once and never reallocates) — same reason
        // HNSWIndex's own per-node lock is a unique_ptr<mutex>, not embedded
        // directly. mutable so find()/range() can lock through a const BPlusTree&.
        mutable std::unique_ptr<std::mutex> mutex;
    };

    // Node pool: every node reference is an index into `nodes_`, never a raw
    // pointer, and `nodes_` is reserved to cfg_.max_nodes once at construction and
    // never reallocates past it — so a reference into an element, once obtained
    // under that element's own lock, stays valid for as long as the lock is held,
    // even while other threads are appending new nodes elsewhere in the arena.
    // Flat, not vector<unique_ptr<Node>>: the reserved capacity is what gives
    // address stability, so there's no need to pay for a second heap indirection
    // per node on top of it (matches HNSWIndex::nodes_ exactly).
    std::vector<Node> nodes_;
    NodeId              root_;
    std::atomic<size_t> size_{0};
    BTreeConfig         cfg_;

    // Guards root_ itself — the one piece of state with no node of its own to be
    // locked through. Held only for the single read/write of root_, never across a
    // descent: once a search has its starting NodeId, all further self-correction
    // (including past a root that's since been demoted by a concurrent split) goes
    // through the ordinary high_key/right_link mechanism, not through root_ again.
    mutable std::mutex root_mutex_;
    // Serialises new_leaf_/new_internal_'s append into nodes_ — concurrent splits
    // in unrelated parts of the tree would otherwise race on the arena's own
    // bookkeeping. Mirrors HNSWIndex's grow_mutex_ exactly, same reason.
    std::mutex grow_mutex_;

    NodeId new_leaf_();
    NodeId new_internal_();

    // Which child of internal node `n` a descent for `key` takes.
    size_t child_index_(const Node& n, uint64_t key) const;

    // Reads root_ under root_mutex_ and locks it, returning both. The only place
    // root_ itself is ever touched outside of insert()'s root-growth step.
    std::pair<NodeId, std::unique_lock<std::mutex>> lock_root_() const;

    // While `cur`'s high_key says `key` belongs further right, hops to right_link,
    // locking the neighbor before releasing cur's own lock (so at most one lock is
    // held at the instant of the hop, never held across it) — the self-correction
    // step that makes a stale idea of "which node covers key" safe to act on.
    void move_right_(NodeId& cur, std::unique_lock<std::mutex>& lock, uint64_t key) const;

    // Descends from an already-locked, already-move_right_'d `cur` to the leaf that
    // would hold `key`, locking one child at a time and releasing each parent
    // before looking at the child (never two locks held at once), move_right_-ing
    // at every level along the way.
    void descend_locked_(NodeId& cur, std::unique_lock<std::mutex>& lock, uint64_t key) const;

    struct SplitResult {
        NodeId   right;      // the new right sibling
        uint64_t separator;  // key to insert into the parent, routing to `right`
    };
    // Splits an overfull leaf: the new right leaf keeps its own copy of its first
    // key as the separator (a leaf must retain every entry it holds). Sets both
    // halves' right_link/high_key so the split is internally consistent (readable
    // via the B-link protocol) the instant the caller releases `id`'s lock, even
    // before any parent has been updated.
    SplitResult split_leaf_(NodeId id);
    // Splits an overfull internal node: the middle key is promoted to the parent
    // and removed from both halves — an internal separator exists only to route,
    // so once it's promoted it has no reason to also stay behind.
    SplitResult split_internal_(NodeId id);

    // --- delete / rebalance ------------------------------------------------
    // Single-threaded only (see the class comment) — no allocation happens
    // anywhere below (no new_leaf_/new_internal_ calls), so unlike
    // split_leaf_/split_internal_, these never need to worry about another thread
    // touching the arena concurrently.

    // `child`'s index within `parent.children` (linear scan; children.size() <=
    // node_capacity + 1, so this is cheap — no parent-index is cached on the child
    // the way it would be with actual parent pointers).
    size_t child_position_(const Node& parent, NodeId child) const;

    // Moves (path, cur) to the leaf immediately after cur in left-to-right order,
    // keeping path a valid ancestor chain for the new cur. remove() needs this
    // because the leaf a normal descent reaches isn't necessarily the one holding
    // the specific id being removed — the same duplicates-spanning-a-split
    // situation find()/range() handle by walking right_link, except here the walk
    // also has to keep the path a later rebalance depends on in sync, which
    // right_link alone doesn't give you. False if cur was already the last leaf.
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
