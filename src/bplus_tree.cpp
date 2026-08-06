#include "bplus_tree.h"

#include <algorithm>
#include <cassert>

namespace vdb {

BPlusTree::BPlusTree(BTreeConfig cfg) : cfg_(cfg) {
    root_ = new_leaf_();  // an empty tree is a single empty leaf, height 1
}

BPlusTree::NodeId BPlusTree::new_leaf_() {
    nodes_.emplace_back();
    nodes_.back().is_leaf = true;
    return static_cast<NodeId>(nodes_.size() - 1);
}

BPlusTree::NodeId BPlusTree::new_internal_() {
    nodes_.emplace_back();
    nodes_.back().is_leaf = false;
    return static_cast<NodeId>(nodes_.size() - 1);
}

size_t BPlusTree::child_index_(const Node& n, uint64_t key) const {
    // First separator >= key; a tie routes to *that* child, not the next one. This
    // has to agree with split_leaf_'s choice of which key to promote (the left
    // half's max, so a separator always means "<= this value is/was on the left")
    // — get the two out of sync and a query for a value that straddles a split
    // lands to the right of some of its own matches, with no way for a *forward*-
    // only leaf-chain walk to ever reach back for them.
    return static_cast<size_t>(std::lower_bound(n.keys.begin(), n.keys.end(), key) -
                               n.keys.begin());
}

BPlusTree::NodeId BPlusTree::descend_to_leaf_(uint64_t key) const {
    NodeId cur = root_;
    while (!nodes_[cur].is_leaf) cur = nodes_[cur].children[child_index_(nodes_[cur], key)];
    return cur;
}

void BPlusTree::insert(uint64_t key, InternalId id) {
    // Descend, recording the ancestors visited (root first) so a split can
    // propagate back up without parent pointers. This bottom-up shape — insert at
    // the leaf, split only the node that actually overflowed, walk back up one
    // level at a time — is also what a later B-link concurrency layer needs: each
    // split touches exactly one node plus the parent it's currently updating, never
    // an arbitrary chain of ancestor locks.
    std::vector<NodeId> path;
    NodeId cur = root_;
    while (!nodes_[cur].is_leaf) {
        path.push_back(cur);
        cur = nodes_[cur].children[child_index_(nodes_[cur], key)];
    }

    {
        Node& leaf = nodes_[cur];
        const size_t pos = static_cast<size_t>(
            std::lower_bound(leaf.keys.begin(), leaf.keys.end(), key) - leaf.keys.begin());
        leaf.keys.insert(leaf.keys.begin() + static_cast<long>(pos), key);
        leaf.values.insert(leaf.values.begin() + static_cast<long>(pos), id);
    }
    ++size_;

    if (nodes_[cur].keys.size() <= cfg_.node_capacity) return;  // fits; nothing to split

    SplitResult sr = split_leaf_(cur);
    NodeId promoted_child = cur;  // the node whose (separator, right) still needs a parent slot

    while (!path.empty()) {
        const NodeId parent = path.back();
        path.pop_back();

        Node& p = nodes_[parent];
        // Find promoted_child's own slot by *identity*, not by routing sr.separator
        // through p as a value. Those agree only when sr.separator is the first
        // matching key in p — with enough duplicate-heavy splits, some earlier
        // sibling can already have promoted the exact same value as its own
        // separator, and child_index_(p, sr.separator) would silently land on
        // *that* slot instead, splicing the new sibling in next to the wrong
        // child and corrupting both the child order and, downstream, the leaf
        // chain built on top of it.
        const size_t ci = child_position_(p, promoted_child);
        p.keys.insert(p.keys.begin() + static_cast<long>(ci), sr.separator);
        p.children.insert(p.children.begin() + static_cast<long>(ci) + 1, sr.right);

        if (p.keys.size() <= cfg_.node_capacity) return;  // absorbed; done

        sr = split_internal_(parent);
        promoted_child = parent;
    }

    // Every ancestor up to and including the root overflowed and split: the tree
    // grows by one level, with a fresh root over the two halves.
    const NodeId new_root_id = new_internal_();
    nodes_[new_root_id].keys     = {sr.separator};
    nodes_[new_root_id].children = {promoted_child, sr.right};
    root_ = new_root_id;
}

BPlusTree::SplitResult BPlusTree::split_leaf_(NodeId id) {
    const size_t n   = nodes_[id].keys.size();  // == node_capacity + 1
    const size_t mid = (n + 1) / 2;              // left keeps the (possibly) larger half

    const NodeId right_id = new_leaf_();
    // new_leaf_() may have grown nodes_ and reallocated its buffer — any reference
    // taken before this call is now dangling. Re-fetch.
    Node& l = nodes_[id];
    Node& r = nodes_[right_id];

    r.keys.assign(l.keys.begin() + static_cast<long>(mid), l.keys.end());
    r.values.assign(l.values.begin() + static_cast<long>(mid), l.values.end());
    l.keys.resize(mid);
    l.values.resize(mid);

    // Splice the new right leaf into the chain between l and whatever l used to
    // point to, so a range scan already in flight (or a fresh one starting to l's
    // left) still walks every leaf in order.
    r.next_leaf = l.next_leaf;
    l.next_leaf = right_id;

    // The separator is the *left* half's max, not the right half's min: with
    // child_index_ routing ties left, a query for exactly this value must land on
    // l first. If a run of duplicates straddles the cut (l.keys.back() ==
    // r.keys.front(), the common case that actually exercises this), using the
    // right half's min instead would send that query straight past l's matches —
    // this is the leaf-split half of the bug fixed in child_index_'s comment.
    return {right_id, l.keys.back()};  // copy: the leaf still owns this entry too
}

BPlusTree::SplitResult BPlusTree::split_internal_(NodeId id) {
    const size_t n   = nodes_[id].keys.size();  // == node_capacity + 1
    const size_t mid = n / 2;                    // middle key is promoted, kept by neither half

    const NodeId right_id = new_internal_();
    Node& l = nodes_[id];  // re-fetch, same reallocation hazard as split_leaf_
    Node& r = nodes_[right_id];

    const uint64_t separator = l.keys[mid];

    r.keys.assign(l.keys.begin() + static_cast<long>(mid) + 1, l.keys.end());
    r.children.assign(l.children.begin() + static_cast<long>(mid) + 1, l.children.end());
    l.keys.resize(mid);
    l.children.resize(mid + 1);

    return {right_id, separator};  // moved, not copied: an internal key is pure routing
}

bool BPlusTree::remove(uint64_t key, InternalId id) {
    std::vector<NodeId> path;
    NodeId cur = root_;
    while (!nodes_[cur].is_leaf) {
        path.push_back(cur);
        cur = nodes_[cur].children[child_index_(nodes_[cur], key)];
    }

    // The target (key, id) might not be in the first leaf a normal descent
    // reaches — same reason find()/range() have to walk next_leaf. Here the walk
    // also has to keep `path` valid for whichever leaf we land on, since
    // rebalance_after_erase_ needs a real ancestor chain, not just a node id.
    for (;;) {
        Node& leaf = nodes_[cur];
        const auto lo = std::lower_bound(leaf.keys.begin(), leaf.keys.end(), key);
        const auto hi = std::upper_bound(leaf.keys.begin(), leaf.keys.end(), key);

        size_t pos = 0;
        bool found = false;
        for (auto it = lo; it != hi; ++it) {
            pos = static_cast<size_t>(it - leaf.keys.begin());
            if (leaf.values[pos] == id) { found = true; break; }
        }

        if (found) {
            leaf.keys.erase(leaf.keys.begin() + static_cast<long>(pos));
            leaf.values.erase(leaf.values.begin() + static_cast<long>(pos));
            --size_;
            rebalance_after_erase_(path, cur);
            return true;
        }

        // hi short of the leaf's end means every entry equal to `key` is already
        // accounted for right here — id genuinely isn't in the tree.
        if (hi != leaf.keys.end()) return false;
        if (!advance_to_next_leaf_(path, cur)) return false;
        if (nodes_[cur].keys.empty() || nodes_[cur].keys.front() != key) return false;
    }
}

size_t BPlusTree::child_position_(const Node& parent, NodeId child) const {
    const auto it = std::find(parent.children.begin(), parent.children.end(), child);
    assert(it != parent.children.end());
    return static_cast<size_t>(it - parent.children.begin());
}

bool BPlusTree::advance_to_next_leaf_(std::vector<NodeId>& path, NodeId& cur) const {
    while (!path.empty()) {
        const NodeId parent = path.back();
        const size_t idx = child_position_(nodes_[parent], cur);
        if (idx + 1 < nodes_[parent].children.size()) {
            // parent stays on path — it's still an ancestor of where we're headed.
            NodeId node = nodes_[parent].children[idx + 1];
            while (!nodes_[node].is_leaf) {
                path.push_back(node);
                node = nodes_[node].children.front();
            }
            cur = node;
            return true;
        }
        // cur was the last child at this level: nothing to its right down here,
        // climb one level and check again.
        path.pop_back();
        cur = parent;
    }
    return false;  // cur was the rightmost leaf in the whole tree
}

void BPlusTree::rebalance_after_erase_(std::vector<NodeId>& path, NodeId node) {
    const size_t min_keys = cfg_.node_capacity / 2;
    while (true) {
        const bool is_root = path.empty();
        if (is_root || nodes_[node].keys.size() >= min_keys) {
            // The one thing still worth checking even when "no rebalance needed":
            // an internal root that just merged down to its last key removes that
            // key entirely, leaving one child and nothing left to route between —
            // the tree is one level taller than it needs to be.
            if (is_root && !nodes_[node].is_leaf && nodes_[node].keys.empty())
                root_ = nodes_[node].children.front();
            return;
        }

        const NodeId parent_id = path.back();
        path.pop_back();
        Node& parent = nodes_[parent_id];
        const size_t ci = child_position_(parent, node);

        const bool has_right = ci + 1 < parent.children.size();
        const bool has_left  = ci > 0;
        // A non-root node always has at least one sibling: its parent held at
        // least min_keys+1 children before this removal even started (the same
        // invariant we're restoring here), so has_right || has_left is guaranteed.

        if (has_right && nodes_[parent.children[ci + 1]].keys.size() > min_keys) {
            borrow_from_right_(parent, ci);
            return;
        }
        if (has_left && nodes_[parent.children[ci - 1]].keys.size() > min_keys) {
            borrow_from_left_(parent, ci);
            return;
        }
        // Neither sibling can spare an entry: merge. merge_with_right_ always
        // absorbs children[ci+1] into children[ci] — merging with the *left*
        // sibling instead is the same operation one index over (our node becomes
        // the thing being absorbed, its left sibling the survivor).
        if (has_right) merge_with_right_(parent, ci);
        else            merge_with_right_(parent, ci - 1);

        // parent lost one key and one child; it may now be underfull itself.
        // path is already parent's own ancestor chain (we popped parent_id off
        // it above), so looping with node = parent_id checks exactly the right
        // thing next.
        node = parent_id;
    }
}

void BPlusTree::borrow_from_right_(Node& parent, size_t ci) {
    Node& underflow = nodes_[parent.children[ci]];
    Node& right      = nodes_[parent.children[ci + 1]];
    if (underflow.is_leaf) {
        const uint64_t borrowed = right.keys.front();
        underflow.keys.push_back(borrowed);
        underflow.values.push_back(right.values.front());
        right.keys.erase(right.keys.begin());
        right.values.erase(right.values.begin());
        parent.keys[ci] = borrowed;  // underflow's new max
    } else {
        // Rotate through the parent: the old separator is the only thing that
        // ever said "underflow's subtree ends here" — it has to move down to
        // keep bounding underflow's newly-adopted child, and right's first key
        // (which used to bound right's first child from above) takes its place.
        underflow.keys.push_back(parent.keys[ci]);
        underflow.children.push_back(right.children.front());
        parent.keys[ci] = right.keys.front();
        right.keys.erase(right.keys.begin());
        right.children.erase(right.children.begin());
    }
}

void BPlusTree::borrow_from_left_(Node& parent, size_t ci) {
    Node& underflow = nodes_[parent.children[ci]];
    Node& left       = nodes_[parent.children[ci - 1]];
    if (underflow.is_leaf) {
        const uint64_t borrowed = left.keys.back();
        underflow.keys.insert(underflow.keys.begin(), borrowed);
        underflow.values.insert(underflow.values.begin(), left.values.back());
        left.keys.pop_back();
        left.values.pop_back();
        parent.keys[ci - 1] = left.keys.back();  // left's new max, post-loan
    } else {
        underflow.keys.insert(underflow.keys.begin(), parent.keys[ci - 1]);
        underflow.children.insert(underflow.children.begin(), left.children.back());
        parent.keys[ci - 1] = left.keys.back();
        left.keys.pop_back();
        left.children.pop_back();
    }
}

void BPlusTree::merge_with_right_(Node& parent, size_t ci) {
    Node& left  = nodes_[parent.children[ci]];
    Node& right = nodes_[parent.children[ci + 1]];
    if (left.is_leaf) {
        left.keys.insert(left.keys.end(), right.keys.begin(), right.keys.end());
        left.values.insert(left.values.end(), right.values.begin(), right.values.end());
        left.next_leaf = right.next_leaf;
    } else {
        // The separator between them is the only record of where "left's
        // children" end and "right's children" begin — pull it down so the
        // merged node still has that boundary, now as a real key instead of
        // something borrowed from the parent.
        left.keys.push_back(parent.keys[ci]);
        left.keys.insert(left.keys.end(), right.keys.begin(), right.keys.end());
        left.children.insert(left.children.end(), right.children.begin(), right.children.end());
    }
    // `right` is now unreachable — same "reclaimed at compact(), not on every
    // write" tradeoff MetadataStore already makes for tombstoned rows, rather
    // than compacting the node arena on every delete.
    parent.keys.erase(parent.keys.begin() + static_cast<long>(ci));
    parent.children.erase(parent.children.begin() + static_cast<long>(ci) + 1);
}

std::vector<InternalId> BPlusTree::find(uint64_t key) const {
    // An exact match is just a zero-width range: range()'s leaf-chain walk is what
    // makes this correct even when key's duplicates spilled across a split, which a
    // single-leaf lookup (this stage's predecessor) could not see.
    std::vector<InternalId> out;
    range(key, key, [&](uint64_t, InternalId id) { out.push_back(id); });
    return out;
}

size_t BPlusTree::height() const {
    size_t h = 1;
    NodeId cur = root_;
    while (!nodes_[cur].is_leaf) {
        assert(!nodes_[cur].children.empty());
        cur = nodes_[cur].children.front();
        ++h;
    }
    return h;
}

}
