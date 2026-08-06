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
    // First separator strictly greater than key; everything up to and including an
    // equal separator routes left-of-it, i.e. a tie goes to the *next* child.
    return static_cast<size_t>(std::upper_bound(n.keys.begin(), n.keys.end(), key) -
                               n.keys.begin());
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
        // Route the new separator through the *pre-update* parent: at this point
        // nothing in p distinguishes "left" from "right" yet, so this lands on
        // promoted_child's own slot, and the new sibling goes immediately after it.
        const size_t ci = child_index_(p, sr.separator);
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

    return {right_id, r.keys.front()};  // copy: the leaf still owns this entry too
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

std::vector<InternalId> BPlusTree::find(uint64_t key) const {
    NodeId cur = root_;
    while (!nodes_[cur].is_leaf) {
        cur = nodes_[cur].children[child_index_(nodes_[cur], key)];
    }
    const Node& leaf = nodes_[cur];
    const auto lo = std::lower_bound(leaf.keys.begin(), leaf.keys.end(), key);
    const auto hi = std::upper_bound(leaf.keys.begin(), leaf.keys.end(), key);

    std::vector<InternalId> out;
    out.reserve(static_cast<size_t>(hi - lo));
    for (auto it = lo; it != hi; ++it)
        out.push_back(leaf.values[static_cast<size_t>(it - leaf.keys.begin())]);
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
