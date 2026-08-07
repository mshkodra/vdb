#include "bplus_tree.h"

#include <algorithm>
#include <cassert>
#include <stdexcept>
#include <thread>

namespace vdb {

BPlusTree::BPlusTree(BTreeConfig cfg) : cfg_(cfg) {
    nodes_.reserve(cfg_.max_nodes);  // never reallocates past this point
    root_ = new_leaf_();             // an empty tree is a single empty leaf, height 1
}

BPlusTree::NodeId BPlusTree::new_leaf_() {
    std::lock_guard<std::mutex> gl(grow_mutex_);
    if (nodes_.size() >= cfg_.max_nodes)
        throw std::length_error("BPlusTree: max_nodes exceeded");
    nodes_.emplace_back();
    Node& n = nodes_.back();
    n.is_leaf = true;
    n.mutex   = std::make_unique<std::mutex>();
    return static_cast<NodeId>(nodes_.size() - 1);
}

BPlusTree::NodeId BPlusTree::new_internal_() {
    std::lock_guard<std::mutex> gl(grow_mutex_);
    if (nodes_.size() >= cfg_.max_nodes)
        throw std::length_error("BPlusTree: max_nodes exceeded");
    nodes_.emplace_back();
    Node& n = nodes_.back();
    n.is_leaf = false;
    n.mutex   = std::make_unique<std::mutex>();
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

std::pair<BPlusTree::NodeId, std::unique_lock<std::mutex>> BPlusTree::lock_root_() const {
    NodeId r;
    {
        std::lock_guard<std::mutex> rl(root_mutex_);
        r = root_;
    }
    // root_mutex_ is released before touching the root *node*'s own lock: once a
    // reader has a starting NodeId, everything past this point is the ordinary
    // high_key/right_link self-correction, not another root_ read.
    return {r, std::unique_lock<std::mutex>(*nodes_[r].mutex)};
}

void BPlusTree::move_right_(NodeId& cur, std::unique_lock<std::mutex>& lock, uint64_t key) const {
    while (nodes_[cur].high_key.has_value() && key > *nodes_[cur].high_key) {
        const NodeId right = nodes_[cur].right_link;
        std::unique_lock<std::mutex> right_lock(*nodes_[right].mutex);
        lock = std::move(right_lock);  // unlocks cur's old lock before cur itself changes
        cur = right;
    }
}

void BPlusTree::descend_locked_(NodeId& cur, std::unique_lock<std::mutex>& lock,
                                uint64_t key) const {
    for (;;) {
        move_right_(cur, lock, key);
        if (nodes_[cur].is_leaf) return;
        const NodeId child = nodes_[cur].children[child_index_(nodes_[cur], key)];
        std::unique_lock<std::mutex> child_lock(*nodes_[child].mutex);
        lock = std::move(child_lock);  // parent's lock releases here, before child is inspected
        cur = child;
    }
}

void BPlusTree::insert(uint64_t key, InternalId id) {
    // Descend using the same self-correcting, one-lock-at-a-time walk find()/
    // range() use — insert has no special privilege here, which is exactly what
    // lets it run concurrently with them.
    NodeId cur;
    std::unique_lock<std::mutex> lock;
    std::tie(cur, lock) = lock_root_();
    descend_locked_(cur, lock, key);

    {
        Node& leaf = nodes_[cur];
        const size_t pos = static_cast<size_t>(
            std::lower_bound(leaf.keys.begin(), leaf.keys.end(), key) - leaf.keys.begin());
        leaf.keys.insert(leaf.keys.begin() + static_cast<long>(pos), key);
        leaf.values.insert(leaf.values.begin() + static_cast<long>(pos), id);
    }
    size_.fetch_add(1, std::memory_order_relaxed);

    if (nodes_[cur].keys.size() <= cfg_.node_capacity) return;  // fits; lock releases on return

    SplitResult sr = split_leaf_(cur);
    NodeId hint = nodes_[cur].parent_hint;  // read while cur's own lock is still held
    lock.unlock();  // the split is already self-consistent; see split_leaf_'s comment
    NodeId promoted_child = cur;

    // Splice (separator, right) into promoted_child's parent, cascading upward if
    // that parent overflows too. See the class comment for why this follows
    // parent_hint (self-corrected via move_right_) rather than searching for the
    // parent by value: with duplicate-heavy data, more than one node can share
    // the exact separator value, so a value-only search has no reliable way to
    // tell which one is actually the right parent.
    for (;;) {
        NodeId parent_id;
        std::unique_lock<std::mutex> parent_lock;

        if (hint != kNoNext) {
            parent_id   = hint;
            parent_lock = std::unique_lock<std::mutex>(*nodes_[parent_id].mutex);
            move_right_(parent_id, parent_lock, sr.separator);
        } else {
            // No hint: promoted_child either is the current root, or some other
            // writer's cascade is mid-way through growing a new root over it
            // right now. Retry until one of those resolves — root growth is a
            // short critical section (allocate a node, set two fields, update
            // root_), so this settles quickly rather than spinning long.
            NodeId fresh_hint = kNoNext;
            for (;;) {
                NodeId new_root_id = kNoNext;
                {
                    std::lock_guard<std::mutex> rl(root_mutex_);
                    if (root_ == promoted_child) {
                        new_root_id = new_internal_();
                        nodes_[new_root_id].keys     = {sr.separator};
                        nodes_[new_root_id].children = {promoted_child, sr.right};
                        // new_root_id's own high_key/parent_hint stay unset: it's
                        // the new topmost node, nothing bounds or parents it.
                        root_ = new_root_id;
                    }
                }
                if (new_root_id != kNoNext) {
                    {
                        std::lock_guard<std::mutex> pcl(*nodes_[promoted_child].mutex);
                        nodes_[promoted_child].parent_hint = new_root_id;
                    }
                    {
                        std::lock_guard<std::mutex> rcl(*nodes_[sr.right].mutex);
                        nodes_[sr.right].parent_hint = new_root_id;
                    }
                    return;
                }
                {
                    std::lock_guard<std::mutex> pcl(*nodes_[promoted_child].mutex);
                    fresh_hint = nodes_[promoted_child].parent_hint;
                }
                if (fresh_hint != kNoNext) break;
                std::this_thread::yield();
            }
            hint        = fresh_hint;
            parent_id   = hint;
            parent_lock = std::unique_lock<std::mutex>(*nodes_[parent_id].mutex);
            move_right_(parent_id, parent_lock, sr.separator);
        }

        Node& p = nodes_[parent_id];
        // Find promoted_child's own slot by *identity*, not by routing
        // sr.separator through p as a value — see the class comment: those agree
        // only when sr.separator happens to be the first matching key in p, and
        // duplicate-heavy data routinely breaks that assumption.
        const size_t ci = child_position_(p, promoted_child);
        p.keys.insert(p.keys.begin() + static_cast<long>(ci), sr.separator);
        p.children.insert(p.children.begin() + static_cast<long>(ci) + 1, sr.right);
        const NodeId spliced_child = sr.right;

        // Record spliced_child's parent as parent_id *now*, still under parent_id's
        // own lock — not after releasing it. parent_id's lock is what excludes any
        // other thread from splitting parent_id concurrently; recording the hint
        // only after unlocking leaves a window where such a split (moving
        // spliced_child to its new right sibling, and correctly updating its hint
        // to say so) could complete first, and this write would then clobber that
        // correct value with a now-stale one. Sequencing this write before the
        // possible split below, in the same thread under the same lock, means
        // split_internal_'s own hint update (if it moves spliced_child) always
        // comes after and wins — never the other way around.
        {
            std::lock_guard<std::mutex> cl(*nodes_[spliced_child].mutex);
            nodes_[spliced_child].parent_hint = parent_id;
        }

        if (p.keys.size() <= cfg_.node_capacity) {  // absorbed; done
            parent_lock.unlock();
            return;
        }

        SplitResult new_sr = split_internal_(parent_id);  // may re-home spliced_child
                                                            // into new_sr.right, correctly
                                                            // overwriting the hint set above
        NodeId next_hint    = nodes_[parent_id].parent_hint;  // read while still locked
        parent_lock.unlock();

        sr             = new_sr;
        hint           = next_hint;
        promoted_child = parent_id;
    }
}

BPlusTree::SplitResult BPlusTree::split_leaf_(NodeId id) {
    // No reallocation hazard here (unlike before concurrency landed): nodes_ is
    // reserved to max_nodes and never grows past it, so a reference taken before
    // new_leaf_() stays valid after it — no re-fetch needed.
    Node& l = nodes_[id];
    const size_t n   = l.keys.size();  // == node_capacity + 1
    const size_t mid = (n + 1) / 2;     // left keeps the (possibly) larger half

    const NodeId right_id = new_leaf_();
    Node& r = nodes_[right_id];

    r.keys.assign(l.keys.begin() + static_cast<long>(mid), l.keys.end());
    r.values.assign(l.values.begin() + static_cast<long>(mid), l.values.end());
    l.keys.resize(mid);
    l.values.resize(mid);

    // r inherits l's old right_link/high_key: it now covers everything l used to
    // cover above the split point. l's own high_key becomes its new max and its
    // right_link now points at r — set here, under l's lock, so the split is
    // fully self-consistent and readable via the B-link protocol the instant the
    // caller releases that lock, even before any parent has heard about r at all.
    r.right_link = l.right_link;
    r.high_key   = l.high_key;
    l.right_link = right_id;
    l.high_key   = l.keys.back();

    return {right_id, l.keys.back()};  // copy: the leaf still owns this entry too
}

BPlusTree::SplitResult BPlusTree::split_internal_(NodeId id) {
    Node& l = nodes_[id];
    const size_t n   = l.keys.size();  // == node_capacity + 1
    const size_t mid = n / 2;           // middle key is promoted, kept by neither half

    const NodeId right_id = new_internal_();
    Node& r = nodes_[right_id];

    const uint64_t separator = l.keys[mid];

    r.keys.assign(l.keys.begin() + static_cast<long>(mid) + 1, l.keys.end());
    r.children.assign(l.children.begin() + static_cast<long>(mid) + 1, l.children.end());
    l.keys.resize(mid);
    l.children.resize(mid + 1);

    r.right_link = l.right_link;
    r.high_key   = l.high_key;
    l.right_link = right_id;
    l.high_key   = separator;

    // Every child that moved to r is now r's, not l's — their parent_hint has to
    // follow, or a future insert() cascading up through one of them would go
    // looking for it under l's old lock instead. Each write takes the *child's*
    // own lock, not l's: parent_hint is guarded by its owning node's mutex (see
    // the class comment), and a thread can already be holding that lock — e.g. a
    // concurrent insert() into c, reached before this split, that's about to read
    // c.parent_hint under c's own lock (insert()'s `NodeId hint = ...` line) —
    // entirely independently of l. l's own lock only excludes other threads that
    // are trying to reach c *through* l; it says nothing about a thread already
    // past that point. Locking each child here, while l is held, is still a
    // strictly top-down (parent-then-child) acquisition — the one nesting this
    // file's single-lock-at-a-time policy allows, same as move_right_'s brief
    // sibling-to-sibling overlap.
    for (NodeId c : r.children) {
        std::lock_guard<std::mutex> cl(*nodes_[c].mutex);
        nodes_[c].parent_hint = right_id;
    }

    return {right_id, separator};  // moved, not copied: an internal key is pure routing
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

// --- delete / rebalance (single-threaded only — see the class comment) -------

bool BPlusTree::remove(uint64_t key, InternalId id) {
    std::vector<NodeId> path;
    NodeId cur = root_;
    while (!nodes_[cur].is_leaf) {
        path.push_back(cur);
        cur = nodes_[cur].children[child_index_(nodes_[cur], key)];
    }

    // The target (key, id) might not be in the first leaf a normal descent
    // reaches — same reason find()/range() have to walk right_link. Here the walk
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
            size_.fetch_sub(1, std::memory_order_relaxed);
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
            if (is_root && !nodes_[node].is_leaf && nodes_[node].keys.empty()) {
                const NodeId new_root = nodes_[node].children.front();
                // The promoted child's high_key/right_link/parent_hint were valid
                // for its old life as a bounded, parented node — a root is neither.
                // Merging is supposed to have already propagated an unbounded
                // high_key/right_link this far left (inheriting transitively from
                // whichever child was truly rightmost), but insert() and future
                // grow/shrink cycles depend on the root's fields being exactly
                // right, not "probably right via inheritance" — so clear them
                // explicitly rather than trust that chain held in every case.
                nodes_[new_root].high_key    = std::nullopt;
                nodes_[new_root].right_link  = kNoNext;
                nodes_[new_root].parent_hint = kNoNext;
                root_ = new_root;
            }
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
        parent.keys[ci]  = borrowed;  // underflow's new max
        underflow.high_key = borrowed;  // keep the node's own bound in sync with
                                         // the parent's — what the B-link walk reads
    } else {
        // Rotate through the parent: the old separator is the only thing that
        // ever said "underflow's subtree ends here" — it has to move down to
        // keep bounding underflow's newly-adopted child, and right's first key
        // (which used to bound right's first child from above) takes its place.
        const uint64_t new_sep = right.keys.front();
        const NodeId   moved   = right.children.front();
        underflow.keys.push_back(parent.keys[ci]);
        underflow.children.push_back(moved);
        parent.keys[ci]    = new_sep;
        underflow.high_key = new_sep;
        right.keys.erase(right.keys.begin());
        right.children.erase(right.children.begin());
        // moved now belongs to underflow, not right — insert() reads this field
        // to find its parent, so it has to follow the move (single-threaded here,
        // so a plain write is enough; no reader can observe the in-between state).
        nodes_[moved].parent_hint = parent.children[ci];
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
        left.high_key       = left.keys.back();
    } else {
        const NodeId moved = left.children.back();
        underflow.keys.insert(underflow.keys.begin(), parent.keys[ci - 1]);
        underflow.children.insert(underflow.children.begin(), moved);
        parent.keys[ci - 1] = left.keys.back();
        left.high_key       = left.keys.back();
        left.keys.pop_back();
        left.children.pop_back();
        // moved now belongs to underflow, not left — see borrow_from_right_'s
        // identical note.
        nodes_[moved].parent_hint = parent.children[ci];
    }
}

void BPlusTree::merge_with_right_(Node& parent, size_t ci) {
    const NodeId left_id = parent.children[ci];
    Node& left  = nodes_[left_id];
    Node& right = nodes_[parent.children[ci + 1]];
    if (left.is_leaf) {
        left.keys.insert(left.keys.end(), right.keys.begin(), right.keys.end());
        left.values.insert(left.values.end(), right.values.begin(), right.values.end());
    } else {
        // The separator between them is the only record of where "left's
        // children" end and "right's children" begin — pull it down so the
        // merged node still has that boundary, now as a real key instead of
        // something borrowed from the parent.
        left.keys.push_back(parent.keys[ci]);
        left.keys.insert(left.keys.end(), right.keys.begin(), right.keys.end());
        left.children.insert(left.children.end(), right.children.begin(), right.children.end());
        // Every one of right's former children now belongs to left — same
        // parent_hint follow-up as the borrow_ functions, just for the whole set
        // right had rather than one.
        for (NodeId c : right.children) nodes_[c].parent_hint = left_id;
    }
    // left absorbs right's reach entirely: right's old high_key/right_link become
    // left's, same as split_leaf_/split_internal_ propagate them the other way.
    left.right_link = right.right_link;
    left.high_key    = right.high_key;
    // `right` is now unreachable — same "reclaimed at compact(), not on every
    // write" tradeoff MetadataStore already makes for tombstoned rows, rather
    // than compacting the node arena on every delete.
    parent.keys.erase(parent.keys.begin() + static_cast<long>(ci));
    parent.children.erase(parent.children.begin() + static_cast<long>(ci) + 1);
}

}
