#include "test.h"

#include "bplus_tree.h"
#include "sortable_bits.h"

#include <algorithm>
#include <map>
#include <random>
#include <set>
#include <vector>

using namespace vdb;

TEST(bplus_tree_starts_empty) {
    BPlusTree t;
    EXPECT(t.size() == 0);
    EXPECT(t.height() == 1);
    EXPECT(t.find(0).empty());
    EXPECT(t.find(0xFFFFFFFFFFFFFFFFULL).empty());
}

TEST(bplus_tree_single_insert_is_found) {
    BPlusTree t;
    t.insert(42, 7);
    EXPECT(t.size() == 1);
    EXPECT(t.find(42) == std::vector<InternalId>{7});
    EXPECT(t.find(41).empty());
    EXPECT(t.find(43).empty());
}

TEST(bplus_tree_unique_keys_survive_many_splits_and_remain_findable) {
    // No duplicates at all, so there's no ambiguity about which leaf a key's entry
    // ended up in after a split — this is the airtight test of the actual splitting
    // and routing logic (leaf split's copy-up, internal split's move-up, and new
    // roots on root overflow), independent of the documented duplicate-spanning
    // caveat.
    BTreeConfig cfg;
    cfg.node_capacity = 8;  // small on purpose: force many splits at modest N
    BPlusTree t(cfg);

    std::mt19937_64 rng(1);
    std::set<uint64_t> seen;
    std::uniform_int_distribution<uint64_t> dist;
    while (seen.size() < 5000) seen.insert(dist(rng));

    std::vector<uint64_t> keys(seen.begin(), seen.end());  // sorted (from the set)
    std::vector<uint64_t> insert_order = keys;
    std::shuffle(insert_order.begin(), insert_order.end(), rng);

    // id i is defined as "the insert-order index of keys[i]" via this map, so
    // lookups below can check both directions without conflating the two orders.
    std::vector<InternalId> id_of_key(keys.size());
    for (size_t i = 0; i < insert_order.size(); ++i) {
        t.insert(insert_order[i], static_cast<InternalId>(i));
        const size_t ki = static_cast<size_t>(
            std::lower_bound(keys.begin(), keys.end(), insert_order[i]) - keys.begin());
        id_of_key[ki] = static_cast<InternalId>(i);
    }

    EXPECT(t.size() == keys.size());
    ASSERT(t.height() > 1);  // confirms splitting (and likely root growth) happened

    for (size_t i = 0; i < keys.size(); ++i) {
        const auto found = t.find(keys[i]);
        ASSERT(found.size() == 1);
        EXPECT(found[0] == id_of_key[i]);
    }

    // A key that was never inserted must never be found. Probe values adjacent (in
    // sorted order) to a real key first, skipping over any that collide with an
    // actual key by chance — exactly where an off-by-one in child_index_'s
    // upper_bound would show up.
    int checked = 0;
    for (uint64_t probe = 1; probe < 100 && checked < 10; ++probe) {
        const uint64_t candidate = keys[0] ^ probe;
        if (std::binary_search(keys.begin(), keys.end(), candidate)) continue;
        EXPECT(t.find(candidate).empty());
        ++checked;
    }
}

TEST(bplus_tree_duplicate_keys_within_one_leaf_are_all_found) {
    BTreeConfig cfg;
    cfg.node_capacity = 64;  // generous: keep everything in the root leaf
    BPlusTree t(cfg);

    // A handful of distinct values, each shared by several rows — the common case
    // for a real column (e.g. a `category` price bucket), well under node_capacity.
    for (InternalId id = 0; id < 5; ++id) t.insert(/*key=*/100, id);
    for (InternalId id = 5; id < 9; ++id) t.insert(/*key=*/200, id);
    t.insert(/*key=*/300, 9);

    ASSERT(t.height() == 1);  // sanity: this test's premise is "no split happened"
    EXPECT(t.size() == 10);

    auto at_100 = t.find(100);
    std::sort(at_100.begin(), at_100.end());
    EXPECT(at_100 == std::vector<InternalId>({0, 1, 2, 3, 4}));

    auto at_200 = t.find(200);
    std::sort(at_200.begin(), at_200.end());
    EXPECT(at_200 == std::vector<InternalId>({5, 6, 7, 8}));

    EXPECT(t.find(300) == std::vector<InternalId>{9});
    EXPECT(t.find(150).empty());
}

TEST(bplus_tree_find_sees_duplicates_that_spilled_across_a_split) {
    // PR5's version of this test pinned down a documented gap: find() only checked
    // the one leaf a key's descent landed on. Leaf chaining is supposed to close
    // that — this proves it actually does, for the exact scenario that used to slip
    // through: enough rows sharing one key that a split cuts the run in half.
    BTreeConfig cfg;
    cfg.node_capacity = 4;
    BPlusTree t(cfg);
    for (InternalId id = 0; id < 5; ++id) t.insert(/*key=*/42, id);

    ASSERT(t.height() > 1);  // confirms the split actually happened
    auto found = t.find(42);
    std::sort(found.begin(), found.end());
    EXPECT(found == std::vector<InternalId>({0, 1, 2, 3, 4}));  // all 5, not just one leaf's share
}

TEST(bplus_tree_range_scan_sees_duplicates_that_spilled_across_a_split) {
    // Same scenario as above, through range() instead of find() — the two share
    // find()'s implementation (find(k) is range(k, k, ...)), but this exercises the
    // hi-bound / exceeded_hi bookkeeping directly rather than through that alias.
    BTreeConfig cfg;
    cfg.node_capacity = 4;
    BPlusTree t(cfg);
    for (InternalId id = 0; id < 5; ++id) t.insert(/*key=*/42, id);
    ASSERT(t.height() > 1);

    std::vector<InternalId> found;
    t.range(42, 42, [&](uint64_t k, InternalId id) {
        EXPECT(k == 42);
        found.push_back(id);
    });
    std::sort(found.begin(), found.end());
    EXPECT(found == std::vector<InternalId>({0, 1, 2, 3, 4}));
}

TEST(bplus_tree_random_mixed_workload_matches_a_multimap_oracle) {
    // Now that duplicates spanning a split are handled correctly (see the two
    // tests above), this checks full completeness, not just soundness: every id
    // find() returns really was inserted under that key, *and* every id that was
    // really inserted under that key comes back. Realistic duplicate density (many
    // rows, few thousand distinct values) with a small capacity, so both ordinary
    // and duplicate-spanning splits are the common case here, not an edge case.
    BTreeConfig cfg;
    cfg.node_capacity = 16;
    BPlusTree t(cfg);

    std::mt19937_64 rng(2);
    std::uniform_int_distribution<uint64_t> key_dist(0, 2000);
    std::map<uint64_t, std::vector<InternalId>> oracle;

    constexpr int kN = 20000;
    for (InternalId id = 0; id < kN; ++id) {
        const uint64_t key = key_dist(rng);
        t.insert(key, id);
        oracle[key].push_back(id);
    }
    EXPECT(t.size() == kN);

    for (uint64_t key = 0; key <= 2000; ++key) {
        auto found = t.find(key);
        std::sort(found.begin(), found.end());
        auto want_it = oracle.find(key);
        std::vector<InternalId> want = want_it == oracle.end() ? std::vector<InternalId>{}
                                                                : want_it->second;
        std::sort(want.begin(), want.end());
        ASSERT(found == want);
    }
}

TEST(bplus_tree_range_scan_matches_an_oracle_over_many_leaves) {
    BTreeConfig cfg;
    cfg.node_capacity = 16;
    BPlusTree t(cfg);

    std::mt19937_64 rng(3);
    std::uniform_int_distribution<uint64_t> key_dist(0, 5000);
    std::map<uint64_t, std::vector<InternalId>> oracle;

    constexpr int kN = 20000;
    for (InternalId id = 0; id < kN; ++id) {
        const uint64_t key = key_dist(rng);
        t.insert(key, id);
        oracle[key].push_back(id);
    }
    ASSERT(t.height() > 2);  // several leaves' worth of splitting, not a fluke

    using Entry = std::pair<uint64_t, InternalId>;
    const auto by_key_then_id = [](const Entry& a, const Entry& b) {
        return a.first != b.first ? a.first < b.first : a.second < b.second;
    };

    // A handful of windows, including whole-range, a single key, and an inverted
    // (lo > hi) bound that must yield nothing rather than misbehave.
    const std::vector<std::pair<uint64_t, uint64_t>> windows = {
        {0, 5000}, {100, 200}, {2500, 2500}, {4999, 5000}, {6000, 7000}, {10, 5},
    };
    for (auto [lo, hi] : windows) {
        std::vector<Entry> got;
        t.range(lo, hi, [&](uint64_t k, InternalId id) { got.emplace_back(k, id); });

        // range() promises ascending key order on its own, before any sorting below.
        for (size_t i = 1; i < got.size(); ++i) ASSERT(got[i - 1].first <= got[i].first);

        std::vector<Entry> want;
        for (auto& [k, ids] : oracle) {
            if (k < lo || k > hi) continue;
            for (InternalId id : ids) want.emplace_back(k, id);
        }
        std::sort(got.begin(), got.end(), by_key_then_id);
        std::sort(want.begin(), want.end(), by_key_then_id);
        EXPECT(got == want);
    }
}

TEST(bplus_tree_height_grows_monotonically_with_volume) {
    BTreeConfig cfg;
    cfg.node_capacity = 4;
    BPlusTree t(cfg);

    size_t last_height = t.height();
    EXPECT(last_height == 1);
    for (uint64_t i = 0; i < 2000; ++i) {
        t.insert(i, static_cast<InternalId>(i));
        const size_t h = t.height();
        ASSERT(h >= last_height);  // never shrinks (no delete in this stage)
        last_height = h;
    }
    EXPECT(last_height > 2);  // this many inserts at capacity 4 must have grown deep
}

TEST(bplus_tree_keys_from_sortable_bits_round_trip_through_insert_and_find) {
    // The intended real usage: a numeric column's Int64/Float64 value goes through
    // sortable_bits() before it ever reaches the tree. This doesn't test ordering
    // (no range scan yet, that's the point of a later stage) — just that the
    // composition of the two pieces this stage and the last one built actually
    // works end to end for both source types.
    // Two trees, not one: sortable_bits() has no type tag baked into its output —
    // sortable_bits(int64_t{0}) and sortable_bits(double{0.0}) land on the exact
    // same uint64_t (0x8000000000000000), and int64_t{-1}/double{-0.0} likewise.
    // That's fine and expected: a real column's index is always one tree per
    // column, so an Int64 column's tree and a Float64 column's tree never share
    // keys in the first place. Mixing the two into one tree here would just be
    // testing a scenario the real system never creates.
    BPlusTree int_tree;
    const std::vector<int64_t> ints = {-100, -1, 0, 1, 100, INT64_MIN, INT64_MAX};
    for (size_t i = 0; i < ints.size(); ++i)
        int_tree.insert(sortable_bits(ints[i]), static_cast<InternalId>(i));

    for (size_t i = 0; i < ints.size(); ++i) {
        EXPECT(int_tree.find(sortable_bits(ints[i])) ==
               std::vector<InternalId>{static_cast<InternalId>(i)});
    }

    BPlusTree double_tree;
    const std::vector<double> doubles = {-2.5, -0.0, 0.0, 1.5, 1e300, -1e300};
    for (size_t i = 0; i < doubles.size(); ++i)
        double_tree.insert(sortable_bits(doubles[i]), static_cast<InternalId>(i));

    for (size_t i = 0; i < doubles.size(); ++i) {
        EXPECT(double_tree.find(sortable_bits(doubles[i])) ==
               std::vector<InternalId>{static_cast<InternalId>(i)});
    }
}
