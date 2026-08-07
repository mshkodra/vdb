#include "test.h"

#include "durable_vdb.h"
#include "metadata.h"
#include "snapshot.h"
#include "vdb.h"

#include <filesystem>
#include <string>
#include <vector>

#include <unistd.h>

using namespace vdb;

namespace {

// category (tag) | price (int64) | in_stock (bool) | rating (float64)
std::vector<AttrSpec> demo_schema() {
    return {{"category", AttrType::Tag},
            {"price", AttrType::Int64},
            {"in_stock", AttrType::Bool},
            {"rating", AttrType::Float64}};
}

VDBConfig demo_config(IndexKind kind = IndexKind::Brute) {
    VDBConfig cfg;
    cfg.kind   = kind;
    cfg.dim    = 2;
    cfg.metric = Metric::L2;
    cfg.schema = demo_schema();
    return cfg;
}

Record row(const std::string& cat, int64_t price, bool stock, double rating,
           std::vector<uint8_t> payload = {}) {
    Record r;
    r.attrs = {attr_tag(cat), attr_int(price), attr_bool(stock), attr_float(rating)};
    r.payload = std::move(payload);
    return r;
}

std::vector<uint8_t> bytes(const std::string& s) { return {s.begin(), s.end()}; }

std::string temp_dir(const char* name) {
    auto p = std::filesystem::temp_directory_path() /
             ("vdb_meta_" + std::string(name) + "_" + std::to_string(::getpid()));
    std::filesystem::remove_all(p);
    return p.string();
}

}  // namespace

TEST(metadata_store_round_trips_every_type) {
    MetadataStore m(demo_schema());
    m.append_row(row("shoes", 8999, true, 4.5));
    m.append_row(row("hats", -20, false, -1.25));

    EXPECT(m.rows() == 2);
    EXPECT(m.get(0, 0).text == "shoes");
    EXPECT(m.get(0, 1).as_int() == 8999);
    EXPECT(m.get(0, 2).as_bool() == true);
    EXPECT(m.get(0, 3).as_double() == 4.5);

    EXPECT(m.get(1, 0).text == "hats");
    EXPECT(m.get(1, 1).as_int() == -20);       // negative survives the u64 punning
    EXPECT(m.get(1, 2).as_bool() == false);
    EXPECT(m.get(1, 3).as_double() == -1.25);  // negative double too
}

TEST(metadata_dictionary_dedupes_and_assigns_dense_codes) {
    MetadataStore m(demo_schema());
    m.append_row(row("shoes", 1, true, 0.0));
    m.append_row(row("hats", 2, true, 0.0));
    m.append_row(row("shoes", 3, true, 0.0));

    uint32_t shoes = 999, hats = 999;
    ASSERT(m.tag_code(0, "shoes", shoes));
    ASSERT(m.tag_code(0, "hats", hats));
    EXPECT(shoes == 0);  // first seen
    EXPECT(hats == 1);

    // Row 0 and row 2 share one dictionary entry: the column holds the same code,
    // which is what makes an equality predicate a single integer compare.
    const uint64_t* col = m.column_raw(0);
    EXPECT(col[0] == shoes);
    EXPECT(col[2] == shoes);
    EXPECT(col[1] == hats);

    uint32_t missing = 0;
    EXPECT(!m.tag_code(0, "socks", missing));  // absent value -> no code
}

TEST(metadata_nulls_are_distinct_from_zero) {
    MetadataStore m(demo_schema());
    Record r;
    r.attrs = {attr_null(), attr_int(0), attr_null(), attr_float(0.0)};
    m.append_row(r);

    EXPECT(!m.present(0, 0));
    EXPECT(m.present(0, 1));
    EXPECT(m.get(0, 0).is_null());
    EXPECT(!m.get(0, 1).is_null());
    EXPECT(m.get(0, 1).as_int() == 0);  // a stored zero is not a null

    // An empty attrs vector is the "all null" shorthand a metadata-less insert uses.
    m.append_row(Record{});
    for (size_t a = 0; a < 4; ++a) EXPECT(!m.present(1, a));
}

TEST(counts_stay_zero_until_marked_live) {
    MetadataStore m(demo_schema());
    m.append_row(row("shoes", 1, true, 0.0));   // appended, not yet published
    m.append_row(row("hats", 2, false, 0.0));

    uint32_t shoes = 999, hats = 999;
    ASSERT(m.tag_code(0, "shoes", shoes));
    ASSERT(m.tag_code(0, "hats", hats));

    // A freshly appended row is pending (mirrors VDB's allocate-before-link-before-
    // publish window): interning the string grows the dictionary, but nothing is
    // counted as live yet.
    EXPECT(m.count(0, shoes) == 0);
    EXPECT(m.count(0, hats) == 0);
    EXPECT(m.count(2, 0) == 0);  // in_stock=false
    EXPECT(m.count(2, 1) == 0);  // in_stock=true

    m.mark_live(0);
    EXPECT(m.count(0, shoes) == 1);
    EXPECT(m.count(0, hats) == 0);
    EXPECT(m.count(2, 1) == 1);  // row 0's in_stock=true

    m.mark_live(1);
    EXPECT(m.count(0, shoes) == 1);
    EXPECT(m.count(0, hats) == 1);
    EXPECT(m.count(2, 0) == 1);  // row 1's in_stock=false
    EXPECT(m.count(2, 1) == 1);
}

TEST(mark_dead_removes_exactly_that_rows_contribution) {
    MetadataStore m(demo_schema());
    m.append_row(row("shoes", 1, true, 0.0));
    m.append_row(row("shoes", 2, true, 0.0));
    m.mark_live(0);
    m.mark_live(1);

    uint32_t shoes = 0;
    ASSERT(m.tag_code(0, "shoes", shoes));
    EXPECT(m.count(0, shoes) == 2);
    EXPECT(m.count(2, 1) == 2);

    m.mark_dead(0);
    EXPECT(m.count(0, shoes) == 1);  // row 1 still live
    EXPECT(m.count(2, 1) == 1);

    m.mark_dead(1);
    EXPECT(m.count(0, shoes) == 0);
    EXPECT(m.count(2, 1) == 0);
}

TEST(set_row_moves_a_live_rows_count_between_values) {
    MetadataStore m(demo_schema());
    m.append_row(row("shoes", 1, true, 0.0));
    m.mark_live(0);

    uint32_t shoes = 0, boots = 0;
    ASSERT(m.tag_code(0, "shoes", shoes));
    EXPECT(m.count(0, shoes) == 1);
    EXPECT(m.count(2, 1) == 1);  // in_stock=true

    m.set_row(0, row("boots", 1, false, 0.0));  // still live, values changed
    ASSERT(m.tag_code(0, "boots", boots));
    EXPECT(m.count(0, shoes) == 0);   // old value's contribution is gone
    EXPECT(m.count(0, boots) == 1);   // new value counted instead
    EXPECT(m.count(2, 1) == 0);       // in_stock flipped true -> false
    EXPECT(m.count(2, 0) == 1);
}

TEST(null_values_are_never_counted) {
    MetadataStore m(demo_schema());
    Record r;
    r.attrs = {attr_null(), attr_int(0), attr_null(), attr_float(0.0)};
    m.append_row(r);
    m.mark_live(0);

    // Nothing to count: the category and in_stock attrs are null on this row.
    EXPECT(m.count(0, 0) == 0);
    EXPECT(m.count(2, 0) == 0);
    EXPECT(m.count(2, 1) == 0);

    // set_row from null to a real value counts the new value with no stale decrement.
    m.set_row(0, row("shoes", 0, true, 0.0));
    uint32_t shoes = 0;
    ASSERT(m.tag_code(0, "shoes", shoes));
    EXPECT(m.count(0, shoes) == 1);
    EXPECT(m.count(2, 1) == 1);

    // And back to null again: the count returns to zero, no leak.
    m.set_row(0, r);
    EXPECT(m.count(0, shoes) == 0);
    EXPECT(m.count(2, 1) == 0);
}

TEST(count_of_a_code_never_seen_is_zero) {
    MetadataStore m(demo_schema());
    m.append_row(row("shoes", 1, true, 0.0));
    m.mark_live(0);
    EXPECT(m.count(0, 999) == 0);  // dictionary code that was never interned
}

TEST(clear_rows_zeroes_counts_but_keeps_dictionary_shape) {
    MetadataStore m(demo_schema());
    m.append_row(row("shoes", 1, true, 0.0));
    m.mark_live(0);

    uint32_t shoes = 0;
    ASSERT(m.tag_code(0, "shoes", shoes));
    EXPECT(m.count(0, shoes) == 1);

    m.clear_rows();
    EXPECT(m.count(0, shoes) == 0);
    EXPECT(m.count(2, 1) == 0);
    // The dictionary itself survives clear_rows(), so the code still resolves.
    uint32_t still = 999;
    EXPECT(m.tag_code(0, "shoes", still));
    EXPECT(still == shoes);
}

TEST(rebuild_derived_state_recomputes_counts_and_postings) {
    MetadataStore m(demo_schema());
    m.append_row(row("shoes", 1, true, 0.0));   // 0: live
    m.append_row(row("shoes", 2, false, 0.0));  // 1: dead
    m.append_row(row("hats", 3, true, 0.0));    // 2: live

    // Simulates a snapshot load: columns are populated directly (as deserialize()
    // does), counts/postings start at zero/empty, and rebuild_derived_state() is the
    // only thing that fills them in, against a liveness bitmap it did not itself
    // maintain. append_row already recorded all three rows in postings — the point
    // of this call is dropping row 1 (dead) from what it rebuilds.
    EXPECT(m.count(0, 0) == 0);

    std::vector<bool> deleted = {false, true, false};
    m.rebuild_derived_state(deleted);

    uint32_t shoes = 0, hats = 0;
    ASSERT(m.tag_code(0, "shoes", shoes));
    ASSERT(m.tag_code(0, "hats", hats));
    EXPECT(m.count(0, shoes) == 1);  // only row 0; row 1 is dead
    EXPECT(m.count(0, hats) == 1);
    EXPECT(m.count(2, 1) == 2);      // in_stock=true on rows 0 and 2
    EXPECT(m.count(2, 0) == 0);      // row 1 (in_stock=false) is dead, not counted

    EXPECT(m.postings(0, shoes) == std::vector<InternalId>{0});
    EXPECT(m.postings(0, hats) == std::vector<InternalId>{2});
    EXPECT(m.postings(2, 1) == std::vector<InternalId>({0, 2}));
    EXPECT(m.postings(2, 0).empty());  // row 1 was the only in_stock=false row, and it's dead

    // Idempotent: calling it again from the same bitmap doesn't double-count or
    // duplicate postings entries.
    m.rebuild_derived_state(deleted);
    EXPECT(m.count(0, shoes) == 1);
    EXPECT(m.count(2, 1) == 2);
    EXPECT(m.postings(2, 1) == std::vector<InternalId>({0, 2}));
}

TEST(postings_records_a_row_at_append_time_before_it_is_live) {
    MetadataStore m(demo_schema());
    m.append_row(row("shoes", 1, true, 0.0));
    m.append_row(row("hats", 2, false, 0.0));

    uint32_t shoes = 0, hats = 0;
    ASSERT(m.tag_code(0, "shoes", shoes));
    ASSERT(m.tag_code(0, "hats", hats));

    // Unlike count() (which stays 0 until mark_live), postings is populated at
    // append time — a pending row is filtered out downstream via deleted_, not by
    // withholding the postings entry.
    EXPECT(m.postings(0, shoes) == std::vector<InternalId>{0});
    EXPECT(m.postings(0, hats) == std::vector<InternalId>{1});
    EXPECT(m.postings(2, 1) == std::vector<InternalId>{0});  // row 0: in_stock=true
    EXPECT(m.postings(2, 0) == std::vector<InternalId>{1});  // row 1: in_stock=false
}

TEST(postings_keeps_stale_entries_after_mark_dead) {
    MetadataStore m(demo_schema());
    m.append_row(row("shoes", 1, true, 0.0));
    m.mark_live(0);
    m.mark_dead(0);

    uint32_t shoes = 0;
    ASSERT(m.tag_code(0, "shoes", shoes));
    // count() reflects the tombstone immediately; postings does not — mark_dead only
    // touches count(), by design (lazy cleanup, reclaimed at compact()).
    EXPECT(m.count(0, shoes) == 0);
    EXPECT(m.postings(0, shoes) == std::vector<InternalId>{0});
}

TEST(postings_keeps_the_old_entry_after_set_row_changes_the_value) {
    MetadataStore m(demo_schema());
    m.append_row(row("shoes", 1, true, 0.0));
    m.mark_live(0);
    m.set_row(0, row("boots", 1, true, 0.0));

    uint32_t shoes = 0, boots = 0;
    ASSERT(m.tag_code(0, "shoes", shoes));
    ASSERT(m.tag_code(0, "boots", boots));
    // The stale "shoes" entry survives (row 0 no longer holds that value, but
    // nothing removed it); the new "boots" entry is added alongside it.
    EXPECT(m.postings(0, shoes) == std::vector<InternalId>{0});
    EXPECT(m.postings(0, boots) == std::vector<InternalId>{0});
    // count() has no such ambiguity: it always reflects the current value only.
    EXPECT(m.count(0, shoes) == 0);
    EXPECT(m.count(0, boots) == 1);
}

TEST(postings_of_a_code_never_seen_is_empty) {
    MetadataStore m(demo_schema());
    m.append_row(row("shoes", 1, true, 0.0));
    EXPECT(m.postings(0, 999).empty());
}

TEST(clear_rows_also_clears_postings_but_keeps_dictionary_shape) {
    MetadataStore m(demo_schema());
    m.append_row(row("shoes", 1, true, 0.0));

    uint32_t shoes = 0;
    ASSERT(m.tag_code(0, "shoes", shoes));
    EXPECT(!m.postings(0, shoes).empty());

    m.clear_rows();
    EXPECT(m.postings(0, shoes).empty());
    EXPECT(m.postings(2, 1).empty());
    // The dictionary (and thus which codes postings() can resolve) survives.
    uint32_t still = 999;
    EXPECT(m.tag_code(0, "shoes", still));
    EXPECT(still == shoes);
}

TEST(permute_rebuilds_postings_dropping_dead_rows_and_renumbering_survivors) {
    MetadataStore m(demo_schema());
    m.append_row(row("shoes", 1, true, 0.0));   // 0
    m.append_row(row("shoes", 2, false, 0.0));  // 1
    m.append_row(row("hats", 3, true, 0.0));    // 2
    for (InternalId i = 0; i < 3; ++i) m.mark_live(i);
    m.mark_dead(1);  // tombstone the middle row, same as VDB::remove() would

    // Survivors keep their relative order but are renumbered: old 0 -> new 0,
    // old 2 -> new 1. This is exactly what VDB::compact() passes to permute().
    m.permute({0, 2});

    uint32_t shoes = 0, hats = 0;
    ASSERT(m.tag_code(0, "shoes", shoes));
    ASSERT(m.tag_code(0, "hats", hats));
    EXPECT(m.postings(0, shoes) == std::vector<InternalId>{0});  // old id 0 -> new id 0
    EXPECT(m.postings(0, hats) == std::vector<InternalId>{1});   // old id 2 -> new id 1
    EXPECT(m.count(0, shoes) == 1);
    EXPECT(m.count(0, hats) == 1);
}

TEST(indexed_flag_defaults_to_false) {
    MetadataStore m(demo_schema());
    for (const auto& s : m.schema()) EXPECT(!s.indexed);
}

TEST(indexed_allowed_only_on_numeric_columns) {
    // Int64 and Float64 may opt into indexed — no throw.
    { std::vector<AttrSpec> s = {{"price", AttrType::Int64, /*indexed=*/true}};
      MetadataStore m(s);
      EXPECT(m.schema()[0].indexed); }
    { std::vector<AttrSpec> s = {{"rating", AttrType::Float64, /*indexed=*/true}};
      MetadataStore m(s);
      EXPECT(m.schema()[0].indexed); }

    // Tag and Bool get a postings list unconditionally and never need this flag —
    // declaring indexed=true on them is a schema error, not a silent no-op.
    bool threw = false;
    try {
        std::vector<AttrSpec> s = {{"category", AttrType::Tag, /*indexed=*/true}};
        MetadataStore m(s);
    } catch (const std::invalid_argument&) { threw = true; }
    EXPECT(threw);

    threw = false;
    try {
        std::vector<AttrSpec> s = {{"in_stock", AttrType::Bool, /*indexed=*/true}};
        MetadataStore m(s);
    } catch (const std::invalid_argument&) { threw = true; }
    EXPECT(threw);
}

TEST(indexed_flag_changes_the_fingerprint) {
    std::vector<AttrSpec> plain  = {{"price", AttrType::Int64, false}};
    std::vector<AttrSpec> idx    = {{"price", AttrType::Int64, true}};
    MetadataStore m_plain(plain);
    MetadataStore m_idx(idx);
    // Same name, same type, different indexedness: a real layout change (the indexed
    // column carries an extra secondary structure once PR9 wires it in), so the
    // fingerprint must distinguish them the same way it distinguishes a retyped or
    // renamed column.
    EXPECT(m_plain.fingerprint() != m_idx.fingerprint());

    // Re-declaring the identical schema is deterministic and reproducible.
    MetadataStore m_idx2(idx);
    EXPECT(m_idx.fingerprint() == m_idx2.fingerprint());
}

namespace {
// price (Int64, indexed) | rating (Float64, indexed) | category (Tag, not indexed)
std::vector<AttrSpec> indexed_schema() {
    return {{"price", AttrType::Int64, /*indexed=*/true},
            {"rating", AttrType::Float64, /*indexed=*/true},
            {"category", AttrType::Tag, /*indexed=*/false}};
}

Record indexed_row(int64_t price, double rating, const std::string& cat) {
    Record r;
    r.attrs = {attr_int(price), attr_float(rating), attr_tag(cat)};
    return r;
}

std::vector<InternalId> collect_range_i64(const MetadataStore& m, size_t attr, int64_t lo,
                                          int64_t hi) {
    std::vector<InternalId> out;
    m.range(attr, lo, hi, [&](InternalId id) { out.push_back(id); });
    return out;
}

std::vector<InternalId> collect_range_f64(const MetadataStore& m, size_t attr, double lo,
                                          double hi) {
    std::vector<InternalId> out;
    m.range(attr, lo, hi, [&](InternalId id) { out.push_back(id); });
    return out;
}

// price (Int64, indexed) | rating (Float64, NOT indexed) | category (Tag) | in_stock (Bool)
// One indexed and one non-indexed numeric column on purpose, to exercise both
// resolve() Range outcomes (materialized allowlist vs. unresolved) side by side.
std::vector<AttrSpec> resolve_schema() {
    return {{"price", AttrType::Int64, /*indexed=*/true},
            {"rating", AttrType::Float64, /*indexed=*/false},
            {"category", AttrType::Tag, /*indexed=*/false},
            {"in_stock", AttrType::Bool, /*indexed=*/false}};
}

Record resolve_row(int64_t price, double rating, const std::string& cat, bool stock) {
    Record r;
    r.attrs = {attr_int(price), attr_float(rating), attr_tag(cat), attr_bool(stock)};
    return r;
}

std::vector<InternalId> sorted(std::vector<InternalId> v) {
    std::sort(v.begin(), v.end());
    return v;
}
}  // namespace

TEST(indexed_range_returns_exact_live_matches) {
    MetadataStore m(indexed_schema());
    // price: 10, 20, 30, 40, 50 — rating: 1.5, 2.5, 3.5, 4.5, 5.5
    for (int i = 0; i < 5; ++i) {
        m.append_row(indexed_row((i + 1) * 10, (i + 1) + 0.5, "cat" + std::to_string(i)));
        m.mark_live(static_cast<InternalId>(i));
    }
    EXPECT(collect_range_i64(m, 0, 20, 40) == std::vector<InternalId>({1, 2, 3}));
    EXPECT(collect_range_i64(m, 0, 0, 5) == std::vector<InternalId>{});      // below all
    EXPECT(collect_range_i64(m, 0, 100, 200) == std::vector<InternalId>{});  // above all
    EXPECT(collect_range_i64(m, 0, 10, 50) == std::vector<InternalId>({0, 1, 2, 3, 4}));

    EXPECT(collect_range_f64(m, 1, 2.0, 4.0) == std::vector<InternalId>({1, 2}));
}

TEST(indexed_range_excludes_dead_and_pending_rows) {
    MetadataStore m(indexed_schema());
    m.append_row(indexed_row(10, 1.0, "a"));  // 0
    m.append_row(indexed_row(20, 2.0, "b"));  // 1
    m.append_row(indexed_row(30, 3.0, "c"));  // 2
    m.mark_live(0);
    m.mark_live(1);
    // Row 2 stays pending — never marked live, same as a not-yet-published VDB insert.

    EXPECT(collect_range_i64(m, 0, 0, 100) == std::vector<InternalId>({0, 1}));

    m.mark_dead(1);
    EXPECT(collect_range_i64(m, 0, 0, 100) == std::vector<InternalId>{0});
}

TEST(indexed_range_reflects_set_row_overwrite) {
    MetadataStore m(indexed_schema());
    m.append_row(indexed_row(10, 1.0, "a"));
    m.mark_live(0);
    EXPECT(collect_range_i64(m, 0, 5, 15) == std::vector<InternalId>{0});

    m.set_row(0, indexed_row(50, 9.0, "a"));
    EXPECT(collect_range_i64(m, 0, 5, 15).empty());     // old value gone
    EXPECT(collect_range_i64(m, 0, 45, 55) == std::vector<InternalId>{0});  // new value present
}

TEST(indexed_range_is_a_noop_on_a_column_without_an_index) {
    // "category" (attr 2) is Tag, never indexed — range() must not throw or crash,
    // just report nothing, the same lenient contract postings()/count() already have
    // for an argument that doesn't apply.
    MetadataStore m(indexed_schema());
    m.append_row(indexed_row(10, 1.0, "a"));
    m.mark_live(0);
    EXPECT(collect_range_i64(m, 2, 0, 1000).empty());
}

TEST(permute_rebuilds_the_index_dropping_dead_rows_and_renumbering_survivors) {
    MetadataStore m(indexed_schema());
    m.append_row(indexed_row(10, 1.0, "a"));  // 0
    m.append_row(indexed_row(20, 2.0, "b"));  // 1
    m.append_row(indexed_row(30, 3.0, "c"));  // 2
    for (InternalId i = 0; i < 3; ++i) m.mark_live(i);
    m.mark_dead(1);  // tombstone the middle row, same as VDB::remove() would

    // Survivors keep their relative order but are renumbered: old 0 -> new 0,
    // old 2 -> new 1 — exactly what VDB::compact() passes to permute().
    m.permute({0, 2});

    EXPECT(collect_range_i64(m, 0, 0, 100) == std::vector<InternalId>({0, 1}));
    EXPECT(collect_range_i64(m, 0, 15, 25).empty());  // old id 1's value (20) is gone
}

TEST(snapshot_round_trip_rebuilds_the_index) {
    MetadataStore m(indexed_schema());
    for (int i = 0; i < 5; ++i) {
        m.append_row(indexed_row((i + 1) * 10, (i + 1) + 0.5, "cat" + std::to_string(i)));
        m.mark_live(static_cast<InternalId>(i));
    }
    m.mark_dead(2);  // price=30 dies

    std::vector<uint8_t> bytes_out;
    m.serialize(bytes_out);

    // Counts/postings/the index are derived, not part of the serialized bytes — this
    // only proves anything if rebuild_derived_state()'s pass over the index actually
    // ran, the same as it already does for count/postings.
    MetadataStore restored(indexed_schema());
    Reader        r(bytes_out.data(), bytes_out.size());
    restored.deserialize(r);
    std::vector<bool> deleted(5, false);
    deleted[2] = true;
    restored.rebuild_derived_state(deleted);

    EXPECT(collect_range_i64(restored, 0, 0, 100) == std::vector<InternalId>({0, 1, 3, 4}));
    EXPECT(collect_range_i64(restored, 0, 25, 35).empty());  // dead row 2's value (30)
}

TEST(resolve_eq_tag_returns_exact_live_allowlist) {
    MetadataStore m(resolve_schema());
    m.append_row(resolve_row(1, 1.0, "shoes", true));  // 0
    m.append_row(resolve_row(2, 2.0, "hats", true));   // 1
    m.append_row(resolve_row(3, 3.0, "shoes", false)); // 2
    for (InternalId i = 0; i < 3; ++i) m.mark_live(i);
    std::vector<bool> deleted(3, false);
    deleted[2] = true;  // row 2 (a "shoes") is dead

    ResolvedPredicate r = m.resolve(pred_eq(2, attr_tag("shoes")), deleted);
    ASSERT(r.allowlist.has_value());
    EXPECT(sorted(*r.allowlist) == std::vector<InternalId>{0});  // row 2 filtered out
    ASSERT(r.selectivity().has_value());
    EXPECT(*r.selectivity() == 1);
}

TEST(resolve_eq_bool_returns_exact_live_allowlist) {
    MetadataStore m(resolve_schema());
    m.append_row(resolve_row(1, 1.0, "a", true));   // 0
    m.append_row(resolve_row(2, 2.0, "b", false));  // 1
    m.append_row(resolve_row(3, 3.0, "c", true));   // 2
    for (InternalId i = 0; i < 3; ++i) m.mark_live(i);
    std::vector<bool> deleted(3, false);

    ResolvedPredicate r = m.resolve(pred_eq(3, attr_bool(true)), deleted);
    ASSERT(r.allowlist.has_value());
    EXPECT(sorted(*r.allowlist) == std::vector<InternalId>({0, 2}));
}

TEST(resolve_eq_on_a_tag_never_seen_returns_an_empty_allowlist_not_unresolved) {
    MetadataStore m(resolve_schema());
    m.append_row(resolve_row(1, 1.0, "shoes", true));
    m.mark_live(0);
    std::vector<bool> deleted(1, false);

    ResolvedPredicate r = m.resolve(pred_eq(2, attr_tag("boots")), deleted);
    ASSERT(r.allowlist.has_value());  // resolved, exact — just an empty match set
    EXPECT(r.allowlist->empty());
    EXPECT(*r.selectivity() == 0);
}

TEST(resolve_range_on_indexed_column_returns_exact_allowlist) {
    MetadataStore m(resolve_schema());
    for (int i = 0; i < 5; ++i) {
        m.append_row(resolve_row((i + 1) * 10, 0.0, "c", true));
        m.mark_live(static_cast<InternalId>(i));
    }

    ResolvedPredicate r = m.resolve(pred_range(0, attr_int(20), attr_int(40)), {});
    ASSERT(r.allowlist.has_value());
    EXPECT(sorted(*r.allowlist) == std::vector<InternalId>({1, 2, 3}));
    EXPECT(*r.selectivity() == 3);
}

TEST(resolve_range_on_a_non_indexed_column_is_unresolved) {
    MetadataStore m(resolve_schema());
    m.append_row(resolve_row(1, 5.0, "c", true));
    m.mark_live(0);

    // "rating" (attr 1) is Float64 but not `indexed` — resolve() must not silently
    // scan; it reports "can't answer this cheaply" instead.
    ResolvedPredicate r = m.resolve(pred_range(1, attr_float(0.0), attr_float(10.0)), {});
    EXPECT(!r.allowlist.has_value());
    EXPECT(!r.selectivity().has_value());
    // The original predicate survives, for a future per-candidate check.
    EXPECT(r.predicate.attr == 1);
    EXPECT(r.predicate.lo.as_double() == 0.0);
    EXPECT(r.predicate.hi.as_double() == 10.0);
}

TEST(resolve_rejects_kind_and_value_type_mismatches) {
    MetadataStore m(resolve_schema());

    auto expect_throw = [&](const Predicate& p) {
        bool threw = false;
        try {
            m.resolve(p, {});
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        EXPECT(threw);
    };

    expect_throw(pred_eq(0, attr_int(5)));               // Eq on Int64 (needs Tag/Bool)
    expect_throw(pred_range(2, attr_tag("x"), attr_tag("y")));  // Range on Tag
    expect_throw(pred_eq(2, attr_bool(true)));            // right kind, wrong value type
    expect_throw(pred_range(0, attr_float(1.0), attr_float(2.0)));  // right kind, wrong bound type
}

TEST(metadata_rejects_schema_violations) {
    MetadataStore m(demo_schema());

    bool threw = false;
    try {
        Record r;
        r.attrs = {attr_tag("shoes")};  // arity 1 against a 4-attribute schema
        m.append_row(r);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    EXPECT(threw);

    threw = false;
    try {
        Record r;
        r.attrs = {attr_int(5), attr_int(1), attr_bool(true), attr_float(0.0)};
        m.append_row(r);  // attr 0 is a Tag, not an Int64
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    EXPECT(threw);

    EXPECT(m.rows() == 0);  // neither bad row was appended
}

TEST(metadata_survives_insert_and_readback) {
    VDB db(demo_config());
    const float a[2] = {0.0f, 0.0f};
    const float b[2] = {5.0f, 5.0f};

    const ExternalId id_a = db.insert(a, row("shoes", 8999, true, 4.5, bytes("sku-A")));
    const ExternalId id_b = db.insert(b, row("hats", 1200, false, 3.0, bytes("sku-B")));

    Record got;
    ASSERT(db.get_metadata(id_a, got));
    EXPECT(got.attrs[0].text == "shoes");
    EXPECT(got.attrs[1].as_int() == 8999);
    EXPECT(got.payload == bytes("sku-A"));

    ASSERT(db.get_metadata(id_b, got));
    EXPECT(got.attrs[0].text == "hats");
    EXPECT(got.payload == bytes("sku-B"));
}

TEST(metadata_insert_without_schema_still_works) {
    VDBConfig cfg;  // no schema declared at all
    cfg.kind = IndexKind::Brute;
    cfg.dim  = 2;
    VDB db(cfg);

    const float v[2] = {1.0f, 1.0f};
    Record r;
    r.payload = bytes("just-a-payload");
    const ExternalId id = db.insert(v, r);

    Record got;
    ASSERT(db.get_metadata(id, got));
    EXPECT(got.attrs.empty());
    EXPECT(got.payload == bytes("just-a-payload"));
}

TEST(search_hits_carry_payload_in_distance_order) {
    VDB db(demo_config());
    const float near_[2]  = {0.0f, 0.0f};
    const float far_[2]   = {9.0f, 9.0f};
    const float query[2]  = {0.1f, 0.1f};

    db.insert(near_, row("shoes", 1, true, 1.0, bytes("near")));
    db.insert(far_, row("hats", 2, true, 2.0, bytes("far")));

    const auto hits = db.search_hits(query, 2);
    ASSERT(hits.size() == 2);
    EXPECT(hits[0].payload == bytes("near"));
    EXPECT(hits[1].payload == bytes("far"));
    EXPECT(hits[0].dist <= hits[1].dist);

    // The bare search() overload still returns ids only, and agrees on the order.
    const auto ids = db.search(query, 2);
    ASSERT(ids.size() == 2);
    EXPECT(ids[0] == hits[0].id);
}

TEST(set_metadata_replaces_row_without_moving_the_vector) {
    VDB db(demo_config());
    const float v[2] = {1.0f, 2.0f};
    const ExternalId id = db.insert(v, row("shoes", 100, true, 1.0, bytes("v1")));

    const size_t deleted_before = db.deleted_count();
    ASSERT(db.set_metadata(id, row("boots", 200, false, 2.0, bytes("v2"))));

    // No tombstone: unlike update(), a metadata-only write leaves the graph alone.
    EXPECT(db.deleted_count() == deleted_before);
    EXPECT(db.size() == 1);

    Record got;
    ASSERT(db.get_metadata(id, got));
    EXPECT(got.attrs[0].text == "boots");
    EXPECT(got.attrs[1].as_int() == 200);
    EXPECT(got.attrs[2].as_bool() == false);
    EXPECT(got.payload == bytes("v2"));

    EXPECT(!db.set_metadata(9999, row("x", 1, true, 1.0)));  // unknown id
}

TEST(update_carries_metadata_forward_by_default) {
    VDB db(demo_config());
    const float v1[2] = {1.0f, 1.0f};
    const float v2[2] = {7.0f, 7.0f};
    const ExternalId id = db.insert(v1, row("shoes", 100, true, 4.0, bytes("keep")));

    ASSERT(db.update(id, v2));  // vector-only update

    Record got;
    ASSERT(db.get_metadata(id, got));
    EXPECT(got.attrs[0].text == "shoes");
    EXPECT(got.attrs[1].as_int() == 100);
    EXPECT(got.payload == bytes("keep"));

    // The explicit overload replaces the row instead.
    ASSERT(db.update(id, v1, row("clogs", 5, false, 1.0, bytes("new"))));
    ASSERT(db.get_metadata(id, got));
    EXPECT(got.attrs[0].text == "clogs");
    EXPECT(got.payload == bytes("new"));
}

TEST(vdb_insert_and_remove_maintain_live_attr_counts) {
    VDB db(demo_config());
    const float v[2] = {0.0f, 0.0f};

    const ExternalId a = db.insert(v, row("shoes", 1, true, 0.0));
    const ExternalId b = db.insert(v, row("shoes", 2, true, 0.0));
    db.insert(v, row("hats", 3, false, 0.0));

    // category is attr 0, in_stock is attr 2 in demo_schema(); codes are assigned in
    // first-seen order, so "shoes" is 0 here.
    EXPECT(db.attr_count(0, 0) == 2);  // shoes
    EXPECT(db.attr_count(0, 1) == 1);  // hats
    EXPECT(db.attr_count(2, 1) == 2);  // in_stock=true
    EXPECT(db.attr_count(2, 0) == 1);  // in_stock=false

    ASSERT(db.remove(a));
    EXPECT(db.attr_count(0, 0) == 1);  // one "shoes" row tombstoned
    EXPECT(db.attr_count(2, 1) == 1);

    ASSERT(db.remove(b));
    EXPECT(db.attr_count(0, 0) == 0);  // no live "shoes" rows left
    EXPECT(db.attr_count(2, 1) == 0);
}

TEST(vdb_update_moves_the_count_from_old_row_to_new) {
    VDB db(demo_config());
    const float v1[2] = {1.0f, 1.0f};
    const float v2[2] = {5.0f, 5.0f};
    const ExternalId id = db.insert(v1, row("shoes", 1, true, 0.0));
    EXPECT(db.attr_count(0, 0) == 1);  // shoes

    // Vector-only update carries the same metadata forward: old node tombstoned,
    // new node live, same attribute value — net count is unchanged.
    ASSERT(db.update(id, v2));
    EXPECT(db.attr_count(0, 0) == 1);

    // Update with a different metadata row moves the count to the new value.
    ASSERT(db.update(id, v1, row("boots", 1, false, 0.0)));
    uint32_t boots_code = 1;  // second tag string seen by this column
    EXPECT(db.attr_count(0, 0) == 0);              // shoes: gone
    EXPECT(db.attr_count(0, boots_code) == 1);      // boots: counted
    EXPECT(db.attr_count(2, 1) == 0);
    EXPECT(db.attr_count(2, 0) == 1);
}

TEST(vdb_set_metadata_moves_the_count_between_values) {
    VDB db(demo_config());
    const float v[2] = {1.0f, 1.0f};
    const ExternalId id = db.insert(v, row("shoes", 1, true, 0.0));
    EXPECT(db.attr_count(0, 0) == 1);

    ASSERT(db.set_metadata(id, row("boots", 1, false, 0.0)));
    EXPECT(db.attr_count(0, 0) == 0);  // shoes count released
    EXPECT(db.attr_count(0, 1) == 1);  // boots counted
    EXPECT(db.attr_count(2, 1) == 0);
    EXPECT(db.attr_count(2, 0) == 1);
}

TEST(compact_permutes_metadata_with_internal_ids) {
    VDB db(demo_config(IndexKind::HNSW));
    std::vector<ExternalId> ids;
    for (int i = 0; i < 10; ++i) {
        const float v[2] = {static_cast<float>(i), static_cast<float>(i)};
        ids.push_back(db.insert(v, row("cat" + std::to_string(i), i, i % 2 == 0,
                                       i * 0.5, bytes("p" + std::to_string(i)))));
    }
    // Tombstone the even ones, so the survivors are not a contiguous id range.
    // Evens have in_stock=true (i % 2 == 0); after removing them, only in_stock=false
    // survivors remain.
    for (int i = 0; i < 10; i += 2) ASSERT(db.remove(ids[i]));
    ASSERT(db.deleted_count() == 5);
    EXPECT(db.attr_count(2, 1) == 0);  // in_stock=true: none left
    EXPECT(db.attr_count(2, 0) == 5);  // in_stock=false: the 5 odd survivors

    db.compact();
    EXPECT(db.deleted_count() == 0);
    EXPECT(db.size() == 5);
    // Compaction only renumbers internal ids; per-code live counts are untouched.
    EXPECT(db.attr_count(2, 1) == 0);
    EXPECT(db.attr_count(2, 0) == 5);

    // Every surviving external id must still see its own row after renumbering.
    for (int i = 1; i < 10; i += 2) {
        Record got;
        ASSERT(db.get_metadata(ids[i], got));
        EXPECT(got.attrs[0].text == "cat" + std::to_string(i));
        EXPECT(got.attrs[1].as_int() == i);
        EXPECT(got.attrs[3].as_double() == i * 0.5);
        EXPECT(got.payload == bytes("p" + std::to_string(i)));
    }
    // And the dead rows are gone with their ids.
    Record dead;
    EXPECT(!db.get_metadata(ids[0], dead));
}

TEST(snapshot_round_trips_metadata) {
    const std::string dir = temp_dir("snap");
    std::filesystem::create_directories(dir);
    const std::string path = dir + "/snapshot";

    std::vector<ExternalId> ids;
    {
        VDB db(demo_config());
        for (int i = 0; i < 5; ++i) {
            const float v[2] = {static_cast<float>(i), 0.0f};
            ids.push_back(db.insert(v, row(i % 2 ? "shoes" : "hats", i * 10, i % 2 == 0,
                                           i * 1.5, bytes("pay" + std::to_string(i)))));
        }
        ASSERT(db.remove(ids[2]));  // keep a tombstone in the image
        save_snapshot(db, path, /*lsn=*/99);
    }

    VDB restored(demo_config());
    EXPECT(load_snapshot(restored, path) == 99);
    EXPECT(restored.size() == 4);

    for (int i = 0; i < 5; ++i) {
        Record got;
        if (i == 2) { EXPECT(!restored.get_metadata(ids[i], got)); continue; }
        ASSERT(restored.get_metadata(ids[i], got));
        EXPECT(got.attrs[0].text == (i % 2 ? "shoes" : "hats"));
        EXPECT(got.attrs[1].as_int() == i * 10);
        EXPECT(got.attrs[3].as_double() == i * 1.5);
        EXPECT(got.payload == bytes("pay" + std::to_string(i)));
    }
    std::filesystem::remove_all(dir);
}

TEST(snapshot_round_trip_rebuilds_attr_counts) {
    const std::string dir = temp_dir("snapcounts");
    std::filesystem::create_directories(dir);
    const std::string path = dir + "/snapshot";

    std::vector<ExternalId> ids;
    {
        VDB db(demo_config());
        for (int i = 0; i < 5; ++i) {
            const float v[2] = {static_cast<float>(i), 0.0f};
            ids.push_back(db.insert(v, row(i % 2 ? "shoes" : "hats", i, i % 2 == 0, 0.0)));
        }
        // Rows: 0=hats/true 1=shoes/false 2=hats/true 3=shoes/false 4=hats/true.
        ASSERT(db.remove(ids[2]));  // one of the three "hats"/in_stock=true rows dies
        // Live: 0(hats,true) 1(shoes,false) 3(shoes,false) 4(hats,true).
        EXPECT(db.attr_count(0, 0 /*hats, first-seen*/) == 2);
        EXPECT(db.attr_count(0, 1 /*shoes*/) == 2);
        save_snapshot(db, path, /*lsn=*/1);
    }

    // Counts are not part of the snapshot bytes — this only proves anything if the
    // fresh instance's rebuild_derived_state() pass actually ran on load.
    VDB restored(demo_config());
    load_snapshot(restored, path);
    EXPECT(restored.size() == 4);
    EXPECT(restored.attr_count(0, 0) == 2);  // hats
    EXPECT(restored.attr_count(0, 1) == 2);  // shoes
    EXPECT(restored.attr_count(2, 1) == 2);  // in_stock=true: rows 0, 4
    EXPECT(restored.attr_count(2, 0) == 2);  // in_stock=false: rows 1, 3
    std::filesystem::remove_all(dir);
}

TEST(snapshot_rejects_a_changed_schema) {
    const std::string dir = temp_dir("snapfp");
    std::filesystem::create_directories(dir);
    const std::string path = dir + "/snapshot";
    {
        VDB db(demo_config());
        const float v[2] = {1.0f, 1.0f};
        db.insert(v, row("shoes", 1, true, 1.0));
        save_snapshot(db, path, 1);
    }

    VDBConfig changed = demo_config();
    changed.schema[1].name = "cost";  // same types, different name -> different layout
    VDB other(changed);

    bool threw = false;
    try {
        load_snapshot(other, path);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    EXPECT(threw);
    std::filesystem::remove_all(dir);
}

TEST(snapshot_rejects_a_schema_that_only_flipped_indexed) {
    const std::string dir = temp_dir("snapfpidx");
    std::filesystem::create_directories(dir);
    const std::string path = dir + "/snapshot";
    {
        VDB db(demo_config());  // price (attr 1, Int64) not indexed
        const float v[2] = {1.0f, 1.0f};
        db.insert(v, row("shoes", 1, true, 1.0));
        save_snapshot(db, path, 1);
    }

    VDBConfig changed = demo_config();
    changed.schema[1].indexed = true;  // same name/type, only indexedness differs
    VDB other(changed);

    bool threw = false;
    try {
        load_snapshot(other, path);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    EXPECT(threw);
    std::filesystem::remove_all(dir);
}

TEST(wal_record_round_trips_metadata) {
    WalRecord in{WalRecordType::Insert, /*lsn=*/7, /*ext_id=*/3, {1.0f, 2.0f}, {}};
    in.meta = row("shoes", -8999, true, 4.5, bytes("blob"));

    const auto frame = encode_record(in);
    WalRecord out;
    size_t    consumed = 0;
    ASSERT(decode_record(frame.data(), frame.size(), out, consumed) == DecodeStatus::Ok);
    EXPECT(consumed == frame.size());
    EXPECT(out.vec == in.vec);
    ASSERT(out.meta.attrs.size() == 4);
    EXPECT(out.meta.attrs[0].type == AttrType::Tag);
    EXPECT(out.meta.attrs[0].text == "shoes");
    EXPECT(out.meta.attrs[1].as_int() == -8999);
    EXPECT(out.meta.attrs[2].as_bool() == true);
    EXPECT(out.meta.attrs[3].as_double() == 4.5);
    EXPECT(out.meta.payload == bytes("blob"));
}

TEST(wal_setmeta_record_round_trips_without_a_vector) {
    WalRecord in{WalRecordType::SetMeta, 11, 4, {}, {}};
    in.meta = row("hats", 42, false, 0.25);

    const auto frame = encode_record(in);
    WalRecord out;
    size_t    consumed = 0;
    ASSERT(decode_record(frame.data(), frame.size(), out, consumed) == DecodeStatus::Ok);
    EXPECT(out.type == WalRecordType::SetMeta);
    EXPECT(out.vec.empty());
    EXPECT(out.meta.attrs[0].text == "hats");
    EXPECT(out.meta.attrs[1].as_int() == 42);
}

TEST(wal_null_attributes_round_trip) {
    WalRecord in{WalRecordType::Insert, 1, 1, {3.0f}, {}};
    in.meta.attrs = {attr_null(), attr_int(7), attr_null(), attr_null()};

    const auto frame = encode_record(in);
    WalRecord out;
    size_t    consumed = 0;
    ASSERT(decode_record(frame.data(), frame.size(), out, consumed) == DecodeStatus::Ok);
    ASSERT(out.meta.attrs.size() == 4);
    EXPECT(out.meta.attrs[0].is_null());
    EXPECT(out.meta.attrs[1].as_int() == 7);
    EXPECT(out.meta.attrs[3].is_null());
}

TEST(durable_recovery_restores_metadata) {
    const std::string dir = temp_dir("durable");
    std::vector<ExternalId> ids;
    {
        DurableVDB db(demo_config(), dir);
        for (int i = 0; i < 6; ++i) {
            const float v[2] = {static_cast<float>(i), 0.0f};
            ids.push_back(db.insert(v, row("c" + std::to_string(i), i * 3, i % 2 == 0,
                                           i * 0.25, bytes("b" + std::to_string(i)))));
        }
        // Exercise all three metadata write paths before the crash.
        const float moved[2] = {100.0f, 0.0f};
        ASSERT(db.update(ids[1], moved));                                  // carry forward
        ASSERT(db.set_metadata(ids[2], row("changed", 777, false, 9.5, bytes("new"))));
        ASSERT(db.remove(ids[3]));
        db.sync();
    }  // destructor only best-effort syncs; the WAL is on disk

    DurableVDB reopened(demo_config(), dir);
    EXPECT(reopened.size() == 5);

    Record got;
    ASSERT(reopened.get_metadata(ids[1], got));
    EXPECT(got.attrs[0].text == "c1");        // update() kept the old row
    EXPECT(got.attrs[1].as_int() == 3);
    EXPECT(got.payload == bytes("b1"));

    ASSERT(reopened.get_metadata(ids[2], got));
    EXPECT(got.attrs[0].text == "changed");   // SetMeta replayed
    EXPECT(got.attrs[1].as_int() == 777);
    EXPECT(got.payload == bytes("new"));

    EXPECT(!reopened.get_metadata(ids[3], got));  // deleted

    ASSERT(reopened.get_metadata(ids[5], got));
    EXPECT(got.attrs[0].text == "c5");
    EXPECT(got.attrs[3].as_double() == 5 * 0.25);

    std::filesystem::remove_all(dir);
}

TEST(durable_recovery_across_a_checkpoint_keeps_metadata) {
    const std::string dir = temp_dir("ckpt");
    std::vector<ExternalId> ids;
    {
        DurableVDB db(demo_config(), dir);
        for (int i = 0; i < 4; ++i) {
            const float v[2] = {static_cast<float>(i), 1.0f};
            ids.push_back(db.insert(v, row("pre" + std::to_string(i), i, true, i,
                                           bytes("pre" + std::to_string(i)))));
        }
        db.checkpoint();  // rows 0..3 now live in the snapshot
        for (int i = 4; i < 7; ++i) {
            const float v[2] = {static_cast<float>(i), 1.0f};
            ids.push_back(db.insert(v, row("post" + std::to_string(i), i, false, i,
                                           bytes("post" + std::to_string(i)))));
        }
        db.sync();  // rows 4..6 only in the WAL tail
    }

    DurableVDB reopened(demo_config(), dir);
    EXPECT(reopened.size() == 7);
    for (int i = 0; i < 7; ++i) {
        Record got;
        ASSERT(reopened.get_metadata(ids[i], got));
        const std::string want = (i < 4 ? "pre" : "post") + std::to_string(i);
        EXPECT(got.attrs[0].text == want);
        EXPECT(got.payload == bytes(want));
    }
    std::filesystem::remove_all(dir);
}

TEST(durable_recovery_restores_attr_counts_across_checkpoint_and_wal_tail) {
    const std::string dir = temp_dir("ckptcounts");
    std::vector<ExternalId> ids;
    {
        DurableVDB db(demo_config(), dir);
        // Pre-checkpoint: 4 rows, in_stock=true, land in the snapshot image.
        for (int i = 0; i < 4; ++i) {
            const float v[2] = {static_cast<float>(i), 1.0f};
            ids.push_back(db.insert(v, row("pre", i, true, i)));
        }
        EXPECT(db.attr_count(2, 1) == 4);
        db.checkpoint();  // exercises save_snapshot; a reopen will exercise rebuild_derived_state

        // Post-checkpoint: 3 more rows, in_stock=false, only in the WAL tail.
        for (int i = 4; i < 7; ++i) {
            const float v[2] = {static_cast<float>(i), 1.0f};
            ids.push_back(db.insert(v, row("post", i, false, i)));
        }
        // One pre-checkpoint row is removed and one post-checkpoint row is updated
        // to a different tag+bool, purely via the WAL — both must replay correctly.
        ASSERT(db.remove(ids[0]));
        ASSERT(db.set_metadata(ids[4], row("changed", 99, true, 0.0)));
        db.sync();
    }

    DurableVDB reopened(demo_config(), dir);
    EXPECT(reopened.size() == 6);
    // Live: pre x3 (ids[1..3], in_stock=true), post x2 (ids[5..6], in_stock=false),
    // changed x1 (ids[4], in_stock=true).
    uint32_t pre = 0, post = 1, changed = 2;
    Record probe;
    ASSERT(reopened.get_metadata(ids[1], probe));
    EXPECT(probe.attrs[0].text == "pre");
    ASSERT(reopened.get_metadata(ids[5], probe));
    EXPECT(probe.attrs[0].text == "post");
    ASSERT(reopened.get_metadata(ids[4], probe));
    EXPECT(probe.attrs[0].text == "changed");

    EXPECT(reopened.attr_count(0, pre) == 3);
    EXPECT(reopened.attr_count(0, post) == 2);
    EXPECT(reopened.attr_count(0, changed) == 1);
    EXPECT(reopened.attr_count(2, 1) == 4);  // in_stock=true: pre x3 + changed x1
    EXPECT(reopened.attr_count(2, 0) == 2);  // in_stock=false: post x2

    std::filesystem::remove_all(dir);
}
