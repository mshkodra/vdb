#pragma once
#include <cstdint>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#include "serialize.h"
#include "vdb_types.h"

namespace vdb {

// Metadata store for filtered ("hybrid") search. Two stores with two jobs:
//
//   columns_  filterable, hot. One dense column per declared attribute, indexed by
//             InternalId. Every column is fixed 8-byte width regardless of declared
//             type (the value is type-punned into a uint64_t), so a predicate is one
//             indexed load and one compare with no per-candidate type dispatch.
//   payload_  opaque, cold. A byte blob per vector, returned with results and never
//             used in a predicate. Touched K times per query, not thousands.
//
// Keyed by InternalId, not ExternalId, for three reasons: (1) a range predicate can't
// be served from a precomputed bitset, so it is evaluated per candidate inside the
// traversal loop, where an ExternalId hash probe would cost three dependent loads;
// (2) `deleted_` is already an InternalId-keyed filter column, so one keyspace lets
// the general predicate machinery eventually absorb the tombstone check; (3) a dense
// vector is safe to read while a writer appends, an unordered_map rehash is not —
// ExternalId keying would foreclose releasing mu_ across index_->search().
// The price is that compact() must permute the columns; see permute().
//
// Sizing, for scale: 1M rows x one uint64 column = 8 MB, against 512 MB of SIFT
// vector data. Fixed-width columnar metadata is rounding error.

enum class AttrType : uint8_t {
    Null    = 0,  // only ever a *value*; a schema never declares it
    Int64   = 1,
    Float64 = 2,
    Bool    = 3,
    Tag     = 4,  // dictionary-encoded string: stored as a uint32 code
};

// One declared attribute. The schema is fixed at VDB construction and persisted
// (by fingerprint) in the WAL and snapshot headers, like dim and metric.
struct AttrSpec {
    std::string name;
    AttrType    type = AttrType::Int64;

    // Opt-in secondary index (sorted array / B-tree) for range predicates on this
    // column, built alongside the base column once the index itself lands. Only
    // meaningful for Int64/Float64 — Tag/Bool get a postings list unconditionally
    // (it's basically free), so this flag stays false for them. Fixed at construction
    // like `type`, and covered by the same fingerprint: turning indexing on or off is
    // a layout change, same as renaming or retyping a column.
    bool indexed = false;
};

// Type-punning helpers. Every column stores 8 raw bytes; the declared type says how
// to read them. memcpy rather than a reinterpret_cast so this stays strict-aliasing
// clean and compiles to a single mov.
inline uint64_t bits_from_int(int64_t v) { uint64_t u; std::memcpy(&u, &v, 8); return u; }
inline uint64_t bits_from_double(double v) { uint64_t u; std::memcpy(&u, &v, 8); return u; }
inline int64_t  int_from_bits(uint64_t u) { int64_t v; std::memcpy(&v, &u, 8); return v; }
inline double   double_from_bits(uint64_t u) { double v; std::memcpy(&v, &u, 8); return v; }

// One attribute value at the API boundary (cold path only — never in a hot loop).
// Self-describing so the WAL encoder needs no schema.
struct AttrValue {
    AttrType    type = AttrType::Null;
    uint64_t    raw  = 0;  // Int64/Float64/Bool payload, type-punned
    std::string text;      // Tag only; the *string*, not the code

    bool    is_null() const { return type == AttrType::Null; }
    int64_t as_int() const { return int_from_bits(raw); }
    double  as_double() const { return double_from_bits(raw); }
    bool    as_bool() const { return raw != 0; }
};

inline AttrValue attr_null() { return AttrValue{}; }
inline AttrValue attr_int(int64_t v) { return AttrValue{AttrType::Int64, bits_from_int(v), {}}; }
inline AttrValue attr_float(double v) {
    return AttrValue{AttrType::Float64, bits_from_double(v), {}};
}
inline AttrValue attr_bool(bool v) {
    return AttrValue{AttrType::Bool, v ? 1u : 0u, {}};
}
inline AttrValue attr_tag(std::string v) {
    return AttrValue{AttrType::Tag, 0, std::move(v)};
}

// The metadata attached to one vector. `attrs` is positional against the declared
// schema; an empty vector means "all null", which is what a metadata-less insert
// passes. `payload` is opaque and may be empty.
struct Record {
    std::vector<AttrValue> attrs;
    std::vector<uint8_t>   payload;

    bool empty() const { return attrs.empty() && payload.empty(); }
};

class MetadataStore {
public:
    MetadataStore() = default;
    explicit MetadataStore(std::vector<AttrSpec> schema);

    const std::vector<AttrSpec>& schema() const { return schema_; }
    size_t                       attr_count() const { return schema_.size(); }
    size_t                       rows() const { return rows_; }

    // Stable hash of the declared schema (names + types). Written into the WAL and
    // snapshot headers and checked on open, so reopening a database with a changed
    // schema fails loudly instead of decoding rows as the wrong type.
    uint64_t fingerprint() const { return fingerprint_; }

    // Throws std::invalid_argument if `rec` does not match the schema. Callers run
    // this *before* mutating anything else, so a bad record can't leave the parallel
    // arrays half-appended.
    void validate(const Record& rec) const;

    // Append one row. Interns any Tag strings into that column's dictionary.
    void append_row(const Record& rec);
    // Append a copy of an existing row — what update(id, vec) uses to carry metadata
    // forward onto the replacement node without re-specifying it.
    void append_row_copy(InternalId src);
    // Overwrite a row in place. O(1), and notably touches no index structure: this is
    // the payoff of storing metadata outside the graph.
    void set_row(InternalId id, const Record& rec);

    // Keep only `live` rows, in the given order: new row i takes old row live[i].
    // Out-of-place so a row is never overwritten before it is read. Dictionaries are
    // left intact — dropping now-unused entries would renumber codes for no benefit.
    void permute(const std::vector<InternalId>& live);

    void clear_rows();

    // Row `id` has just become visible to search (VDB's publish step) / has just
    // stopped being visible (tombstoned). Tag/Bool columns keep an exact per-code
    // live count incrementally, at these two call sites plus set_row() — not
    // recomputed by scanning, so a selectivity read is O(1) regardless of table size.
    void mark_live(InternalId id);
    void mark_dead(InternalId id);

    // Rebuilds every column's live counts and postings lists from scratch against a
    // caller-supplied liveness bitmap. Neither is part of the serialized snapshot
    // bytes — this is what a snapshot load calls once, after deserialize(), since
    // that path writes columns_ directly rather than going through
    // mark_live/mark_dead/append_row.
    void rebuild_derived_state(const std::vector<bool>& deleted);

    AttrValue                   get(InternalId id, size_t attr) const;
    Record                      get_row(InternalId id) const;
    const std::vector<uint8_t>& payload(InternalId id) const;

    // --- the hot-loop seam -----------------------------------------------------
    // What a filter strategy will bind once at query-plan time and then use with no
    // per-candidate dispatch: resolve a tag string to its integer code, take the raw
    // column base pointer, and test a candidate with a load + compare.
    const uint64_t* column_raw(size_t attr) const { return columns_[attr].data.data(); }
    bool present(InternalId id, size_t attr) const { return columns_[attr].present[id]; }
    bool tag_code(size_t attr, const std::string& s, uint32_t& out) const;

    // Exact count of live rows where column `attr` holds `code` (a dictionary code
    // for Tag, 0/1 for Bool). O(1) — the payoff of maintaining it incrementally rather
    // than scanning. 0 for a code the column has never seen.
    uint32_t count(size_t attr, uint32_t code) const {
        const auto& c = columns_[attr].count;
        return code < c.size() ? c[code] : 0;
    }

    // The simplest possible inverted index: every InternalId ever written with this
    // column holding `code`. Append-only, like the HNSW graph's tombstoned nodes —
    // a row that was later deleted or overwritten to a different value leaves a
    // stale entry here rather than paying an O(k) find-and-remove on every write.
    // A consumer must additionally check present()/column_raw() (and, for liveness,
    // VDB's own deleted_) before trusting an entry; postings.size() only equals
    // count() exactly right after compact() rebuilds it (see permute()).
    const std::vector<InternalId>& postings(size_t attr, uint32_t code) const {
        static const std::vector<InternalId> empty;
        const auto& p = columns_[attr].postings;
        return code < p.size() ? p[code] : empty;
    }

    void serialize(std::vector<uint8_t>& out) const;
    void deserialize(Reader& r);

private:
    struct Column {
        AttrType              type = AttrType::Int64;
        std::vector<uint64_t> data;     // by InternalId, type-punned fixed width
        std::vector<bool>     present;  // by InternalId, null bitmap (1 bit/row)

        // Tag columns only. `dict` maps code -> string; `codes` is the reverse, used
        // at insert to intern. Codes are dense and assigned in first-seen order.
        std::vector<std::string>                  dict;
        std::unordered_map<std::string, uint32_t> codes;

        // Tag/Bool only: count[code] = number of *live* rows holding that code.
        // Parallel to `dict` for Tag (grows alongside it); fixed at size 2 for Bool.
        // Empty, untouched, for Int64/Float64 columns.
        std::vector<uint32_t> count;

        // Tag/Bool only: postings[code] = every InternalId ever written with this
        // value, in no particular order. Same shape as `count` (dict-sized for Tag,
        // size 2 for Bool) but append-only — see the public postings() accessor.
        std::vector<std::vector<InternalId>> postings;
    };

    // Tag/Bool columns only, no-op otherwise: if row `id` is present in column `a`,
    // add (increment=true) or remove (increment=false) that value's contribution to
    // its live count. Shared by mark_live/mark_dead/set_row.
    void adjust_count_(size_t a, InternalId id, bool increment);

    // Tag/Bool columns only, no-op otherwise: if row `id` is present in column `a`,
    // append it to that value's postings list. Never removes — see postings().
    void add_posting_(size_t a, InternalId id);

    uint32_t intern_(Column& c, const std::string& s);
    void     compute_fingerprint_();

    std::vector<AttrSpec>             schema_;
    std::vector<Column>               columns_;  // parallel to schema_
    std::vector<std::vector<uint8_t>> payload_;  // by InternalId
    size_t                            rows_        = 0;
    uint64_t                          fingerprint_ = 0;
};

}
