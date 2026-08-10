#pragma once
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "bplus_tree.h"
#include "serialize.h"
#include "sortable_bits.h"
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
    Text    = 5,  // tokenized into many terms, inverted-indexed (Phase B, hybrid search)
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
inline AttrValue attr_text(std::string v) {
    return AttrValue{AttrType::Text, 0, std::move(v)};
}

// Lowercase, split on runs of non-alphanumeric bytes, drop empty tokens. No
// stopword removal, no stemming — the simplest tokenizer that produces a
// deterministic term set, and deliberately easy to swap out later (nothing
// downstream depends on *how* a string becomes terms, only that the same string
// always produces the same terms). Shared by MetadataStore::intern_terms_ (index
// time) and, later, Phase B's query-time scorer — a query's term set must be
// produced the same way a document's was, or matches silently miss.
std::vector<std::string> tokenize_text(const std::string& s);

// The metadata attached to one vector. `attrs` is positional against the declared
// schema; an empty vector means "all null", which is what a metadata-less insert
// passes. `payload` is opaque and may be empty.
struct Record {
    std::vector<AttrValue> attrs;
    std::vector<uint8_t>   payload;

    bool empty() const { return attrs.empty() && payload.empty(); }
};

// One filter condition on one declared attribute — the scope this stops at, on
// purpose: Tag/Bool equality, or a numeric range, no conjunctions yet (see
// docs/design/METADATA.md's "Decisions taken" #2 and METADATA_DETAILS.md §9.3 —
// AND/OR composition is deferred work, not an oversight here). `eq`/`lo`/`hi` reuse
// AttrValue rather than a fresh typed union, the same self-describing shape a
// Record's own attrs already use.
struct Predicate {
    enum class Kind : uint8_t { Eq, Range };

    size_t    attr;
    Kind      kind;
    AttrValue eq;  // Kind::Eq only: the value to match (Tag text, or Bool)
    AttrValue lo, hi;  // Kind::Range only: inclusive bounds (Int64 or Float64)
};

inline Predicate pred_eq(size_t attr, AttrValue v) {
    return Predicate{attr, Predicate::Kind::Eq, std::move(v), attr_null(), attr_null()};
}
inline Predicate pred_range(size_t attr, AttrValue lo, AttrValue hi) {
    return Predicate{attr, Predicate::Kind::Range, attr_null(), std::move(lo), std::move(hi)};
}

// What MetadataStore::resolve() turns a Predicate into — the "hot-loop seam"
// artifact (docs/design/METADATA_DETAILS.md §4): computed *once* per query, so
// whatever later consumes this (a post-filter, a pre-filter's brute-force
// allowlist, an in-traversal admit check — PRs 11-13, not built yet) does a plain
// walk or bit test per candidate, never a per-candidate dispatch on predicate kind.
//
//   allowlist populated  — every *live* InternalId matching the predicate, exact.
//                           Order is unspecified (cheapest to produce; nothing
//                           downstream needs a particular order — a bitset build,
//                           a brute-force scan, and a set intersection are all
//                           order-agnostic). Resolvable in two cases: Tag/Bool
//                           equality (via postings(), filtered against the caller's
//                           liveness bitmap — postings alone isn't live-only, see
//                           its own doc comment) and a numeric range on an
//                           `indexed` column (via the B+-tree's range(), which is
//                           already live-only by construction — no filtering
//                           needed there).
//   allowlist == nullopt — a numeric range on a column that isn't `indexed`:
//                           materializing an exact match set would cost an O(N)
//                           scan, so this isn't done eagerly. `predicate` is still
//                           here for a future per-candidate check. No selectivity
//                           estimate is available for this case either — that
//                           needs a real estimator (histogram, sampling, or an
//                           O(N) prepass), called out as separate, unbuilt work in
//                           METADATA_DETAILS.md §4/§9, not invented here.
// One scored match from MetadataStore::search_text() — a BM25-ranked lexical
// result. `score` is higher-is-better (a similarity/relevance score), the opposite
// sense of a vector search's distance — sorted descending, not ascending.
struct TextMatch {
    InternalId id;
    float      score;
};

struct ResolvedPredicate {
    Predicate                              predicate;
    std::optional<std::vector<InternalId>> allowlist;

    // Exact selectivity — free once `allowlist` exists, since resolve() already
    // paid for it. nullopt exactly when `allowlist` is (see above).
    std::optional<size_t> selectivity() const {
        if (!allowlist) return std::nullopt;
        return allowlist->size();
    }
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
    //
    // Text reuses this exact field as its per-term document frequency (BM25's `df`):
    // a Text row can hold many term codes rather than one, so count[code] here means
    // "how many live rows contain this term," maintained the same incremental way.
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
    //
    // Text reuses this exact field too: postings(attr, term_code) is every live-or-
    // not row that ever contained that term, one entry per row (not per occurrence
    // — a repeated term within one row is deduplicated before posting, see
    // intern_terms_).
    const std::vector<InternalId>& postings(size_t attr, uint32_t code) const {
        static const std::vector<InternalId> empty;
        const auto& p = columns_[attr].postings;
        return code < p.size() ? p[code] : empty;
    }

    // Text only: resolves a token to column `attr`'s term id — the same dict/codes
    // fields tag_code() already looks up (a "tag" and a "term" are different things
    // at the call site; the underlying dictionary storage is identical either way).
    // False if the token was never interned by any row written so far.
    bool text_term_code(size_t attr, const std::string& term, uint32_t& out) const {
        return tag_code(attr, term, out);
    }

    // Text only: row `id`'s total token count in column `attr` (occurrences
    // included) — BM25's `doc_len`. 0 for an absent (null) value.
    uint32_t text_doc_len(size_t attr, InternalId id) const {
        const Column& c = columns_[attr];
        return c.present[id] ? c.doc_len[id] : 0;
    }

    // Text only: column `attr`'s live document count and average document length —
    // BM25's `N` and `avgdl`. Maintained incrementally alongside count[code] (same
    // call sites: mark_live/mark_dead/set_row/rebuild_derived_state), not scanned
    // per query. 0 if there are no live rows in this column.
    size_t text_live_count(size_t attr) const { return columns_[attr].live_doc_count; }
    double text_avg_doc_len(size_t attr) const {
        const Column& c = columns_[attr];
        return c.live_doc_count ? static_cast<double>(c.total_doc_len) / c.live_doc_count : 0.0;
    }

    // Every *live* InternalId whose column `attr` holds a value in [lo, hi]
    // (inclusive), in ascending key order, via emit(id) — a callback rather than a
    // materialized vector for the same reason BPlusTree::range() is, one level
    // down. Only meaningful for an `indexed` Int64/Float64 column; a no-op call on
    // any other column (same lenient contract postings()/count() already have for
    // an argument that doesn't apply — no throw, just nothing to report).
    //
    // Unlike postings(), this index holds exactly the live rows, not an
    // append-only superset: insert() runs at mark_live, remove() at mark_dead, and
    // a remove+insert pair at set_row's overwrite. That costs a real B+-tree
    // remove() (descent plus a possible rebalance) on every delete/overwrite of an
    // indexed column, instead of postings' O(1) append — paid because a stale
    // entry here would both slow the range scan that has to walk past it and
    // corrupt the exact-selectivity number a future cost-based planner (PR 10)
    // needs from a range query, the same wart postings.size() already has and
    // count() exists to route around.
    template <class Emit>
    void range(size_t attr, int64_t lo, int64_t hi, Emit&& emit) const {
        range_(attr, sortable_bits(lo), sortable_bits(hi), std::forward<Emit>(emit));
    }
    template <class Emit>
    void range(size_t attr, double lo, double hi, Emit&& emit) const {
        range_(attr, sortable_bits(lo), sortable_bits(hi), std::forward<Emit>(emit));
    }

    // Turns a Predicate into a ResolvedPredicate — see that struct's own doc
    // comment for the two shapes this produces and why. `deleted` is the caller's
    // liveness bitmap (VDB's own, same parameter rebuild_derived_state() already
    // takes and for the identical reason: MetadataStore itself has no notion of
    // "live," only VDB does). Throws std::invalid_argument if the predicate's kind
    // or value/bound types don't match column `attr`'s declared type — the same
    // eager, described-mismatch philosophy validate() already uses for a Record.
    ResolvedPredicate resolve(const Predicate& pred, const std::vector<bool>& deleted) const;

    // BM25-ranked search over Text column `attr` (Phase B, B4:
    // docs/plans/HYBRID_SEARCH_PLAN.md, kept local). Tokenizes `query` with the same
    // tokenize_text() documents were indexed with (a different tokenizer here would
    // make terms silently fail to match) and treats it as a *set* of distinct terms
    // — a repeated query word doesn't add extra weight, the common/simple form of
    // Okapi BM25 rather than a query-term-frequency-weighted variant.
    //
    // Candidates are the union of every query term's postings, live-filtered against
    // `deleted` (same reason resolve() takes it: MetadataStore has no notion of
    // "live" on its own) — a document needs at least *one* query term present (OR
    // semantics), not all of them, standard for ranked lexical search.
    //
    // Per-candidate term frequency (BM25's `tf`) isn't tracked persistently (B2 only
    // stores presence and total length, not occurrence counts per (term, doc) pair
    // — see HYBRID_SEARCH_PLAN.md's B4 note) — recomputed here by re-tokenizing each
    // candidate's raw text once. That only touches documents postings already
    // narrowed to, not the full corpus, the same cost shape pre-filter's brute-force
    // scan already has over its own allowlist.
    //
    // idf uses the modern always-non-negative BM25 variant, ln(1 + (N-df+0.5)/(df+0.5))
    // — the classic Robertson-Sparck-Jones form can go negative for a term present in
    // over half the corpus, which would let a "common" term actively *hurt* a
    // document's score instead of just contributing little.
    //
    // `allowlist`, when non-null, additionally restricts candidates to that id set
    // — how a structured predicate (Phase A) gates this ranker for hybrid search
    // (B5): the caller resolves the predicate once and passes its allowlist
    // through, same as collect_/prefilter_scan_ take an already-resolved predicate
    // rather than re-resolving per candidate.
    //
    // Returns the top `K` live (and, if `allowlist` is given, matching) documents by
    // score, descending. Throws std::invalid_argument if `attr` isn't a Text column.
    std::vector<TextMatch> search_text(size_t attr, const std::string& query, size_t K,
                                       const std::vector<bool>& deleted,
                                       const std::vector<InternalId>* allowlist = nullptr,
                                       float k1 = 1.2f, float b = 0.75f) const;

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

        // Tag/Bool/Text only: postings[code] = every InternalId ever written with
        // this value (Tag/Bool) or containing this term (Text), in no particular
        // order. Same shape as `count` (dict-sized for Tag/Text, size 2 for Bool)
        // but append-only — see the public postings() accessor.
        std::vector<std::vector<InternalId>> postings;

        // Text only. A row's "value" is many terms, not one, so it can't live in
        // `data`/`present` above the way every other type's does — this is Text's
        // analogue of `data[id]`. text_terms[id] holds the row's de-duplicated term
        // codes (indices into `dict`/`codes` above, interned via the same intern_()
        // Tag values use) — deliberately de-duplicated, since this is what postings/
        // doc_freq walk to decide "does this doc contain this term" (a doc counts
        // once, not once per occurrence). Not serialized — deserialize() rebuilds it
        // by re-tokenizing raw_text against the freshly-loaded dict, the same
        // "derive, don't duplicate" choice count/postings/index already make.
        std::vector<std::vector<uint32_t>> text_terms;

        // Text only. Row `id`'s *total* token count, occurrences included — BM25's
        // length norm (`doc_len` in B2's design) cares how long a document is, not
        // how large its vocabulary is, so this is deliberately not text_terms[id]
        // .size() (a repeated word inflates doc_len but not the term-code set above).
        // Same derive-don't-duplicate treatment as text_terms: not serialized,
        // recomputed by deserialize() alongside it.
        std::vector<uint32_t> doc_len;

        // Text only. BM25's N and avgdl, maintained incrementally at the same call
        // sites doc_freq (count[code]) already is (mark_live/mark_dead/set_row/
        // rebuild_derived_state) rather than scanned per query. Like count[code],
        // unaffected by permute(): compact() only keeps rows that were already
        // live, so these totals don't change, only which InternalIds hold them.
        size_t   live_doc_count = 0;  // live rows with a present value here
        uint64_t total_doc_len  = 0;  // sum of doc_len over those rows

        // Text only. The original string, by InternalId, exactly as written —
        // tokenization is lossy (word order, casing, punctuation all gone), so this
        // is what get()/get_row() return for a Text attribute, and what deserialize()
        // re-tokenizes to rebuild text_terms/doc_len above. Same shape and same
        // reason as payload_: raw bytes, serialized directly, never derived.
        std::vector<std::string> raw_text;

        // indexed Int64/Float64 only, null otherwise: keyed on sortable_bits(value),
        // holding InternalId. unique_ptr, not BPlusTree by value, because BPlusTree
        // isn't movable (its per-tree mutexes are direct members, like Node::mutex
        // one level down forces a unique_ptr there too) — and columns_ has to stay a
        // plain, resizable std::vector<Column>. Not part of the serialized snapshot
        // bytes (derived state, same as count/postings): rebuilt from scratch in
        // rebuild_derived_state()/permute()/clear_rows(), same call sites and same
        // reason those already rebuild postings.
        std::unique_ptr<BPlusTree> index;
    };

    // Tag/Bool columns only, no-op otherwise: if row `id` is present in column `a`,
    // add (increment=true) or remove (increment=false) that value's contribution to
    // its live count. Shared by mark_live/mark_dead/set_row.
    void adjust_count_(size_t a, InternalId id, bool increment);

    // Tag/Bool columns only, no-op otherwise: if row `id` is present in column `a`,
    // append it to that value's postings list. Never removes — see postings().
    void add_posting_(size_t a, InternalId id);

    // indexed Int64/Float64 columns only, no-op otherwise: if row `id` is present in
    // column `a`, insert (increment=true) or remove (increment=false) it from that
    // column's B+-tree, keyed on its *current* c.data[id] at call time — so a caller
    // removing an old value must call this *before* overwriting c.data[id], the same
    // before/after dance adjust_count_ already requires (see set_row).
    void adjust_index_(size_t a, InternalId id, bool increment);

    // Int64/Float64 only (asserts otherwise — every caller already knows the column's
    // declared type): c.data[id]'s raw bits, reinterpreted as the declared type and
    // run through sortable_bits(). The one place a Column's raw uint64_t storage gets
    // turned into a B+-tree key.
    uint64_t sortable_key_(const Column& c, InternalId id) const;

    template <class Emit>
    void range_(size_t attr, uint64_t lo, uint64_t hi, Emit&& emit) const {
        const Column& c = columns_[attr];
        if (!c.index) return;
        c.index->range(lo, hi, [&](uint64_t /*key*/, InternalId id) { emit(id); });
    }

    uint32_t intern_(Column& c, const std::string& s);

    // Text only: tokenize_text(text), intern each unique token into `c`'s dictionary
    // (via intern_() above — same dict Tag values are interned into), and return the
    // row's de-duplicated term-code set (text_terms[id]'s value). `out_doc_len`
    // receives the *un*-deduplicated token count (doc_len[id]'s value) — tokenize_text
    // is only run once per call, so both come out of the same pass rather than
    // tokenizing twice. Used at insert (append_row/set_row) and again at
    // deserialize() to rebuild text_terms/doc_len from raw_text without persisting
    // redundant arrays.
    std::vector<uint32_t> intern_terms_(Column& c, const std::string& text, uint32_t& out_doc_len);

    void compute_fingerprint_();

    std::vector<AttrSpec>             schema_;
    std::vector<Column>               columns_;  // parallel to schema_
    std::vector<std::vector<uint8_t>> payload_;  // by InternalId
    size_t                            rows_        = 0;
    uint64_t                          fingerprint_ = 0;
};

}
