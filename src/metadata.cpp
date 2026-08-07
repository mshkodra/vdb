#include "metadata.h"

#include <algorithm>
#include <stdexcept>

namespace vdb {

namespace {

const char* type_name(AttrType t) {
    switch (t) {
        case AttrType::Null:    return "null";
        case AttrType::Int64:   return "int64";
        case AttrType::Float64: return "float64";
        case AttrType::Bool:    return "bool";
        case AttrType::Tag:     return "tag";
    }
    return "?";
}

}  // namespace

MetadataStore::MetadataStore(std::vector<AttrSpec> schema) : schema_(std::move(schema)) {
    columns_.resize(schema_.size());
    for (size_t a = 0; a < schema_.size(); ++a) {
        if (schema_[a].type == AttrType::Null)
            throw std::invalid_argument("attribute '" + schema_[a].name +
                                        "' declared with type null");
        if (schema_[a].indexed && schema_[a].type != AttrType::Int64 &&
            schema_[a].type != AttrType::Float64)
            throw std::invalid_argument("attribute '" + schema_[a].name + "' is indexed but " +
                                        type_name(schema_[a].type) +
                                        " columns are not indexable (Tag/Bool get a"
                                        " postings list unconditionally)");
        columns_[a].type = schema_[a].type;
        // Bool's two codes (false/true) are known up front, unlike Tag's dictionary,
        // which grows as strings are interned.
        if (columns_[a].type == AttrType::Bool) {
            columns_[a].count.assign(2, 0);
            columns_[a].postings.assign(2, {});
        }
        if (schema_[a].indexed) columns_[a].index = std::make_unique<BPlusTree>();
    }
    compute_fingerprint_();
}

// FNV-1a over the declared (name, type, indexed) triples. Order-sensitive on purpose:
// the columns are positional, so reordering the schema really is a different layout.
void MetadataStore::compute_fingerprint_() {
    uint64_t h = 1469598103934665603ull;
    auto mix   = [&h](uint8_t byte) {
        h ^= byte;
        h *= 1099511628211ull;
    };
    for (const auto& s : schema_) {
        for (char c : s.name) mix(static_cast<uint8_t>(c));
        mix(0);
        mix(static_cast<uint8_t>(s.type));
        mix(s.indexed ? 1 : 0);
    }
    fingerprint_ = h;
}

void MetadataStore::validate(const Record& rec) const {
    if (rec.attrs.empty()) return;  // "all null" shorthand
    if (rec.attrs.size() != schema_.size())
        throw std::invalid_argument("metadata arity mismatch: got " +
                                    std::to_string(rec.attrs.size()) + ", schema declares " +
                                    std::to_string(schema_.size()));
    for (size_t a = 0; a < rec.attrs.size(); ++a) {
        const AttrValue& v = rec.attrs[a];
        if (v.is_null()) continue;  // an explicit null is always allowed
        if (v.type != schema_[a].type)
            throw std::invalid_argument("attribute '" + schema_[a].name + "' expects " +
                                        type_name(schema_[a].type) + ", got " +
                                        type_name(v.type));
    }
}

uint32_t MetadataStore::intern_(Column& c, const std::string& s) {
    auto it = c.codes.find(s);
    if (it != c.codes.end()) return it->second;
    const uint32_t code = static_cast<uint32_t>(c.dict.size());
    c.dict.push_back(s);
    c.codes.emplace(s, code);
    c.count.push_back(0);        // a brand-new code starts with zero live rows
    c.postings.emplace_back();   // ...and an empty postings list
    return code;
}

// Tag/Bool only; a no-op for any other column. `increment` is the row's current
// contribution being added or removed from that value's live count.
void MetadataStore::adjust_count_(size_t a, InternalId id, bool increment) {
    Column& c = columns_[a];
    if ((c.type != AttrType::Tag && c.type != AttrType::Bool) || !c.present[id]) return;
    const uint32_t code = static_cast<uint32_t>(c.data[id]);
    if (increment) ++c.count[code];
    else           --c.count[code];
}

// Tag/Bool only; a no-op for any other column or an absent value. Never removes an
// existing entry — see the postings() accessor's doc comment for why.
void MetadataStore::add_posting_(size_t a, InternalId id) {
    Column& c = columns_[a];
    if ((c.type != AttrType::Tag && c.type != AttrType::Bool) || !c.present[id]) return;
    const uint32_t code = static_cast<uint32_t>(c.data[id]);
    c.postings[code].push_back(id);
}

uint64_t MetadataStore::sortable_key_(const Column& c, InternalId id) const {
    return c.type == AttrType::Int64 ? sortable_bits(int_from_bits(c.data[id]))
                                      : sortable_bits(double_from_bits(c.data[id]));
}

// indexed columns only; a no-op for any other column or an absent value. Reads
// c.data[id] at call time, so — unlike adjust_count_, which only *reads* the code
// to increment/decrement a count — a caller removing an old value must call this
// before overwriting c.data[id] (see set_row).
void MetadataStore::adjust_index_(size_t a, InternalId id, bool increment) {
    Column& c = columns_[a];
    if (!c.index || !c.present[id]) return;
    const uint64_t key = sortable_key_(c, id);
    if (increment) c.index->insert(key, id);
    else           c.index->remove(key, id);
}

void MetadataStore::mark_live(InternalId id) {
    for (size_t a = 0; a < columns_.size(); ++a) {
        adjust_count_(a, id, /*increment=*/true);
        adjust_index_(a, id, /*increment=*/true);
    }
}

void MetadataStore::mark_dead(InternalId id) {
    for (size_t a = 0; a < columns_.size(); ++a) {
        adjust_count_(a, id, /*increment=*/false);
        adjust_index_(a, id, /*increment=*/false);
    }
}

void MetadataStore::rebuild_derived_state(const std::vector<bool>& deleted) {
    for (auto& c : columns_) {
        if (c.type == AttrType::Tag || c.type == AttrType::Bool) {
            std::fill(c.count.begin(), c.count.end(), 0u);
            for (auto& p : c.postings) p.clear();
        }
        // Fresh, empty tree rather than an O(k) walk removing every entry — same
        // "rebuild from scratch" call this function already makes for count/postings.
        if (c.index) c.index = std::make_unique<BPlusTree>();
    }

    for (InternalId id = 0; id < rows_; ++id) {
        if (deleted[id]) continue;
        for (size_t a = 0; a < columns_.size(); ++a) {
            adjust_count_(a, id, /*increment=*/true);
            adjust_index_(a, id, /*increment=*/true);
            add_posting_(a, id);
        }
    }
}

bool MetadataStore::tag_code(size_t attr, const std::string& s, uint32_t& out) const {
    const Column& c = columns_[attr];
    auto it = c.codes.find(s);
    if (it == c.codes.end()) return false;  // value absent from the dictionary
    out = it->second;
    return true;
}

void MetadataStore::append_row(const Record& rec) {
    validate(rec);
    const InternalId id = static_cast<InternalId>(rows_);
    for (size_t a = 0; a < columns_.size(); ++a) {
        Column& c = columns_[a];
        if (rec.attrs.empty() || rec.attrs[a].is_null()) {
            c.data.push_back(0);
            c.present.push_back(false);
            continue;
        }
        const AttrValue& v = rec.attrs[a];
        // Tag values arrive as strings and are interned here, so the hot loop only
        // ever sees an integer code.
        c.data.push_back(v.type == AttrType::Tag ? intern_(c, v.text) : v.raw);
        c.present.push_back(true);
    }
    payload_.push_back(rec.payload);
    ++rows_;
    // Postings record every value a row was ever written with, independent of
    // mark_live/mark_dead — a still-pending row (VDB hasn't published it yet) is
    // filtered out at read time via deleted_, the same way a pending row already
    // reads as absent from search().
    for (size_t a = 0; a < columns_.size(); ++a) add_posting_(a, id);
}

void MetadataStore::append_row_copy(InternalId src) {
    const InternalId id = static_cast<InternalId>(rows_);
    for (auto& c : columns_) {
        c.data.push_back(c.data[src]);
        c.present.push_back(c.present[src]);
    }
    payload_.push_back(payload_[src]);
    ++rows_;
    for (size_t a = 0; a < columns_.size(); ++a) add_posting_(a, id);
}

void MetadataStore::set_row(InternalId id, const Record& rec) {
    validate(rec);
    // A row reached here is always live (set_metadata resolves it through
    // ext_to_int_, which only holds published rows), so the value being replaced was
    // already counted — drop it before overwriting, add the new value back after.
    for (size_t a = 0; a < columns_.size(); ++a) {
        adjust_count_(a, id, /*increment=*/false);
        adjust_index_(a, id, /*increment=*/false);
        Column& c = columns_[a];
        if (rec.attrs.empty() || rec.attrs[a].is_null()) {
            c.data[id]    = 0;
            c.present[id] = false;
        } else {
            const AttrValue& v = rec.attrs[a];
            c.data[id]    = (v.type == AttrType::Tag ? intern_(c, v.text) : v.raw);
            c.present[id] = true;
        }
        adjust_count_(a, id, /*increment=*/true);
        adjust_index_(a, id, /*increment=*/true);
        // The old value's postings entry (if any) goes stale, not removed — same
        // lazy-cleanup story as add_posting_ everywhere else.
        add_posting_(a, id);
    }
    payload_[id] = rec.payload;
}

void MetadataStore::permute(const std::vector<InternalId>& live) {
    for (auto& c : columns_) {
        std::vector<uint64_t> data;
        std::vector<bool>     present;
        data.reserve(live.size());
        present.reserve(live.size());
        for (InternalId src : live) {
            data.push_back(c.data[src]);
            present.push_back(c.present[src]);
        }
        c.data.swap(data);
        c.present.swap(present);

        // Postings are the one structure permute() doesn't just carry forward under
        // translation: compact() is exactly the point where the lazy stale/dead
        // entries postings() warns about get reclaimed, so rebuild each list fresh
        // against the new numbering rather than remapping the old one.
        if (c.type == AttrType::Tag || c.type == AttrType::Bool) {
            for (auto& p : c.postings) p.clear();
            for (InternalId new_id = 0; new_id < c.present.size(); ++new_id)
                if (c.present[new_id])
                    c.postings[static_cast<uint32_t>(c.data[new_id])].push_back(new_id);
        }

        // Same reason and same fresh-then-repopulate shape as postings just above:
        // the InternalIds an indexed column's B+-tree holds are exactly what
        // renumbers here, so remapping the old tree in place isn't an option —
        // rebuild it against the new numbering (which c.data/c.present, read below,
        // already carry, from the swap above).
        if (c.index) {
            c.index = std::make_unique<BPlusTree>();
            for (InternalId new_id = 0; new_id < c.present.size(); ++new_id)
                if (c.present[new_id]) c.index->insert(sortable_key_(c, new_id), new_id);
        }
    }

    std::vector<std::vector<uint8_t>> payload;
    payload.reserve(live.size());
    for (InternalId src : live) payload.push_back(std::move(payload_[src]));
    payload_.swap(payload);

    rows_ = live.size();
}

void MetadataStore::clear_rows() {
    for (auto& c : columns_) {
        c.data.clear();
        c.present.clear();
        // Dictionaries survive: codes already written into a snapshot must keep
        // meaning the same string. Counts reset to zero but keep dict-sized shape —
        // every code that existed still exists, it just has no live rows right now.
        std::fill(c.count.begin(), c.count.end(), 0u);
        for (auto& p : c.postings) p.clear();
        if (c.index) c.index = std::make_unique<BPlusTree>();
    }
    payload_.clear();
    rows_ = 0;
}

AttrValue MetadataStore::get(InternalId id, size_t attr) const {
    const Column& c = columns_[attr];
    if (!c.present[id]) return attr_null();
    if (c.type == AttrType::Tag) {
        AttrValue v = attr_tag(c.dict[static_cast<size_t>(c.data[id])]);
        v.raw       = c.data[id];  // keep the code alongside the string
        return v;
    }
    return AttrValue{c.type, c.data[id], {}};
}

Record MetadataStore::get_row(InternalId id) const {
    Record rec;
    rec.attrs.reserve(columns_.size());
    for (size_t a = 0; a < columns_.size(); ++a) rec.attrs.push_back(get(id, a));
    rec.payload = payload_[id];
    return rec;
}

ResolvedPredicate MetadataStore::resolve(const Predicate& pred, const std::vector<bool>& deleted) const {
    const Column& c = columns_[pred.attr];
    ResolvedPredicate out;
    out.predicate = pred;

    if (pred.kind == Predicate::Kind::Eq) {
        if (c.type != AttrType::Tag && c.type != AttrType::Bool)
            throw std::invalid_argument("Predicate::Eq on attribute '" + schema_[pred.attr].name +
                                        "': " + type_name(c.type) +
                                        " columns don't support equality predicates"
                                        " (only Tag/Bool)");
        if (pred.eq.type != c.type)
            throw std::invalid_argument("Predicate::Eq value type does not match column '" +
                                        schema_[pred.attr].name + "' (expects " +
                                        type_name(c.type) + ")");

        uint32_t code;
        if (c.type == AttrType::Tag) {
            if (!tag_code(pred.attr, pred.eq.text, code)) {
                out.allowlist = std::vector<InternalId>{};  // value never interned: no matches
                return out;
            }
        } else {
            code = pred.eq.as_bool() ? 1u : 0u;
        }
        // postings() is an append-only superset (dead/pending entries included) —
        // filter against the caller's liveness bitmap to make this exact.
        std::vector<InternalId> ids;
        for (InternalId id : postings(pred.attr, code))
            if (id < deleted.size() && !deleted[id]) ids.push_back(id);
        out.allowlist = std::move(ids);
        return out;
    }

    // Kind::Range
    if (c.type != AttrType::Int64 && c.type != AttrType::Float64)
        throw std::invalid_argument("Predicate::Range on attribute '" + schema_[pred.attr].name +
                                    "': " + type_name(c.type) +
                                    " columns don't support range predicates"
                                    " (only Int64/Float64)");
    if (pred.lo.type != c.type || pred.hi.type != c.type)
        throw std::invalid_argument("Predicate::Range bound type does not match column '" +
                                    schema_[pred.attr].name + "' (expects " + type_name(c.type) +
                                    ")");

    // Not indexed: an exact match set would cost an O(N) scan, so this isn't done
    // eagerly here — allowlist stays nullopt, predicate stays for a future
    // per-candidate check. See ResolvedPredicate's own doc comment.
    if (!c.index) return out;

    std::vector<InternalId> ids;
    // range() is already live-only by construction (see its own doc comment) — no
    // filtering needed here, unlike the Eq/postings path above.
    if (c.type == AttrType::Int64)
        range(pred.attr, pred.lo.as_int(), pred.hi.as_int(), [&](InternalId id) { ids.push_back(id); });
    else
        range(pred.attr, pred.lo.as_double(), pred.hi.as_double(),
              [&](InternalId id) { ids.push_back(id); });
    out.allowlist = std::move(ids);
    return out;
}

const std::vector<uint8_t>& MetadataStore::payload(InternalId id) const {
    return payload_[id];
}

void MetadataStore::serialize(std::vector<uint8_t>& out) const {
    put<uint64_t>(out, fingerprint_);
    put<uint64_t>(out, rows_);
    put<uint64_t>(out, columns_.size());
    for (const auto& c : columns_) {
        put<uint8_t>(out, static_cast<uint8_t>(c.type));
        put<uint64_t>(out, c.data.size());
        for (uint64_t v : c.data) put<uint64_t>(out, v);
        for (bool p : c.present) put<uint8_t>(out, p ? 1 : 0);
        put<uint64_t>(out, c.dict.size());
        for (const auto& s : c.dict) put_string(out, s);
    }
    put<uint64_t>(out, payload_.size());
    for (const auto& p : payload_) put_bytes(out, p);
}

void MetadataStore::deserialize(Reader& r) {
    const uint64_t fp = r.get<uint64_t>();
    if (fp != fingerprint_)
        throw std::runtime_error("snapshot metadata schema mismatch");

    rows_ = static_cast<size_t>(r.get<uint64_t>());

    const uint64_t n_cols = r.get<uint64_t>();
    if (n_cols != columns_.size())
        throw std::runtime_error("snapshot metadata column count mismatch");
    for (auto& c : columns_) {
        c.type = static_cast<AttrType>(r.get<uint8_t>());
        const uint64_t n = r.get<uint64_t>();
        c.data.resize(n);
        for (uint64_t i = 0; i < n; ++i) c.data[i] = r.get<uint64_t>();
        c.present.resize(n);
        for (uint64_t i = 0; i < n; ++i) c.present[i] = r.get<uint8_t>() != 0;

        const uint64_t n_dict = r.get<uint64_t>();
        c.dict.resize(n_dict);
        c.codes.clear();
        c.codes.reserve(n_dict);
        for (uint64_t i = 0; i < n_dict; ++i) {
            c.dict[i] = r.get_string();
            c.codes.emplace(c.dict[i], static_cast<uint32_t>(i));
        }

        // Counts and postings are derived, not serialized: sized here to match the
        // loaded dictionary (Tag) / the fixed false-true pair (Bool), then filled by
        // a rebuild_derived_state() pass once the caller knows which rows are live.
        if (c.type == AttrType::Tag) {
            c.count.assign(c.dict.size(), 0);
            c.postings.assign(c.dict.size(), {});
        } else if (c.type == AttrType::Bool) {
            c.count.assign(2, 0);
            c.postings.assign(2, {});
        } else {
            c.count.clear();
            c.postings.clear();
        }
    }

    const uint64_t n_pay = r.get<uint64_t>();
    payload_.resize(n_pay);
    for (uint64_t i = 0; i < n_pay; ++i) payload_[i] = r.get_bytes();
}

}
