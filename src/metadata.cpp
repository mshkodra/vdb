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
        columns_[a].type = schema_[a].type;
        // Bool's two codes (false/true) are known up front, unlike Tag's dictionary,
        // which grows as strings are interned.
        if (columns_[a].type == AttrType::Bool) columns_[a].count.assign(2, 0);
    }
    compute_fingerprint_();
}

// FNV-1a over the declared (name, type) pairs. Order-sensitive on purpose: the
// columns are positional, so reordering the schema really is a different layout.
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
    c.count.push_back(0);  // a brand-new code starts with zero live rows
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

void MetadataStore::mark_live(InternalId id) {
    for (size_t a = 0; a < columns_.size(); ++a) adjust_count_(a, id, /*increment=*/true);
}

void MetadataStore::mark_dead(InternalId id) {
    for (size_t a = 0; a < columns_.size(); ++a) adjust_count_(a, id, /*increment=*/false);
}

void MetadataStore::rebuild_counts(const std::vector<bool>& deleted) {
    for (auto& c : columns_)
        if (c.type == AttrType::Tag || c.type == AttrType::Bool)
            std::fill(c.count.begin(), c.count.end(), 0u);

    for (InternalId id = 0; id < rows_; ++id) {
        if (deleted[id]) continue;
        for (size_t a = 0; a < columns_.size(); ++a) adjust_count_(a, id, /*increment=*/true);
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
}

void MetadataStore::append_row_copy(InternalId src) {
    for (auto& c : columns_) {
        c.data.push_back(c.data[src]);
        c.present.push_back(c.present[src]);
    }
    payload_.push_back(payload_[src]);
    ++rows_;
}

void MetadataStore::set_row(InternalId id, const Record& rec) {
    validate(rec);
    // A row reached here is always live (set_metadata resolves it through
    // ext_to_int_, which only holds published rows), so the value being replaced was
    // already counted — drop it before overwriting, add the new value back after.
    for (size_t a = 0; a < columns_.size(); ++a) {
        adjust_count_(a, id, /*increment=*/false);
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

        // Counts are derived, not serialized: sized here to match the loaded
        // dictionary (Tag) / the fixed false-true pair (Bool), then filled by a
        // rebuild_counts() pass once the caller knows which rows are live.
        if (c.type == AttrType::Tag) c.count.assign(c.dict.size(), 0);
        else if (c.type == AttrType::Bool) c.count.assign(2, 0);
        else c.count.clear();
    }

    const uint64_t n_pay = r.get<uint64_t>();
    payload_.resize(n_pay);
    for (uint64_t i = 0; i < n_pay; ++i) payload_[i] = r.get_bytes();
}

}
