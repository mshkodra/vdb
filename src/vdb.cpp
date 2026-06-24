#include <vdb.h>

#include <utility>

namespace {

float l2_squared(const float* a, const float* b, size_t d) {
    float s = 0.0f;
    for (size_t i = 0; i < d; ++i) {
        const float diff = a[i] - b[i];
        s += diff * diff;
    }
    return s;
}

}  // namespace

VDB::VDB(VDBConfig cfg) : config_(std::move(cfg)) {
    if (!config_.dist_fn) config_.dist_fn = &l2_squared;

    switch (config_.type) {
        case IndexType::HNSW:
            hnsw_ = std::make_unique<HNSWIndex>(config_.hnsw, config_.dist_fn);
            break;
    }
}

// Record the internal<->external relationship for a freshly created internal node.
// InternalIds are handed out densely and monotonically by the index, so int_to_ext_
// grows by exactly one slot per call.
void VDB::bind(InternalId internal, ExternalId ext) {
    if (internal >= int_to_ext_.size()) int_to_ext_.resize(internal + 1);
    int_to_ext_[internal] = ext;
    ext_to_int_[ext]      = internal;
}

ExternalId VDB::insert(const float* vec) {
    const InternalId internal = hnsw_->insert(vec);
    const ExternalId ext      = next_ext_id_++;
    bind(internal, ext);
    return ext;
}

std::vector<ExternalId> VDB::search(const float* query, size_t K, size_t ef) const {
    // The index already excludes tombstoned nodes from results, so every InternalId
    // it returns has a valid, live mapping in int_to_ext_.
    const auto internal = hnsw_->search(query, K, ef);

    std::vector<ExternalId> out;
    out.reserve(internal.size());
    for (InternalId iid : internal) out.push_back(int_to_ext_[iid]);
    return out;
}

const float* VDB::get(ExternalId id) const {
    const auto it = ext_to_int_.find(id);
    if (it == ext_to_int_.end()) return nullptr;  // absent or already deleted
    return hnsw_->get(it->second);
}

bool VDB::remove(ExternalId id) {
    const auto it = ext_to_int_.find(id);
    if (it == ext_to_int_.end()) return false;

    // Tombstone the graph node (kept for connectivity) and drop the key so the
    // database reports it as gone immediately. We deliberately leave int_to_ext_
    // intact: the node may still appear in a traversal until it is filtered out.
    hnsw_->mark_deleted(it->second);
    ext_to_int_.erase(it);
    return true;
}

bool VDB::update(ExternalId id, const float* vec) {
    const auto it = ext_to_int_.find(id);
    if (it == ext_to_int_.end()) return false;

    // delete-old + insert-new, rebinding the same ExternalId to the new internal
    // node. Logged as a single atomic op once the WAL exists (Phase 2).
    hnsw_->mark_deleted(it->second);
    const InternalId new_internal = hnsw_->insert(vec);
    bind(new_internal, id);  // overwrites it->second and refreshes int_to_ext_
    return true;
}

bool   VDB::contains(ExternalId id) const { return ext_to_int_.count(id) != 0; }
size_t VDB::size() const { return ext_to_int_.size(); }
size_t VDB::dim()  const { return hnsw_->dim(); }
