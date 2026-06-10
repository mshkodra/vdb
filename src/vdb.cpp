#include <vdb.h>

#include <stdexcept>
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

InternalId VDB::insert(const float* vec) {
    return hnsw_->insert(vec);
}

std::vector<InternalId> VDB::search(const float* query, size_t K, size_t ef) const {
    return hnsw_->search(query, K, ef);
}

const float* VDB::get(InternalId id) const {
    return hnsw_->get(id);
}

size_t VDB::size() const { return hnsw_->size(); }
size_t VDB::dim()  const { return hnsw_->dim(); }
