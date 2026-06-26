#include "vdb.h"

#include "brute_index.h"
#include "hnsw_index.h"
#include "ivf_index.h"

#include <utility>

namespace vdb {

namespace {
std::unique_ptr<Index> make_index(const VDBConfig& cfg, DistanceFn dist_fn) {
    switch (cfg.kind) {
        case IndexKind::Brute:
            return std::make_unique<BruteIndex>(cfg.dim, std::move(dist_fn));
        case IndexKind::IVF: {
            IVFConfig ivf; ivf.dim = cfg.dim;
            return std::make_unique<IVFIndex>(ivf, std::move(dist_fn));
        }
        case IndexKind::HNSW: {
            HNSWConfig hnsw; hnsw.dim = cfg.dim;
            return std::make_unique<HNSWIndex>(hnsw, std::move(dist_fn));
        }
    }
    return nullptr;
}
}  // namespace

VDB::VDB(VDBConfig cfg) : config_(cfg) {
    index_ = make_index(config_, metric_fn(config_.metric));
}

ExternalId VDB::insert(const float* vec) {
    // TODO Stage 5: allocate an ExternalId, map it to the InternalId from add().
    return static_cast<ExternalId>(index_->add(vec));
}

std::vector<ExternalId> VDB::search(const float* query, size_t K) const {
    // TODO Stage 5: translate InternalIds back to ExternalIds, skip tombstones.
    std::vector<ExternalId> out;
    for (auto& [id, dist] : index_->search(query, K)) {
        (void)dist;
        out.push_back(static_cast<ExternalId>(id));
    }
    return out;
}

size_t VDB::size() const { return index_->size(); }
size_t VDB::dim() const { return index_->dim(); }

}  // namespace vdb
