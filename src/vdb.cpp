#include "vdb.h"

#include "brute_index.h"
#include "hnsw_index.h"
#include "ivf_index.h"

#include <algorithm>
#include <cassert>
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
}

VDB::VDB(VDBConfig cfg) : config_(cfg) {
    index_ = make_index(config_, metric_fn(config_.metric));
}

InternalId VDB::append_(ExternalId ext, const float* vec) {
    const InternalId iid = index_->add(vec);
    // The parallel arrays assume the index numbers nodes sequentially, matching
    // our own append order. Anything else would desync id translation.
    assert(iid == int_to_ext_.size());
    (void)iid;
    int_to_ext_.push_back(ext);
    deleted_.push_back(false);
    vectors_.emplace_back(vec, vec + config_.dim);
    return static_cast<InternalId>(int_to_ext_.size() - 1);
}

ExternalId VDB::insert(const float* vec) {
    const ExternalId ext = next_ext_id_++;
    const InternalId iid = append_(ext, vec);
    ext_to_int_[ext] = iid;
    ++live_count_;
    return ext;
}

bool VDB::remove(ExternalId id) {
    auto it = ext_to_int_.find(id);
    if (it == ext_to_int_.end()) return false;
    deleted_[it->second] = true;   // tombstone; node stays in the graph
    ext_to_int_.erase(it);
    --live_count_;
    ++deleted_count_;
    return true;
}

bool VDB::update(ExternalId id, const float* vec) {
    auto it = ext_to_int_.find(id);
    if (it == ext_to_int_.end()) return false;
    // Tombstone the old node, insert a new one, and repoint the same external id.
    // Live count is unchanged (one out, one in); one more tombstone is created.
    deleted_[it->second] = true;
    ++deleted_count_;
    it->second = append_(id, vec);
    return true;
}

bool VDB::contains(ExternalId id) const {
    return ext_to_int_.find(id) != ext_to_int_.end();
}

std::vector<ExternalId> VDB::search(const float* query, size_t K) const {
    if (K == 0) return {};

    // Tombstoned hits are dropped after the index returns them, so ask for enough
    // extra to still land K live results. K + deleted_count_ is exact for the
    // brute oracle (worst case: every tombstone ranks ahead of the live top-K)
    // and a safe over-fetch for the ANN indexes. Unbounded tombstone growth is
    // the pressure that motivates compact().
    const size_t want = std::min(K + deleted_count_, index_->size());

    std::vector<ExternalId> out;
    out.reserve(K);
    for (auto& [iid, dist] : index_->search(query, want)) {
        (void)dist;
        if (deleted_[iid]) continue;
        out.push_back(int_to_ext_[iid]);
        if (out.size() == K) break;
    }
    return out;
}

void VDB::compact() {
    // Collect live vectors in internal-id order (deterministic rebuild).
    std::vector<std::vector<float>> live_vecs;
    std::vector<ExternalId>         live_exts;
    live_vecs.reserve(live_count_);
    live_exts.reserve(live_count_);
    for (InternalId i = 0; i < int_to_ext_.size(); ++i) {
        if (deleted_[i]) continue;
        live_vecs.push_back(std::move(vectors_[i]));
        live_exts.push_back(int_to_ext_[i]);
    }

    // Fresh index and identity maps; external ids carry over unchanged.
    index_ = make_index(config_, metric_fn(config_.metric));
    ext_to_int_.clear();
    int_to_ext_.clear();
    deleted_.clear();
    vectors_.clear();
    deleted_count_ = 0;

    // IVF needs a trained coarse quantizer before add(); feed it the live set as
    // a contiguous buffer. Brute/HNSW inherit a no-op train().
    if (config_.kind == IndexKind::IVF && !live_vecs.empty()) {
        std::vector<float> flat;
        flat.reserve(live_vecs.size() * config_.dim);
        for (const auto& v : live_vecs)
            flat.insert(flat.end(), v.begin(), v.end());
        index_->train(flat.data(), live_vecs.size());
    }

    for (size_t i = 0; i < live_vecs.size(); ++i) {
        const InternalId iid = append_(live_exts[i], live_vecs[i].data());
        ext_to_int_[live_exts[i]] = iid;
    }
    // live_count_ and next_ext_id_ are unchanged by compaction.
}

}
