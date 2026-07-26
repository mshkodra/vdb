#include "vdb.h"

#include "brute_index.h"
#include "distance.h"
#include "hnsw_index.h"
#include "ivf_index.h"

#include <algorithm>
#include <cassert>
#include <utility>

namespace vdb {

namespace {
// Second half of the dispatch: given a chosen index template (Brute/IVF/HNSW), pick
// the metric functor at runtime and build the matching instantiation. The metric is
// erased behind the virtual Index* the moment we return, so this is the only place
// the concrete IndexT<Dist> type is ever named — exactly what lets each index inline
// its distance loop while callers still hold a plain Index*.
template <template <class> class IndexT, class Arg>
std::unique_ptr<Index> make_for_metric(Metric m, Arg&& arg) {
    switch (m) {
        case Metric::L2:
            return std::make_unique<IndexT<L2>>(std::forward<Arg>(arg));
        case Metric::InnerProduct:
            return std::make_unique<IndexT<InnerProduct>>(std::forward<Arg>(arg));
        case Metric::Cosine:
            return std::make_unique<IndexT<Cosine>>(std::forward<Arg>(arg));
    }
    return nullptr;
}
}  // namespace

std::unique_ptr<Index> VDB::make_index_(const VDBConfig& cfg) {
    switch (cfg.kind) {
        case IndexKind::Brute:
            return make_for_metric<BruteIndex>(cfg.metric, cfg.dim);
        case IndexKind::IVF: {
            IVFConfig ivf; ivf.dim = cfg.dim;
            return make_for_metric<IVFIndex>(cfg.metric, ivf);
        }
        case IndexKind::HNSW: {
            HNSWConfig hnsw; hnsw.dim = cfg.dim;
            return make_for_metric<HNSWIndex>(cfg.metric, hnsw);
        }
    }
    return nullptr;
}

VDB::VDB(VDBConfig cfg) : config_(cfg) {
    index_ = make_index_(config_);
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

ExternalId VDB::reserve_id() {
    std::unique_lock<std::shared_mutex> lk(mu_);
    return next_ext_id_++;
}

void VDB::insert_reserved(ExternalId ext, const float* vec) {
    InternalId iid;
    // Phase 1 (exclusive, brief): reserve the node, size the arrays. The node is
    // marked deleted_=true so it is invisible while it is being linked.
    {
        std::unique_lock<std::shared_mutex> lk(mu_);
        iid = index_->allocate(vec);
        assert(iid == int_to_ext_.size());
        int_to_ext_.push_back(ext);
        deleted_.push_back(true);   // pending: invisible until publish
        ++deleted_count_;
        vectors_.emplace_back(vec, vec + config_.dim);
        if (ext >= next_ext_id_) next_ext_id_ = ext + 1;  // keep ahead (replay path)
    }
    // Phase 2 (no lock): wire into the graph. Runs concurrently with other inserts'
    // links and with searches, guarded by the index's own per-node locks.
    index_->link(iid);
    // Phase 3 (exclusive, brief): publish — the node becomes live and reachable.
    {
        std::unique_lock<std::shared_mutex> lk(mu_);
        deleted_[iid] = false;
        --deleted_count_;
        ext_to_int_[ext] = iid;
        ++live_count_;
    }
}

ExternalId VDB::insert(const float* vec) {
    const ExternalId ext = reserve_id();
    insert_reserved(ext, vec);
    return ext;
}

bool VDB::remove(ExternalId id) {
    std::unique_lock<std::shared_mutex> lk(mu_);
    auto it = ext_to_int_.find(id);
    if (it == ext_to_int_.end()) return false;
    deleted_[it->second] = true;   // tombstone; node stays in the graph
    ext_to_int_.erase(it);
    --live_count_;
    ++deleted_count_;
    return true;
}

bool VDB::update(ExternalId id, const float* vec) {
    InternalId new_iid, old_iid;
    // Phase 1: allocate the replacement (pending). The old node stays live for now.
    {
        std::unique_lock<std::shared_mutex> lk(mu_);
        auto it = ext_to_int_.find(id);
        if (it == ext_to_int_.end()) return false;
        old_iid = it->second;
        new_iid = index_->allocate(vec);
        assert(new_iid == int_to_ext_.size());
        int_to_ext_.push_back(id);
        deleted_.push_back(true);   // new node pending
        ++deleted_count_;
        vectors_.emplace_back(vec, vec + config_.dim);
    }
    // Phase 2 (no lock): link the replacement into the graph.
    index_->link(new_iid);
    // Phase 3: swap old→new atomically under one exclusive hold, so a concurrent
    // reader sees the old vector or the new one, never neither. Live count is
    // unchanged (one out, one in); a tombstone is left behind for the old node.
    {
        std::unique_lock<std::shared_mutex> lk(mu_);
        deleted_[new_iid] = false; --deleted_count_;   // new goes live
        deleted_[old_iid] = true;  ++deleted_count_;   // old becomes a tombstone
        ext_to_int_[id] = new_iid;
    }
    return true;
}

bool VDB::contains(ExternalId id) const {
    std::shared_lock<std::shared_mutex> lk(mu_);
    return ext_to_int_.find(id) != ext_to_int_.end();
}

std::vector<ExternalId> VDB::search(const float* query, size_t K) const {
    if (K == 0) return {};

    // Shared lock held across the whole query: the index call is thread-safe on its
    // own, but the results loop reads deleted_/int_to_ext_, which a writer's publish
    // could grow. Holding shared throughout is the simple, correct choice; it does
    // block writers' brief exclusive phases for the search duration — a later step
    // can release mu_ across index_->search() and re-take it only for the loop.
    std::shared_lock<std::shared_mutex> lk(mu_);

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
    std::unique_lock<std::shared_mutex> lk(mu_);  // stop-the-world for the rebuild
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
    index_ = make_index_(config_);
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
