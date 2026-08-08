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

// The pre-filter path's brute-force scan: same nth_element-then-sort shape as
// BruteIndex<Dist>::search, just over `allowlist` instead of every row. `Dist` is
// picked at compile time (this function is only ever instantiated with L2/
// InnerProduct/Cosine below), so the distance loop inlines exactly as it does inside
// BruteIndex itself.
template <class Dist>
std::vector<std::pair<InternalId, float>> prefilter_scan_impl(
    const float* query, size_t K, size_t dim, const std::vector<InternalId>& allowlist,
    const std::vector<std::vector<float>>& vectors) {
    using Entry = std::pair<InternalId, float>;
    Dist dist_;

    std::vector<Entry> all;
    all.reserve(allowlist.size());
    for (InternalId iid : allowlist) all.push_back({iid, dist_(query, vectors[iid].data(), dim)});

    auto by_distance = [](const Entry& a, const Entry& b) { return a.second < b.second; };
    const size_t k = std::min(K, all.size());
    std::nth_element(all.begin(), all.begin() + k, all.end(), by_distance);
    all.resize(k);
    std::sort(all.begin(), all.end(), by_distance);
    return all;
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

VDB::VDB(VDBConfig cfg) : config_(cfg), meta_(cfg.schema) {
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
    insert_reserved(ext, vec, Record{});
}

void VDB::insert_reserved(ExternalId ext, const float* vec, const Record& meta) {
    // Validate before touching anything: a schema violation must not leave the
    // parallel arrays half-appended (the index slot would already be allocated).
    meta_.validate(meta);

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
        meta_.append_row(meta);     // stays in step with the arrays above
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
        meta_.mark_live(iid);  // now visible to search: count it
        ext_to_int_[ext] = iid;
        ++live_count_;
    }
}

ExternalId VDB::insert(const float* vec) {
    return insert(vec, Record{});
}

ExternalId VDB::insert(const float* vec, const Record& meta) {
    meta_.validate(meta);  // fail before minting an id we would then have to waste
    const ExternalId ext = reserve_id();
    insert_reserved(ext, vec, meta);
    return ext;
}

bool VDB::remove(ExternalId id) {
    std::unique_lock<std::shared_mutex> lk(mu_);
    auto it = ext_to_int_.find(id);
    if (it == ext_to_int_.end()) return false;
    deleted_[it->second] = true;   // tombstone; node stays in the graph
    meta_.mark_dead(it->second);   // no longer visible to search: stop counting it
    ext_to_int_.erase(it);
    --live_count_;
    ++deleted_count_;
    return true;
}

bool VDB::update(ExternalId id, const float* vec) {
    return update_(id, vec, nullptr);
}

bool VDB::update(ExternalId id, const float* vec, const Record& meta) {
    return update_(id, vec, &meta);
}

// `meta == nullptr` means "carry the existing row forward onto the replacement node",
// which is what a vector-only update wants: the attributes did not change.
bool VDB::update_(ExternalId id, const float* vec, const Record* meta) {
    if (meta) meta_.validate(*meta);

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
        if (meta) meta_.append_row(*meta);
        else      meta_.append_row_copy(old_iid);
    }
    // Phase 2 (no lock): link the replacement into the graph.
    index_->link(new_iid);
    // Phase 3: swap old→new atomically under one exclusive hold, so a concurrent
    // reader sees the old vector or the new one, never neither. Live count is
    // unchanged (one out, one in); a tombstone is left behind for the old node.
    {
        std::unique_lock<std::shared_mutex> lk(mu_);
        deleted_[new_iid] = false; --deleted_count_;   // new goes live
        meta_.mark_live(new_iid);
        deleted_[old_iid] = true;  ++deleted_count_;   // old becomes a tombstone
        meta_.mark_dead(old_iid);
        ext_to_int_[id] = new_iid;
    }
    return true;
}

bool VDB::contains(ExternalId id) const {
    std::shared_lock<std::shared_mutex> lk(mu_);
    return ext_to_int_.find(id) != ext_to_int_.end();
}

// Metadata-only update. Note what is *absent*: no index_->allocate, no link, no
// tombstone. Rewriting a row is a handful of stores into the columns, because the
// metadata lives beside the graph rather than inside it.
bool VDB::set_metadata(ExternalId id, const Record& meta) {
    meta_.validate(meta);
    std::unique_lock<std::shared_mutex> lk(mu_);
    auto it = ext_to_int_.find(id);
    if (it == ext_to_int_.end()) return false;
    meta_.set_row(it->second, meta);
    return true;
}

bool VDB::get_metadata(ExternalId id, Record& out) const {
    std::shared_lock<std::shared_mutex> lk(mu_);
    auto it = ext_to_int_.find(id);
    if (it == ext_to_int_.end()) return false;
    out = meta_.get_row(it->second);
    return true;
}

// Caller holds mu_. Templated on the sink so the emit call inlines — the same
// reasoning as the distance functors: this loop must not pay an indirect call.
template <class Emit>
void VDB::collect_(const float* query, size_t K, Emit&& emit, const ResolvedPredicate* pred) const {
    // Tombstoned hits (and, with a predicate, non-matches) are dropped after the
    // index returns them, so ask for enough extra to still land K live results.
    // Unbounded tombstone growth is the pressure that motivates compact().
    //
    // This over-fetch is a *post-filter* margin, and it only stays bounded because
    // the predicate is exact and resolved up front. A predicate matching fraction s
    // of the db needs K + N(1-s) — with no predicate that's K + deleted_count_
    // (s = live fraction); with one, `pred->allowlist` (already live-only) gives the
    // exact match count, so K + (index_->size() - allowlist->size()) is exact too.
    std::vector<bool> in_allowlist;
    size_t want;
    if (pred) {
        if (!pred->allowlist)
            throw std::invalid_argument(
                "collect_: predicate did not resolve to an allowlist (range on a "
                "non-indexed column) — per-candidate evaluation isn't supported yet");
        want = std::min(K + (index_->size() - pred->allowlist->size()), index_->size());
        in_allowlist.assign(index_->size(), false);
        for (InternalId id : *pred->allowlist) in_allowlist[id] = true;
    } else {
        want = std::min(K + deleted_count_, index_->size());
    }

    size_t taken = 0;
    for (auto& [iid, dist] : index_->search(query, want)) {
        if (deleted_[iid]) continue;
        if (pred && !in_allowlist[iid]) continue;
        emit(iid, dist);
        if (++taken == K) break;
    }
}

std::vector<ExternalId> VDB::search(const float* query, size_t K) const {
    if (K == 0) return {};

    // Shared lock held across the whole query: the index call is thread-safe on its
    // own, but the results loop reads deleted_/int_to_ext_, which a writer's publish
    // could grow. Holding shared throughout is the simple, correct choice; it does
    // block writers' brief exclusive phases for the search duration — a later step
    // can release mu_ across index_->search() and re-take it only for the loop.
    std::shared_lock<std::shared_mutex> lk(mu_);

    std::vector<ExternalId> out;
    out.reserve(K);
    collect_(query, K, [&](InternalId iid, float) { out.push_back(int_to_ext_[iid]); });
    return out;
}

std::vector<Hit> VDB::search_hits(const float* query, size_t K) const {
    if (K == 0) return {};
    std::shared_lock<std::shared_mutex> lk(mu_);

    std::vector<Hit> out;
    out.reserve(K);
    // The payload copy happens here, under the lock, and only for the K survivors.
    collect_(query, K, [&](InternalId iid, float dist) {
        out.push_back(Hit{int_to_ext_[iid], dist, meta_.payload(iid)});
    });
    return out;
}

std::vector<ExternalId> VDB::search(const float* query, size_t K, const Predicate& pred) const {
    if (K == 0) return {};
    std::shared_lock<std::shared_mutex> lk(mu_);

    ResolvedPredicate resolved = meta_.resolve(pred, deleted_);
    std::vector<ExternalId> out;
    out.reserve(K);
    collect_(
        query, K, [&](InternalId iid, float) { out.push_back(int_to_ext_[iid]); }, &resolved);
    return out;
}

std::vector<Hit> VDB::search_hits(const float* query, size_t K, const Predicate& pred) const {
    if (K == 0) return {};
    std::shared_lock<std::shared_mutex> lk(mu_);

    ResolvedPredicate resolved = meta_.resolve(pred, deleted_);
    std::vector<Hit> out;
    out.reserve(K);
    collect_(
        query, K,
        [&](InternalId iid, float dist) { out.push_back(Hit{int_to_ext_[iid], dist, meta_.payload(iid)}); },
        &resolved);
    return out;
}

std::vector<std::pair<InternalId, float>> VDB::prefilter_scan_(
    const float* query, size_t K, const std::vector<InternalId>& allowlist) const {
    switch (config_.metric) {
        case Metric::L2:
            return prefilter_scan_impl<L2>(query, K, config_.dim, allowlist, vectors_);
        case Metric::InnerProduct:
            return prefilter_scan_impl<InnerProduct>(query, K, config_.dim, allowlist, vectors_);
        case Metric::Cosine:
            return prefilter_scan_impl<Cosine>(query, K, config_.dim, allowlist, vectors_);
    }
    return {};
}

template <class Emit>
void VDB::collect_prefiltered_(const float* query, size_t K, const Predicate& pred, Emit&& emit) const {
    ResolvedPredicate resolved = meta_.resolve(pred, deleted_);
    if (!resolved.allowlist)
        throw std::invalid_argument(
            "collect_prefiltered_: predicate did not resolve to an allowlist (range on a "
            "non-indexed column) — pre-filter needs a materialized match set");

    for (auto& [iid, dist] : prefilter_scan_(query, K, *resolved.allowlist)) emit(iid, dist);
}

std::vector<ExternalId> VDB::search_prefiltered(const float* query, size_t K, const Predicate& pred) const {
    if (K == 0) return {};
    std::shared_lock<std::shared_mutex> lk(mu_);

    std::vector<ExternalId> out;
    out.reserve(K);
    collect_prefiltered_(query, K, pred, [&](InternalId iid, float) { out.push_back(int_to_ext_[iid]); });
    return out;
}

std::vector<Hit> VDB::search_hits_prefiltered(const float* query, size_t K, const Predicate& pred) const {
    if (K == 0) return {};
    std::shared_lock<std::shared_mutex> lk(mu_);

    std::vector<Hit> out;
    out.reserve(K);
    collect_prefiltered_(query, K, pred, [&](InternalId iid, float dist) {
        out.push_back(Hit{int_to_ext_[iid], dist, meta_.payload(iid)});
    });
    return out;
}

void VDB::compact() {
    std::unique_lock<std::shared_mutex> lk(mu_);  // stop-the-world for the rebuild
    // Collect live vectors in internal-id order (deterministic rebuild).
    std::vector<std::vector<float>> live_vecs;
    std::vector<ExternalId>         live_exts;
    std::vector<InternalId>         live_ids;
    live_vecs.reserve(live_count_);
    live_exts.reserve(live_count_);
    live_ids.reserve(live_count_);
    for (InternalId i = 0; i < int_to_ext_.size(); ++i) {
        if (deleted_[i]) continue;
        live_vecs.push_back(std::move(vectors_[i]));
        live_exts.push_back(int_to_ext_[i]);
        live_ids.push_back(i);
    }

    // The cost of keying metadata by InternalId: renumbering the nodes renumbers the
    // rows. Because the rebuild below re-adds the live vectors in exactly this order,
    // new internal id i holds old row live_ids[i] — one out-of-place permutation.
    meta_.permute(live_ids);

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
