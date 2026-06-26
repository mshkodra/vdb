#include <vdb.h>

#include <snapshot.h>

#include <filesystem>
#include <mutex>
#include <shared_mutex>
#include <utility>

namespace fs = std::filesystem;

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

    if (!config_.data_dir.empty()) {
        durable_ = true;
        recover();
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

// ---------------------------------------------------------------------------
// In-memory state transitions. These are the *only* code that mutates the index
// and the id maps; both the live write path and WAL replay funnel through them,
// which is what guarantees "replay reproduces exactly what was logged".
// ---------------------------------------------------------------------------

void VDB::do_insert(ExternalId ext, const float* vec) {
    const InternalId internal = hnsw_->insert(vec);
    bind(internal, ext);
    // Keep the minter ahead of every id we have ever seen (crucial during replay,
    // where ids arrive from the log rather than being freshly minted here).
    if (ext >= next_ext_id_) next_ext_id_ = ext + 1;
}

void VDB::do_remove(ExternalId ext) {
    const auto it = ext_to_int_.find(ext);
    if (it == ext_to_int_.end()) return;  // idempotent: replaying a delete twice is safe
    hnsw_->mark_deleted(it->second);
    ext_to_int_.erase(it);
}

void VDB::do_update(ExternalId ext, const float* vec) {
    const auto it = ext_to_int_.find(ext);
    if (it == ext_to_int_.end()) return;
    hnsw_->mark_deleted(it->second);             // tombstone the old slot
    const InternalId internal = hnsw_->insert(vec);
    bind(internal, ext);                         // rebind same ext -> new slot
}

void VDB::apply(const WalRecord& rec) {
    switch (rec.type) {
        case WalRecordType::Insert: do_insert(rec.ext_id, rec.vec.data()); break;
        case WalRecordType::Update: do_update(rec.ext_id, rec.vec.data()); break;
        case WalRecordType::Delete: do_remove(rec.ext_id); break;
        case WalRecordType::Checkpoint: break;  // breadcrumb only; no state change
    }
    if (rec.lsn > current_lsn_) current_lsn_ = rec.lsn;
}

// ---------------------------------------------------------------------------
// Durability: log BEFORE apply.
// ---------------------------------------------------------------------------

void VDB::log(WalRecordType type, ExternalId ext, const float* vec) {
    if (!durable_) return;

    WalRecord rec;
    rec.type   = type;
    rec.lsn    = current_lsn_ + 1;   // monotonic; only advanced once durable
    rec.ext_id = ext;
    if (type == WalRecordType::Insert || type == WalRecordType::Update) {
        rec.vec.assign(vec, vec + hnsw_->dim());
    }

    wal_->append(rec);               // append + fsync per policy = the durability point
    current_lsn_ = rec.lsn;          // advance only after the bytes are (per policy) safe
}

void VDB::recover() {
    fs::create_directories(config_.data_dir);

    // 1. Load the newest valid snapshot, if any, into the index + id maps.
    uint64_t snap_lsn = 0, snap_next = 1, snap_id = 0;
    std::vector<ExternalId> i2e;
    const bool have_snap = snapshot::load_latest(
        config_.data_dir, static_cast<uint32_t>(config_.hnsw.dim), config_.metric,
        snap_lsn, snap_next, i2e, *hnsw_, snap_id);

    if (have_snap) {
        int_to_ext_   = std::move(i2e);
        next_ext_id_  = snap_next;
        current_lsn_  = snap_lsn;
        snapshot_seq_ = snap_id;
        // Reconstruct the forward map: a slot is a live key iff it is not tombstoned.
        // (After an update, the old slot is deleted and the new one — sharing the same
        // ExternalId — is live, so this picks the live slot for each key.)
        ext_to_int_.clear();
        ext_to_int_.reserve(int_to_ext_.size());
        for (InternalId i = 0; i < int_to_ext_.size(); ++i) {
            if (!hnsw_->is_deleted(i)) ext_to_int_[int_to_ext_[i]] = i;
        }
    }

    // 2. Open the WAL and replay everything that postdates the snapshot. replay()
    //    stops at (and truncates) the first torn/corrupt record — the crash tail.
    const std::string wal_path = (fs::path(config_.data_dir) / "wal.log").string();
    wal_ = std::make_unique<Wal>();
    wal_->set_fsync_interval_ms(config_.fsync_interval_ms);
    wal_->open(wal_path, static_cast<uint32_t>(config_.hnsw.dim), config_.metric,
               config_.durability);

    wal_->replay([this, snap_lsn](const WalRecord& rec) {
        if (rec.lsn <= snap_lsn) return;  // already folded into the snapshot
        apply(rec);
    });
}

// ---------------------------------------------------------------------------
// Public mutating API: mint/validate, then log-before-apply.
// ---------------------------------------------------------------------------

ExternalId VDB::insert(const float* vec) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    const ExternalId ext = next_ext_id_;  // peek; do_insert advances the minter
    log(WalRecordType::Insert, ext, vec);
    do_insert(ext, vec);
    maybe_auto_checkpoint();
    return ext;
}

bool VDB::remove(ExternalId id) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    if (ext_to_int_.find(id) == ext_to_int_.end()) return false;  // unknown → don't log
    log(WalRecordType::Delete, id, nullptr);
    do_remove(id);
    maybe_auto_checkpoint();
    return true;
}

bool VDB::update(ExternalId id, const float* vec) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    if (ext_to_int_.find(id) == ext_to_int_.end()) return false;
    log(WalRecordType::Update, id, vec);
    do_update(id, vec);
    maybe_auto_checkpoint();
    return true;
}

std::vector<ExternalId> VDB::search(const float* query, size_t K, size_t ef) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    // The index already excludes tombstoned nodes from results, so every InternalId
    // it returns has a valid, live mapping in int_to_ext_.
    const auto internal = hnsw_->search(query, K, ef);

    std::vector<ExternalId> out;
    out.reserve(internal.size());
    for (InternalId iid : internal) out.push_back(int_to_ext_[iid]);
    return out;
}

const float* VDB::get(ExternalId id) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    const auto it = ext_to_int_.find(id);
    if (it == ext_to_int_.end()) return nullptr;  // absent or already deleted
    return hnsw_->get(it->second);
}

void VDB::checkpoint() {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    do_checkpoint();
}

void VDB::do_checkpoint() {
    if (!durable_) return;

    snapshot_seq_++;
    snapshot::save(config_.data_dir, snapshot_seq_,
                   static_cast<uint32_t>(config_.hnsw.dim), config_.metric,
                   current_lsn_, next_ext_id_, int_to_ext_, *hnsw_);

    // The snapshot now durably holds every change up to current_lsn_, so the WAL's
    // existing records are redundant. Rotate it: discard them and drop a CHECKPOINT
    // breadcrumb as the new first record.
    WalRecord cp;
    cp.type        = WalRecordType::Checkpoint;
    cp.lsn         = ++current_lsn_;
    cp.snapshot_id = snapshot_seq_;
    wal_->rotate(cp);

    ops_since_checkpoint_ = 0;
}

void VDB::maybe_auto_checkpoint() {
    // Called from inside a mutation that already holds mutex_ exclusively, so this
    // invokes do_checkpoint() directly rather than the locking checkpoint().
    if (!durable_ || config_.checkpoint_threshold_ops == 0) return;
    if (++ops_since_checkpoint_ >= config_.checkpoint_threshold_ops) do_checkpoint();
}

bool VDB::contains(ExternalId id) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return ext_to_int_.count(id) != 0;
}

size_t VDB::size() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return ext_to_int_.size();
}

size_t VDB::dim() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return hnsw_->dim();
}
