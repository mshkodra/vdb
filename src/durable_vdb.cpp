#include "durable_vdb.h"

#include "snapshot.h"

#include <algorithm>
#include <filesystem>

namespace vdb {

DurableVDB::DurableVDB(VDBConfig cfg, const std::string& data_dir, Options opts)
    : config_(cfg),
      data_dir_(data_dir),
      wal_dir_(data_dir + "/wal"),
      snapshot_path_(data_dir + "/snapshot"),
      opts_(opts),
      db_(cfg) {
    std::filesystem::create_directories(data_dir_);
    recover_();
    last_sync_ = std::chrono::steady_clock::now();
}

DurableVDB::~DurableVDB() {
    try {
        wal_.sync();  // best-effort clean shutdown
    } catch (...) {
    }
}

void DurableVDB::recover_() {
    uint64_t snapshot_lsn = 0;
    if (std::filesystem::exists(snapshot_path_))
        snapshot_lsn = load_snapshot(db_, snapshot_path_);

    wal_.open(wal_dir_, static_cast<uint32_t>(config_.dim),
              static_cast<uint8_t>(config_.metric), db_.schema_fingerprint(), opts_.wal);
    wal_.replay([&](const WalRecord& r) {
        if (r.lsn > snapshot_lsn) apply_(r);  // skip records already in the snapshot
    });

    next_lsn_    = std::max(snapshot_lsn, wal_.max_lsn()) + 1;
    durable_lsn_ = next_lsn_ - 1;  // everything recovered from disk is durable
}

void DurableVDB::apply_(const WalRecord& r) {
    switch (r.type) {
        case WalRecordType::Insert:
            // Replay uses the logged id directly, so the mapping is order-independent.
            db_.insert_reserved(r.ext_id, r.vec.data(), r.meta);
            break;
        case WalRecordType::Update:
            db_.update(r.ext_id, r.vec.data(), r.meta);
            break;
        case WalRecordType::SetMeta:
            db_.set_metadata(r.ext_id, r.meta);
            break;
        case WalRecordType::Delete:
            db_.remove(r.ext_id);
            break;
        case WalRecordType::Checkpoint:
            break;
    }
}

// Group commit: wait until `lsn` is on stable storage, batching the fsync across
// concurrent writers. One thread becomes the leader and fsyncs once for everyone;
// followers wait and wake when the leader has advanced durable_lsn_ past their lsn.
// The fsync runs with commit_mutex_ released, so appends keep flowing into the batch.
void DurableVDB::group_commit_(uint64_t lsn) {
    std::unique_lock<std::mutex> lk(commit_mutex_);
    while (durable_lsn_ < lsn) {
        if (syncing_) {
            commit_cv_.wait(lk);
        } else {
            syncing_ = true;
            lk.unlock();
            const uint64_t synced = wal_.sync();  // dup+fsync, concurrent with appends
            lk.lock();
            if (synced > durable_lsn_) durable_lsn_ = synced;
            syncing_ = false;
            commit_cv_.notify_all();
        }
    }
}

void DurableVDB::maybe_flush_() {
    std::unique_lock<std::mutex> lk(commit_mutex_);
    const auto now = std::chrono::steady_clock::now();
    if (syncing_ || now - last_sync_ < std::chrono::milliseconds(opts_.flush_interval_ms))
        return;
    syncing_ = true;
    lk.unlock();
    const uint64_t synced = wal_.sync();
    lk.lock();
    if (synced > durable_lsn_) durable_lsn_ = synced;
    last_sync_ = std::chrono::steady_clock::now();
    syncing_   = false;
    commit_cv_.notify_all();
}

void DurableVDB::finish_op_() {
    std::lock_guard<std::mutex> g(commit_mutex_);
    --in_flight_;
    commit_cv_.notify_all();  // a checkpoint may be waiting for the count to hit 0
}

ExternalId DurableVDB::insert(const float* vec, const Record& meta) {
    ExternalId ext;
    uint64_t   lsn;
    bool       do_ckpt = false;
    // Serial prefix: reserve the id, assign the lsn, append the record — atomically,
    // so WAL order == lsn order == ext-id order.
    {
        std::lock_guard<std::mutex> wg(write_mutex_);
        ext = db_.reserve_id();
        lsn = next_lsn_++;
        wal_.append(WalRecord{WalRecordType::Insert, lsn, ext,
                              std::vector<float>(vec, vec + config_.dim), meta});
        {
            std::lock_guard<std::mutex> cg(commit_mutex_);
            ++in_flight_;
        }
        if (++ops_since_checkpoint_ >= opts_.checkpoint_ops) {
            ops_since_checkpoint_ = 0;
            do_ckpt               = true;  // this writer owns the checkpoint
        }
    }

    if (opts_.policy == Policy::PerOpSync) group_commit_(lsn);  // durable before apply
    db_.insert_reserved(ext, vec, meta);                        // apply
    if (opts_.policy == Policy::Periodic) maybe_flush_();       // relaxed: lazy fsync

    finish_op_();
    if (do_ckpt) checkpoint();
    return ext;
}

bool DurableVDB::remove(ExternalId id) {
    if (!db_.contains(id)) return false;
    uint64_t lsn;
    bool     do_ckpt = false;
    {
        std::lock_guard<std::mutex> wg(write_mutex_);
        lsn = next_lsn_++;
        wal_.append(WalRecord{WalRecordType::Delete, lsn, id, {}, {}});
        {
            std::lock_guard<std::mutex> cg(commit_mutex_);
            ++in_flight_;
        }
        if (++ops_since_checkpoint_ >= opts_.checkpoint_ops) {
            ops_since_checkpoint_ = 0;
            do_ckpt               = true;
        }
    }

    if (opts_.policy == Policy::PerOpSync) group_commit_(lsn);
    db_.remove(id);
    if (opts_.policy == Policy::Periodic) maybe_flush_();

    finish_op_();
    if (do_ckpt) checkpoint();
    return true;
}

bool DurableVDB::update(ExternalId id, const float* vec) {
    return update_(id, vec, nullptr);
}

bool DurableVDB::update(ExternalId id, const float* vec, const Record& meta) {
    return update_(id, vec, &meta);
}

bool DurableVDB::update_(ExternalId id, const float* vec, const Record* meta) {
    if (!db_.contains(id)) return false;
    uint64_t lsn;
    bool     do_ckpt = false;
    Record   effective;
    {
        std::lock_guard<std::mutex> wg(write_mutex_);
        // A vector-only update keeps the existing attributes. Materialize them here,
        // inside the serialised prefix, so the log record is self-contained: replay
        // never has to reconstruct "whatever the row happened to be" by tracking
        // earlier records. Same principle as logging the full vector rather than a
        // delta.
        if (meta) effective = *meta;
        else if (!db_.get_metadata(id, effective)) return false;

        lsn = next_lsn_++;
        wal_.append(WalRecord{WalRecordType::Update, lsn, id,
                              std::vector<float>(vec, vec + config_.dim), effective});
        {
            std::lock_guard<std::mutex> cg(commit_mutex_);
            ++in_flight_;
        }
        if (++ops_since_checkpoint_ >= opts_.checkpoint_ops) {
            ops_since_checkpoint_ = 0;
            do_ckpt               = true;
        }
    }

    if (opts_.policy == Policy::PerOpSync) group_commit_(lsn);
    db_.update(id, vec, effective);
    if (opts_.policy == Policy::Periodic) maybe_flush_();

    finish_op_();
    if (do_ckpt) checkpoint();
    return true;
}

bool DurableVDB::set_metadata(ExternalId id, const Record& meta) {
    if (!db_.contains(id)) return false;
    uint64_t lsn;
    bool     do_ckpt = false;
    {
        std::lock_guard<std::mutex> wg(write_mutex_);
        lsn = next_lsn_++;
        wal_.append(WalRecord{WalRecordType::SetMeta, lsn, id, {}, meta});
        {
            std::lock_guard<std::mutex> cg(commit_mutex_);
            ++in_flight_;
        }
        if (++ops_since_checkpoint_ >= opts_.checkpoint_ops) {
            ops_since_checkpoint_ = 0;
            do_ckpt               = true;
        }
    }

    if (opts_.policy == Policy::PerOpSync) group_commit_(lsn);
    db_.set_metadata(id, meta);
    if (opts_.policy == Policy::Periodic) maybe_flush_();

    finish_op_();
    if (do_ckpt) checkpoint();
    return true;
}

void DurableVDB::checkpoint() {
    std::unique_lock<std::mutex> wg(write_mutex_);  // block new prefixes
    // Wait for prefixed-but-unapplied ops to finish, so db_ is quiescent and the
    // snapshot is a consistent point-in-time image.
    {
        std::unique_lock<std::mutex> cg(commit_mutex_);
        commit_cv_.wait(cg, [&] { return in_flight_ == 0; });
    }

    const uint64_t lsn = next_lsn_ - 1;  // highest lsn applied so far

    save_snapshot(db_, snapshot_path_, lsn);
    wal_.append(WalRecord{WalRecordType::Checkpoint, next_lsn_++, snapshot_id_++, {}, {}});
    wal_.sync();
    wal_.truncate(lsn);

    std::lock_guard<std::mutex> cg(commit_mutex_);
    durable_lsn_ = std::max(durable_lsn_, lsn);
    last_sync_   = std::chrono::steady_clock::now();
}

void DurableVDB::sync() {
    const uint64_t synced = wal_.sync();
    std::lock_guard<std::mutex> cg(commit_mutex_);
    if (synced > durable_lsn_) durable_lsn_ = synced;
    last_sync_ = std::chrono::steady_clock::now();
}

}
