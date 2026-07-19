#include "durable_vdb.h"

#include "snapshot.h"

#include <algorithm>
#include <cassert>
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
              static_cast<uint8_t>(config_.metric), opts_.wal);
    wal_.replay([&](const WalRecord& r) {
        if (r.lsn > snapshot_lsn) apply_(r);  // skip records already in the snapshot
    });

    next_lsn_ = std::max(snapshot_lsn, wal_.max_lsn()) + 1;
}

void DurableVDB::apply_(const WalRecord& r) {
    switch (r.type) {
        case WalRecordType::Insert: {
            const ExternalId got = db_.insert(r.vec.data());
            assert(got == r.ext_id);
            (void)got;
            break;
        }
        case WalRecordType::Update:
            db_.update(r.ext_id, r.vec.data());
            break;
        case WalRecordType::Delete:
            db_.remove(r.ext_id);
            break;
        case WalRecordType::Checkpoint:
            break;
    }
}

void DurableVDB::log_(WalRecord&& r) {
    wal_.append(r);
    if (opts_.policy == Policy::PerOpSync) wal_.sync();  // durable before apply
}

ExternalId DurableVDB::insert(const float* vec) {
    const ExternalId id = db_.peek_next_id();
    log_(WalRecord{WalRecordType::Insert, next_lsn_++, id,
                   std::vector<float>(vec, vec + config_.dim)});

    const ExternalId got = db_.insert(vec);
    assert(got == id);
    (void)got;

    ++ops_since_checkpoint_;
    maybe_flush_();
    maybe_checkpoint_();
    return id;
}

bool DurableVDB::remove(ExternalId id) {
    if (!db_.contains(id)) return false;
    log_(WalRecord{WalRecordType::Delete, next_lsn_++, id, {}});
    db_.remove(id);

    ++ops_since_checkpoint_;
    maybe_flush_();
    maybe_checkpoint_();
    return true;
}

bool DurableVDB::update(ExternalId id, const float* vec) {
    if (!db_.contains(id)) return false;
    log_(WalRecord{WalRecordType::Update, next_lsn_++, id,
                   std::vector<float>(vec, vec + config_.dim)});
    db_.update(id, vec);

    ++ops_since_checkpoint_;
    maybe_flush_();
    maybe_checkpoint_();
    return true;
}

void DurableVDB::maybe_flush_() {
    if (opts_.policy != Policy::Periodic) return;
    const auto now = std::chrono::steady_clock::now();
    if (now - last_sync_ >= std::chrono::milliseconds(opts_.flush_interval_ms)) {
        wal_.sync();
        last_sync_ = now;
    }
}

void DurableVDB::maybe_checkpoint_() {
    if (ops_since_checkpoint_ >= opts_.checkpoint_ops) checkpoint();
}

void DurableVDB::checkpoint() {
    wal_.sync();
    const uint64_t lsn = next_lsn_ - 1;  // highest lsn applied so far

    save_snapshot(db_, snapshot_path_, lsn);
    wal_.append(WalRecord{WalRecordType::Checkpoint, next_lsn_++, snapshot_id_++, {}});
    wal_.sync();
    wal_.truncate(lsn);

    ops_since_checkpoint_ = 0;
    last_sync_            = std::chrono::steady_clock::now();
}

void DurableVDB::sync() {
    wal_.sync();
    last_sync_ = std::chrono::steady_clock::now();
}

}
