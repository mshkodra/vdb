#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <hnswindex.h>  // HNSWIndex, ExternalId

// A snapshot (a.k.a. checkpoint) is a full, self-contained serialization of the
// database at one instant: the identity layer (next_ext_id + the internal->external
// map) plus the entire HNSW graph. Recovery loads the newest snapshot, then replays
// only the WAL records that came *after* it — so the WAL never has to be replayed
// from the beginning of time, and it can be truncated once a snapshot covers it.
//
// On-disk layout (snapshot.<id> file):
//   header:
//     u32 magic ("VDBS"), u16 version, u16 pad, u32 dim, u8 metric, pad to 32
//     u64 lsn_at_snapshot   — every change with LSN <= this is included
//     u64 next_ext_id       — so id minting resumes without recycling
//     u64 node_count        — == int_to_ext.size()
//   identity:
//     u64[node_count]       — int_to_ext (ExternalId per internal slot)
//   graph:
//     HNSWIndex::save() blob (vectors, neighbours, tombstones, entry point)
//
// Files are written to a temp name, fsync'd, then atomically rename()d into place,
// so a crash mid-write never replaces a good snapshot with a half-written one.
namespace snapshot {

// Write a snapshot for `id` into `dir`. Throws std::runtime_error on I/O failure.
// After a successful rename, older snapshot files in `dir` are deleted.
void save(const std::string& dir, uint64_t id, uint32_t dim, uint8_t metric,
          uint64_t lsn_at_snapshot, uint64_t next_ext_id,
          const std::vector<ExternalId>& int_to_ext, const HNSWIndex& idx);

// Load the newest valid snapshot in `dir` into `idx` and the out-params. Returns
// false if `dir` has no loadable snapshot (caller then starts from an empty index).
// (dim/metric are validated against the file header.)
bool load_latest(const std::string& dir, uint32_t expect_dim, uint8_t expect_metric,
                 uint64_t& lsn_at_snapshot, uint64_t& next_ext_id,
                 std::vector<ExternalId>& int_to_ext, HNSWIndex& idx,
                 uint64_t& snapshot_id);

}  // namespace snapshot
