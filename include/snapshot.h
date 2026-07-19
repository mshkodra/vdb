#pragma once
#include <cstdint>
#include <string>

namespace vdb {

class VDB;

// Serialize the whole database (id maps, tombstones, vectors, and the index graph)
// to `path` via a temp file + atomic rename. `lsn` is the last log record included.
void save_snapshot(const VDB& db, const std::string& path, uint64_t lsn);

// Load a snapshot into `db`, which must be freshly constructed with matching
// config. Returns the snapshot's lsn.
uint64_t load_snapshot(VDB& db, const std::string& path);

}
