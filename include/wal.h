#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include <hnswindex.h>  // ExternalId

// Durability policy: when does append() force bytes to stable storage?
//
//   Always  — fsync after every record. Safest, slowest (a disk round-trip per
//             write). This is Postgres' synchronous_commit=on / Redis' appendfsync=always.
//   EveryMs — fsync at most once per `fsync_interval_ms`. Bounds data loss on a
//             crash to the last window while amortizing the fsync cost across
//             many writes. (Redis appendfsync=everysec is this with 1000ms.)
//   Never   — never fsync; rely on the OS to flush eventually. A clean process
//             exit still survives (bytes reach the page cache via write()), but a
//             power cut can lose recently "committed" writes. Fastest.
enum class DurabilityPolicy { Always, EveryMs, Never };

// A single log record. `lsn` (Log Sequence Number) is a monotonically increasing
// id assigned by the caller (VDB) — it is the vocabulary that ties WAL records to
// snapshots ("this snapshot contains every change up to LSN N").
enum class WalRecordType : uint8_t {
    Insert     = 1,  // ext_id + vector
    Delete     = 2,  // ext_id
    Update     = 3,  // ext_id + vector  (delete+insert as ONE atomic record)
    Checkpoint = 4,  // snapshot_id      (breadcrumb: WAL continues from snapshot N)
};

struct WalRecord {
    WalRecordType      type;
    uint64_t           lsn         = 0;
    ExternalId         ext_id      = 0;  // Insert/Delete/Update
    std::vector<float> vec;              // Insert/Update
    uint64_t           snapshot_id = 0;  // Checkpoint
};

// Append-only, self-describing, checksummed log file.
//
// On-disk layout:
//   [ 32-byte header: magic, version, dim, metric ]
//   then a stream of records, each framed as:
//     u32 length   = byte length of (type + lsn + payload)
//     u32 crc32c   = checksum over (length-bytes ++ type ++ lsn ++ payload)
//     u8  type
//     u64 lsn
//     payload (depends on type)
//
// The length prefix lets us frame records without parsing them; the CRC lets
// recovery distinguish a clean log tail from a torn/half-written record after a
// crash (see replay()).
class Wal {
public:
    Wal() = default;
    ~Wal();

    Wal(const Wal&)            = delete;
    Wal& operator=(const Wal&) = delete;

    // Open `path` for read+append, creating it (with a fresh header) if absent.
    // If it already exists, the header is validated against (dim, metric) and a
    // std::runtime_error is thrown on mismatch — this guards against opening a
    // database with a configuration different from the one that wrote it.
    void open(const std::string& path, uint32_t dim, uint8_t metric,
              DurabilityPolicy policy);

    // Replay every intact record from the start of the log, invoking `cb` for
    // each. Stops at the first torn/corrupt record (bad CRC or short read),
    // truncating the file at that boundary so future appends start from a clean
    // point. After replay the file offset is positioned at the (possibly
    // truncated) end, ready for append(). Must be called once, right after open().
    void replay(const std::function<void(const WalRecord&)>& cb);

    // Frame, checksum, and write `rec` to the end of the log, then fsync per the
    // configured policy. This is the durability point for Always.
    void append(const WalRecord& rec);

    // Force any buffered bytes to stable storage now, regardless of policy.
    void sync();

    // Discard all records, keeping only the header, then write `first` as the new
    // first record. Used after a snapshot: the snapshot already captures all state
    // up to its LSN, so the old WAL records are redundant and can be reclaimed.
    void rotate(const WalRecord& first);

    void close();

private:
    void write_header(uint32_t dim, uint8_t metric);
    void read_and_check_header(uint32_t dim, uint8_t metric);
    void maybe_sync();
    void do_fsync();

    int              fd_          = -1;
    DurabilityPolicy policy_      = DurabilityPolicy::Always;
    uint32_t         dim_         = 0;
    int64_t          last_sync_ms_ = 0;  // for EveryMs
    uint32_t         fsync_interval_ms_ = 1000;

public:
    // Bytes occupied by the fixed file header (records start here).
    static constexpr long kHeaderSize = 32;
    void set_fsync_interval_ms(uint32_t ms) { fsync_interval_ms_ = ms; }
};
