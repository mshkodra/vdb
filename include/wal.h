#pragma once
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace vdb {

// On-disk WAL: a 32-byte header then length-prefixed, CRC32C-checked records.
// Logical logging: each record is an operation to replay, not graph edges.

constexpr uint32_t kWalMagic      = 0x56444257u;  // "VDBW"
constexpr uint16_t kWalVersion    = 1;
constexpr size_t   kWalHeaderSize = 32;

enum class WalRecordType : uint8_t {
    Insert     = 1,
    Delete     = 2,
    Update     = 3,
    Checkpoint = 4,
};

// vec is set only for Insert/Update. Checkpoint carries snapshot_id in ext_id.
struct WalRecord {
    WalRecordType      type;
    uint64_t           lsn;
    uint64_t           ext_id;
    std::vector<float> vec;
};

struct WalHeader {
    uint32_t magic   = kWalMagic;
    uint16_t version = kWalVersion;
    uint32_t dim     = 0;
    uint8_t  metric  = 0;
};

std::vector<uint8_t> encode_header(uint32_t dim, uint8_t metric);
bool decode_header(const uint8_t* data, size_t avail, WalHeader& out);

// Frame: [length u32][crc u32][type u8][lsn u64][ext_id u64][vec?]. length counts
// the body; the crc covers the length field and the body.
std::vector<uint8_t> encode_record(const WalRecord& r);

enum class DecodeStatus {
    Ok,          // complete, checksum-valid record
    Incomplete,  // fewer bytes than a full frame (torn tail)
    BadCrc,      // fully framed but checksum mismatch (corruption)
};

DecodeStatus decode_record(const uint8_t* data, size_t avail,
                           WalRecord& out, size_t& consumed);

// Segmented append-only log. Records never straddle a segment boundary, so every
// sealed segment is complete and only the active one can hold a torn tail.
// Segment numbers increase forever; reclaim unlinks whole covered segments.
//
// Usage: open(), then replay() once, then append()/sync() and truncate().
//
// Thread-safety (Stage 7). append(), sync(), truncate(), and max_lsn() may run
// concurrently. `io_mu_` guards the fd/segment metadata. sync() holds it only long
// enough to dup() the active fd and read the high-water lsn, then fsyncs the *dup*
// with the lock released — so an fsync (group commit) runs while other writers keep
// appending. The dup is an independent fd to the same file, so a concurrent rotate
// closing the original fd can't pull it out from under the fsync. replay()/open()
// are single-threaded (used before concurrent operation begins).
class Wal {
public:
    struct Options {
        size_t segment_max_bytes = 16 * 1024 * 1024;
    };

    Wal() = default;
    ~Wal();
    Wal(const Wal&) = delete;
    Wal& operator=(const Wal&) = delete;

    void open(const std::string& dir, uint32_t dim, uint8_t metric, Options opts);
    void open(const std::string& dir, uint32_t dim, uint8_t metric) {
        open(dir, dim, metric, Options{});
    }

    // Replay records in lsn order. Truncates a torn tail on the active segment;
    // throws on corruption in a sealed segment.
    void replay(const std::function<void(const WalRecord&)>& on_record);

    void append(const WalRecord& r);

    // fsync the log — the durability point. Returns the highest lsn now durable
    // (the high-water at the moment the fsync began), so a group-commit caller can
    // advance its durable_lsn for every writer the one fsync covered.
    uint64_t sync();

    // Unlink sealed segments whose highest lsn <= upto_lsn.
    void truncate(uint64_t upto_lsn);

    uint64_t max_lsn() const {
        std::lock_guard<std::mutex> g(io_mu_);
        return highest_lsn_;
    }

private:
    struct Segment {
        uint64_t    number;
        std::string path;
        uint64_t    first_lsn = 0;
        uint64_t    last_lsn  = 0;
        bool        sealed    = false;
    };

    void                  open_new_segment_();
    void                  rotate_();
    std::string           segment_path_(uint64_t number) const;
    std::vector<uint64_t> discover_segments_() const;
    std::vector<uint8_t>  read_file_(const std::string& path) const;
    void                  write_at_(int fd, long long off, const void* data, size_t len);
    void                  fsync_dir_();
    static void           sys_check(bool ok, const char* what);

    mutable std::mutex   io_mu_;  // guards active_fd_/active_bytes_/segments_/highest_lsn_
    std::string          dir_;
    uint32_t             dim_     = 0;
    uint8_t              metric_  = 0;
    size_t               seg_max_ = 16 * 1024 * 1024;
    std::vector<Segment> segments_;
    int                  active_fd_    = -1;
    size_t               active_bytes_ = 0;
    uint64_t             highest_lsn_  = 0;
};

}
