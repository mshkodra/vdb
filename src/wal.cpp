#include <wal.h>

#include <crc32c.h>
#include <serialize.h>

#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>

#include <chrono>
#include <cstring>
#include <stdexcept>

namespace {

constexpr uint32_t kMagic   = 0x56444257u;  // "VDBW"
constexpr uint16_t kVersion = 1;

int64_t now_ms() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count();
}

// write() can return short; loop until the whole buffer is written.
void write_all(int fd, const void* data, size_t len) {
    const auto* p = static_cast<const uint8_t*>(data);
    size_t written = 0;
    while (written < len) {
        const ssize_t n = ::write(fd, p + written, len - written);
        if (n < 0) {
            if (errno == EINTR) continue;
            throw std::runtime_error(std::string("wal write failed: ") + std::strerror(errno));
        }
        written += static_cast<size_t>(n);
    }
}

// read() into buf; returns the number of bytes actually read (0 at clean EOF,
// < len for a torn tail). Retries on EINTR.
size_t read_some(int fd, void* buf, size_t len) {
    auto* p = static_cast<uint8_t*>(buf);
    size_t got = 0;
    while (got < len) {
        const ssize_t n = ::read(fd, p + got, len - got);
        if (n < 0) {
            if (errno == EINTR) continue;
            throw std::runtime_error(std::string("wal read failed: ") + std::strerror(errno));
        }
        if (n == 0) break;  // EOF
        got += static_cast<size_t>(n);
    }
    return got;
}

}  // namespace

Wal::~Wal() { close(); }

void Wal::close() {
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
}

void Wal::do_fsync() {
#ifdef __APPLE__
    // On macOS, fsync() does NOT guarantee the data reaches the platter — it only
    // flushes to the drive cache. F_FULLFSYNC is the real barrier. Fall back to
    // fsync if the device rejects the fcntl (some filesystems don't support it).
    if (::fcntl(fd_, F_FULLFSYNC) == 0) return;
#endif
    if (::fsync(fd_) != 0) {
        throw std::runtime_error(std::string("wal fsync failed: ") + std::strerror(errno));
    }
}

void Wal::maybe_sync() {
    switch (policy_) {
        case DurabilityPolicy::Always:
            do_fsync();
            break;
        case DurabilityPolicy::EveryMs: {
            const int64_t t = now_ms();
            if (last_sync_ms_ == 0 || t - last_sync_ms_ >= fsync_interval_ms_) {
                do_fsync();
                last_sync_ms_ = t;
            }
            break;
        }
        case DurabilityPolicy::Never:
            break;
    }
}

void Wal::sync() {
    if (fd_ >= 0) do_fsync();
}

void Wal::write_header(uint32_t dim, uint8_t metric) {
    uint8_t hdr[kHeaderSize] = {0};
    std::memcpy(hdr + 0, &kMagic, 4);
    std::memcpy(hdr + 4, &kVersion, 2);
    // 2 bytes pad at offset 6
    std::memcpy(hdr + 8, &dim, 4);
    hdr[12] = metric;
    // remainder reserved (zero)
    write_all(fd_, hdr, kHeaderSize);
    do_fsync();
}

void Wal::read_and_check_header(uint32_t dim, uint8_t metric) {
    uint8_t hdr[kHeaderSize] = {0};
    if (::lseek(fd_, 0, SEEK_SET) < 0) {
        throw std::runtime_error("wal lseek failed");
    }
    if (read_some(fd_, hdr, kHeaderSize) != kHeaderSize) {
        throw std::runtime_error("wal header is truncated");
    }
    uint32_t magic;   std::memcpy(&magic, hdr + 0, 4);
    uint16_t version; std::memcpy(&version, hdr + 4, 2);
    uint32_t fdim;    std::memcpy(&fdim, hdr + 8, 4);
    const uint8_t fmetric = hdr[12];

    if (magic != kMagic)     throw std::runtime_error("wal: bad magic (not a VDB WAL)");
    if (version != kVersion) throw std::runtime_error("wal: unsupported version");
    if (fdim != dim)         throw std::runtime_error("wal: dim mismatch with config");
    if (fmetric != metric)   throw std::runtime_error("wal: metric mismatch with config");
}

void Wal::open(const std::string& path, uint32_t dim, uint8_t metric,
               DurabilityPolicy policy) {
    policy_ = policy;
    dim_    = dim;

    fd_ = ::open(path.c_str(), O_RDWR | O_CREAT, 0644);
    if (fd_ < 0) {
        throw std::runtime_error(std::string("wal open failed: ") + std::strerror(errno));
    }

    struct stat st {};
    if (::fstat(fd_, &st) != 0) {
        throw std::runtime_error("wal fstat failed");
    }

    if (st.st_size == 0) {
        write_header(dim, metric);
    } else {
        read_and_check_header(dim, metric);
    }
}

void Wal::replay(const std::function<void(const WalRecord&)>& cb) {
    // Position right after the header; records begin there.
    if (::lseek(fd_, kHeaderSize, SEEK_SET) < 0) {
        throw std::runtime_error("wal lseek failed");
    }

    long good_end = kHeaderSize;  // offset of the last clean record boundary

    while (true) {
        uint32_t length = 0, crc_stored = 0;

        // Read the frame prefix (length + crc). A short read here means we are at
        // a clean EOF (8 == 0 bytes left) or a torn frame header — either way the
        // log ends at good_end.
        uint8_t prefix[8];
        const size_t pn = read_some(fd_, prefix, 8);
        if (pn == 0) break;            // clean EOF on a record boundary
        if (pn < 8) break;             // torn frame header → stop
        std::memcpy(&length, prefix + 0, 4);
        std::memcpy(&crc_stored, prefix + 4, 4);

        // Sanity-bound the length so a corrupt prefix can't make us allocate wildly.
        if (length == 0 || length > (1u << 28)) break;

        std::vector<uint8_t> body(length);
        if (read_some(fd_, body.data(), length) != length) break;  // torn payload

        // Recompute the CRC over (length-bytes ++ body) and compare. A mismatch is
        // the signature of a half-written record from a crash mid-append: this
        // record and everything after it is untrustworthy, so we stop here.
        uint32_t crc = crc::crc32c(0, prefix, 4);  // the length field
        crc = crc::crc32c(crc, body.data(), body.size());
        if (crc != crc_stored) break;

        // Parse the body: type, lsn, then a type-specific payload.
        size_t off = 0;
        uint8_t type_raw = 0;
        if (!ser::get_pod(body, off, type_raw)) break;
        WalRecord rec;
        rec.type = static_cast<WalRecordType>(type_raw);
        if (!ser::get_pod(body, off, rec.lsn)) break;

        bool ok = true;
        switch (rec.type) {
            case WalRecordType::Insert:
            case WalRecordType::Update: {
                ok = ser::get_pod(body, off, rec.ext_id);
                rec.vec.resize(dim_);
                if (ok && off + dim_ * sizeof(float) > body.size()) ok = false;
                if (ok) {
                    std::memcpy(rec.vec.data(), body.data() + off, dim_ * sizeof(float));
                    off += dim_ * sizeof(float);
                }
                break;
            }
            case WalRecordType::Delete:
                ok = ser::get_pod(body, off, rec.ext_id);
                break;
            case WalRecordType::Checkpoint:
                ok = ser::get_pod(body, off, rec.snapshot_id);
                break;
            default:
                ok = false;  // unknown type → treat as corruption
        }
        if (!ok) break;

        cb(rec);
        good_end = ::lseek(fd_, 0, SEEK_CUR);
    }

    // Drop any torn tail so the next append starts from a clean boundary, and
    // position the file there.
    if (::ftruncate(fd_, good_end) != 0) {
        throw std::runtime_error("wal ftruncate failed");
    }
    if (::lseek(fd_, good_end, SEEK_SET) < 0) {
        throw std::runtime_error("wal lseek failed");
    }
}

void Wal::append(const WalRecord& rec) {
    // Build the body = type + lsn + payload.
    std::vector<uint8_t> body;
    ser::put_pod(body, static_cast<uint8_t>(rec.type));
    ser::put_pod(body, rec.lsn);
    switch (rec.type) {
        case WalRecordType::Insert:
        case WalRecordType::Update:
            ser::put_pod(body, rec.ext_id);
            ser::put_bytes(body, rec.vec.data(), rec.vec.size() * sizeof(float));
            break;
        case WalRecordType::Delete:
            ser::put_pod(body, rec.ext_id);
            break;
        case WalRecordType::Checkpoint:
            ser::put_pod(body, rec.snapshot_id);
            break;
    }

    const uint32_t length = static_cast<uint32_t>(body.size());
    uint8_t len_bytes[4];
    std::memcpy(len_bytes, &length, 4);

    uint32_t crc = crc::crc32c(0, len_bytes, 4);
    crc = crc::crc32c(crc, body.data(), body.size());

    // One contiguous buffer → one write() syscall → no interleaving with other
    // records (we are single-threaded, but it also keeps the record atomic at the
    // syscall layer).
    std::vector<uint8_t> frame;
    frame.reserve(8 + body.size());
    ser::put_pod(frame, length);
    ser::put_pod(frame, crc);
    frame.insert(frame.end(), body.begin(), body.end());

    write_all(fd_, frame.data(), frame.size());
    maybe_sync();
}

void Wal::rotate(const WalRecord& first) {
    if (::ftruncate(fd_, kHeaderSize) != 0) {
        throw std::runtime_error("wal rotate truncate failed");
    }
    if (::lseek(fd_, kHeaderSize, SEEK_SET) < 0) {
        throw std::runtime_error("wal lseek failed");
    }
    append(first);
    do_fsync();  // a checkpoint must be durable before we trust the rotation
}
