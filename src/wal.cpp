#include "wal.h"

#include "crc32c.h"
#include "serialize.h"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <stdexcept>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace vdb {
namespace {

// Little-endian host assumed; serialization is raw memcpy of native integers.
// `put` comes from serialize.h (identical semantics); only the raw-pointer read
// below is local, since the header's Reader is a cursor rather than a random-access
// getter and the fixed-offset header fields are read positionally.
template <typename T>
T get(const uint8_t* p) {
    T v;
    std::memcpy(&v, p, sizeof(T));
    return v;
}

// type + lsn + one u64 (ext_id or snapshot_id); the rest of the body is per-type.
constexpr size_t kBodyPrefix = sizeof(uint8_t) + sizeof(uint64_t) + sizeof(uint64_t);

bool carries_vector(WalRecordType t) {
    return t == WalRecordType::Insert || t == WalRecordType::Update;
}
bool carries_meta(WalRecordType t) {
    return carries_vector(t) || t == WalRecordType::SetMeta;
}

// Self-describing so the encoder/decoder never need the schema: each value writes
// its own type byte. The VDB layer still validates against the schema on apply.
void put_meta(std::vector<uint8_t>& body, const Record& m) {
    put(body, static_cast<uint32_t>(m.attrs.size()));
    for (const auto& a : m.attrs) {
        put(body, static_cast<uint8_t>(a.type));
        if (a.type == AttrType::Null) continue;
        if (a.type == AttrType::Tag || a.type == AttrType::Text) {
            put(body, static_cast<uint32_t>(a.text.size()));
            body.insert(body.end(), a.text.begin(), a.text.end());
        } else {
            put(body, a.raw);
        }
    }
    put(body, static_cast<uint32_t>(m.payload.size()));
    body.insert(body.end(), m.payload.begin(), m.payload.end());
}

// Bounds-checked via serialize.h's Reader, which throws past the end; decode_record
// catches that and reports BadCrc (a checksum-valid but malformed frame).
Record get_meta(Reader& rd) {
    Record m;
    const uint32_t n = rd.get<uint32_t>();
    m.attrs.reserve(n);
    for (uint32_t i = 0; i < n; ++i) {
        AttrValue v;
        v.type = static_cast<AttrType>(rd.get<uint8_t>());
        switch (v.type) {
            case AttrType::Null: break;
            case AttrType::Tag:
            case AttrType::Text: {
                const uint32_t len = rd.get<uint32_t>();
                v.text.resize(len);
                for (uint32_t k = 0; k < len; ++k)
                    v.text[k] = static_cast<char>(rd.get<uint8_t>());
                break;
            }
            case AttrType::Int64:
            case AttrType::Float64:
            case AttrType::Bool:
                v.raw = rd.get<uint64_t>();
                break;
            default:
                throw std::runtime_error("wal: bad attribute type");
        }
        m.attrs.push_back(std::move(v));
    }
    const uint32_t plen = rd.get<uint32_t>();
    m.payload.resize(plen);
    for (uint32_t k = 0; k < plen; ++k) m.payload[k] = rd.get<uint8_t>();
    return m;
}

}  // namespace

std::vector<uint8_t> encode_header(uint32_t dim, uint8_t metric, uint64_t schema_fp) {
    std::vector<uint8_t> buf;
    buf.reserve(kWalHeaderSize);
    put(buf, kWalMagic);
    put(buf, kWalVersion);
    put(buf, dim);
    put(buf, metric);
    put(buf, schema_fp);
    buf.resize(kWalHeaderSize, 0);
    return buf;
}

bool decode_header(const uint8_t* data, size_t avail, WalHeader& out) {
    if (avail < kWalHeaderSize) return false;
    size_t off = 0;
    out.magic     = get<uint32_t>(data + off); off += sizeof(uint32_t);
    out.version   = get<uint16_t>(data + off); off += sizeof(uint16_t);
    out.dim       = get<uint32_t>(data + off); off += sizeof(uint32_t);
    out.metric    = get<uint8_t>(data + off);  off += sizeof(uint8_t);
    out.schema_fp = get<uint64_t>(data + off);
    return out.magic == kWalMagic && out.version == kWalVersion;
}

std::vector<uint8_t> encode_record(const WalRecord& r) {
    std::vector<uint8_t> body;
    body.reserve(kBodyPrefix + sizeof(uint32_t) + r.vec.size() * sizeof(float));
    put(body, static_cast<uint8_t>(r.type));
    put(body, r.lsn);
    put(body, r.ext_id);
    if (carries_vector(r.type)) {
        put(body, static_cast<uint32_t>(r.vec.size()));
        const auto* vp = reinterpret_cast<const uint8_t*>(r.vec.data());
        body.insert(body.end(), vp, vp + r.vec.size() * sizeof(float));
    }
    if (carries_meta(r.type)) put_meta(body, r.meta);

    const uint32_t length = static_cast<uint32_t>(body.size());

    // crc over the length field then the body, so a corrupted length is caught too.
    std::vector<uint8_t> crc_input;
    crc_input.reserve(sizeof(uint32_t) + body.size());
    put(crc_input, length);
    crc_input.insert(crc_input.end(), body.begin(), body.end());
    const uint32_t crc = crc32c(crc_input.data(), crc_input.size());

    std::vector<uint8_t> frame;
    frame.reserve(2 * sizeof(uint32_t) + body.size());
    put(frame, length);
    put(frame, crc);
    frame.insert(frame.end(), body.begin(), body.end());
    return frame;
}

DecodeStatus decode_record(const uint8_t* data, size_t avail,
                           WalRecord& out, size_t& consumed) {
    consumed = 0;

    constexpr size_t kFrameHeader = 2 * sizeof(uint32_t);
    if (avail < kFrameHeader) return DecodeStatus::Incomplete;

    const uint32_t length     = get<uint32_t>(data);
    const uint32_t stored_crc = get<uint32_t>(data + sizeof(uint32_t));
    if (avail < kFrameHeader + static_cast<size_t>(length))
        return DecodeStatus::Incomplete;

    const uint8_t* body = data + kFrameHeader;

    std::vector<uint8_t> crc_input;
    crc_input.reserve(sizeof(uint32_t) + length);
    crc_input.insert(crc_input.end(), data, data + sizeof(uint32_t));
    crc_input.insert(crc_input.end(), body, body + length);
    if (crc32c(crc_input.data(), crc_input.size()) != stored_crc)
        return DecodeStatus::BadCrc;

    // Checksum-valid but malformed framing: treat as corruption, not a valid decode.
    if (length < kBodyPrefix) return DecodeStatus::BadCrc;

    // Every field past the prefix is variable-length now, so read through a
    // bounds-checked cursor and turn an overrun into BadCrc.
    try {
        Reader rd(body, length);
        out.type   = static_cast<WalRecordType>(rd.get<uint8_t>());
        out.lsn    = rd.get<uint64_t>();
        out.ext_id = rd.get<uint64_t>();
        out.vec.clear();
        out.meta = Record{};
        if (carries_vector(out.type)) {
            const uint32_t n = rd.get<uint32_t>();
            out.vec.resize(n);
            for (uint32_t i = 0; i < n; ++i) out.vec[i] = rd.get<float>();
        }
        if (carries_meta(out.type)) out.meta = get_meta(rd);
    } catch (const std::exception&) {
        return DecodeStatus::BadCrc;
    }

    consumed = kFrameHeader + length;
    return DecodeStatus::Ok;
}

// ---- Wal --------------------------------------------------------------------

Wal::~Wal() {
    if (active_fd_ >= 0) ::close(active_fd_);
}

void Wal::sys_check(bool ok, const char* what) {
    if (!ok) throw std::runtime_error(std::string(what) + ": " + std::strerror(errno));
}

std::string Wal::segment_path_(uint64_t number) const {
    char name[32];
    std::snprintf(name, sizeof(name), "wal-%012llu.log",
                  static_cast<unsigned long long>(number));
    return (std::filesystem::path(dir_) / name).string();
}

std::vector<uint64_t> Wal::discover_segments_() const {
    const std::string kExample = "wal-000000000001.log";
    std::vector<uint64_t> numbers;
    for (const auto& entry : std::filesystem::directory_iterator(dir_)) {
        if (!entry.is_regular_file()) continue;
        const std::string name = entry.path().filename().string();
        if (name.size() != kExample.size()) continue;
        if (name.compare(0, 4, "wal-") != 0) continue;
        if (name.compare(name.size() - 4, 4, ".log") != 0) continue;
        try {
            numbers.push_back(std::stoull(name.substr(4, name.size() - 8)));
        } catch (...) {
            continue;
        }
    }
    std::sort(numbers.begin(), numbers.end());
    return numbers;
}

void Wal::write_at_(int fd, long long off, const void* data, size_t len) {
    const auto* p = static_cast<const uint8_t*>(data);
    size_t written = 0;
    while (written < len) {
        const ssize_t n = ::pwrite(fd, p + written, len - written,
                                   static_cast<off_t>(off) + static_cast<off_t>(written));
        sys_check(n > 0, "pwrite WAL");
        written += static_cast<size_t>(n);
    }
}

void Wal::fsync_dir_() {
    const int dfd = ::open(dir_.c_str(), O_RDONLY);
    sys_check(dfd >= 0, "open dir for fsync");
    if (::fsync(dfd) != 0 && errno != EINVAL) {  // some platforms reject dir fsync
        const int saved = errno;
        ::close(dfd);
        errno = saved;
        sys_check(false, "fsync dir");
    }
    ::close(dfd);
}

std::vector<uint8_t> Wal::read_file_(const std::string& path) const {
    const int fd = ::open(path.c_str(), O_RDONLY);
    sys_check(fd >= 0, "open WAL segment for read");
    struct stat st {};
    if (::fstat(fd, &st) != 0) {
        const int saved = errno;
        ::close(fd);
        errno = saved;
        sys_check(false, "fstat WAL segment");
    }
    std::vector<uint8_t> buf(static_cast<size_t>(st.st_size));
    size_t rd = 0;
    while (rd < buf.size()) {
        const ssize_t n = ::read(fd, buf.data() + rd, buf.size() - rd);
        if (n == 0) break;
        if (n < 0) {
            const int saved = errno;
            ::close(fd);
            errno = saved;
            sys_check(false, "read WAL segment");
        }
        rd += static_cast<size_t>(n);
    }
    ::close(fd);
    buf.resize(rd);
    return buf;
}

void Wal::open_new_segment_() {
    const uint64_t number = segments_.empty() ? 1 : segments_.back().number + 1;
    const std::string path = segment_path_(number);

    const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0644);
    sys_check(fd >= 0, "create WAL segment");

    const auto hdr = encode_header(dim_, metric_, schema_fp_);
    write_at_(fd, 0, hdr.data(), hdr.size());
    sys_check(::fsync(fd) == 0, "fsync new WAL segment");
    fsync_dir_();  // persist the new directory entry

    segments_.push_back(Segment{number, path, 0, 0, false});
    active_fd_    = fd;
    active_bytes_ = hdr.size();
}

void Wal::rotate_() {
    sys_check(::fsync(active_fd_) == 0, "fsync before rotate");
    sys_check(::close(active_fd_) == 0, "close active segment");
    active_fd_ = -1;
    segments_.back().sealed = true;
    open_new_segment_();
}

void Wal::open(const std::string& dir, uint32_t dim, uint8_t metric, uint64_t schema_fp,
               Options opts) {
    dir_        = dir;
    dim_        = dim;
    metric_     = metric;
    schema_fp_  = schema_fp;
    seg_max_    = opts.segment_max_bytes;
    std::filesystem::create_directories(dir_);

    const std::vector<uint64_t> numbers = discover_segments_();
    if (numbers.empty()) {
        open_new_segment_();
        return;
    }

    for (size_t i = 0; i < numbers.size(); ++i) {
        segments_.push_back(Segment{numbers[i], segment_path_(numbers[i]), 0, 0,
                                    /*sealed=*/i + 1 < numbers.size()});
    }
    const int fd = ::open(segments_.back().path.c_str(), O_WRONLY, 0644);
    sys_check(fd >= 0, "open active WAL segment");
    active_fd_    = fd;
    active_bytes_ = static_cast<size_t>(std::filesystem::file_size(segments_.back().path));
}

void Wal::replay(const std::function<void(const WalRecord&)>& on_record) {
    for (size_t si = 0; si < segments_.size(); ++si) {
        Segment&   seg       = segments_[si];
        const bool is_active = (si + 1 == segments_.size());
        const std::vector<uint8_t> bytes = read_file_(seg.path);

        WalHeader h;
        if (!decode_header(bytes.data(), bytes.size(), h))
            throw std::runtime_error("WAL segment header invalid: " + seg.path);
        if (h.dim != dim_ || h.metric != metric_)
            throw std::runtime_error("WAL config mismatch: " + seg.path);

        size_t off  = kWalHeaderSize;
        bool   torn = false;
        while (off < bytes.size()) {
            WalRecord    rec;
            size_t       consumed = 0;
            const DecodeStatus st =
                decode_record(bytes.data() + off, bytes.size() - off, rec, consumed);

            if (st == DecodeStatus::Ok) {
                on_record(rec);
                if (seg.first_lsn == 0) seg.first_lsn = rec.lsn;
                seg.last_lsn = rec.lsn;
                if (rec.lsn > highest_lsn_) highest_lsn_ = rec.lsn;
                off += consumed;
                continue;
            }

            if (!is_active)
                throw std::runtime_error("WAL corruption in sealed segment: " + seg.path);

            // Torn tail: cut back to the last good boundary for a clean append point.
            sys_check(::ftruncate(active_fd_, static_cast<off_t>(off)) == 0,
                      "ftruncate torn tail");
            sys_check(::fsync(active_fd_) == 0, "fsync after truncate");
            active_bytes_ = off;
            torn          = true;
            break;
        }
        if (is_active && !torn) active_bytes_ = bytes.size();
    }
}

void Wal::append(const WalRecord& r) {
    const std::vector<uint8_t> frame = encode_record(r);  // pure; no shared state

    std::lock_guard<std::mutex> g(io_mu_);
    sys_check(active_fd_ >= 0, "WAL not open");

    // Seal-early rotate; the "has a record" guard avoids looping on an oversized one.
    if (active_bytes_ > kWalHeaderSize && active_bytes_ + frame.size() > seg_max_)
        rotate_();

    write_at_(active_fd_, static_cast<long long>(active_bytes_), frame.data(), frame.size());
    active_bytes_ += frame.size();

    Segment& seg = segments_.back();
    if (seg.first_lsn == 0) seg.first_lsn = r.lsn;
    seg.last_lsn = r.lsn;
    if (r.lsn > highest_lsn_) highest_lsn_ = r.lsn;
}

uint64_t Wal::sync() {
    // Capture the fd and high-water under the lock, then fsync the *dup* with the
    // lock released so concurrent appends (and even a rotation closing the original
    // fd) proceed. The dup is an independent fd to the same file; sealed segments
    // were already fsync'd at rotation, so every record <= `target` is durable.
    int      dupfd;
    uint64_t target;
    {
        std::lock_guard<std::mutex> g(io_mu_);
        sys_check(active_fd_ >= 0, "WAL not open");
        dupfd = ::dup(active_fd_);
        sys_check(dupfd >= 0, "dup WAL fd");
        target = highest_lsn_;
    }
    const int rc    = ::fsync(dupfd);
    const int saved = errno;
    ::close(dupfd);
    errno = saved;
    sys_check(rc == 0, "fsync WAL");
    return target;
}

void Wal::truncate(uint64_t upto_lsn) {
    std::lock_guard<std::mutex> g(io_mu_);
    std::vector<Segment> survivors;
    survivors.reserve(segments_.size());
    bool removed = false;

    for (size_t i = 0; i < segments_.size(); ++i) {
        Segment&   seg       = segments_[i];
        const bool is_active = (i + 1 == segments_.size());
        if (!is_active && seg.sealed && seg.last_lsn != 0 && seg.last_lsn <= upto_lsn) {
            sys_check(::unlink(seg.path.c_str()) == 0, "unlink WAL segment");
            removed = true;
        } else {
            survivors.push_back(seg);
        }
    }

    segments_.swap(survivors);
    if (removed) fsync_dir_();
}

}
