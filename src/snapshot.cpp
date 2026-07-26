#include "snapshot.h"

#include "distance.h"
#include "serialize.h"
#include "vdb.h"

#include <cerrno>
#include <cstring>
#include <filesystem>
#include <stdexcept>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace vdb {
namespace {

constexpr uint32_t kSnapMagic   = 0x56444253u;  // "VDBS"
constexpr uint16_t kSnapVersion = 1;

void sys_throw(const char* what) {
    throw std::runtime_error(std::string(what) + ": " + std::strerror(errno));
}

void write_file_atomic(const std::string& path, const std::vector<uint8_t>& bytes) {
    const std::string tmp = path + ".tmp";
    const int fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) sys_throw("open snapshot tmp");

    size_t off = 0;
    while (off < bytes.size()) {
        const ssize_t n = ::pwrite(fd, bytes.data() + off, bytes.size() - off,
                                   static_cast<off_t>(off));
        if (n <= 0) { ::close(fd); sys_throw("pwrite snapshot"); }
        off += static_cast<size_t>(n);
    }
    if (::fsync(fd) != 0) { ::close(fd); sys_throw("fsync snapshot"); }
    ::close(fd);

    if (::rename(tmp.c_str(), path.c_str()) != 0) sys_throw("rename snapshot");

    const auto dir = std::filesystem::path(path).parent_path();
    const int dfd = ::open(dir.empty() ? "." : dir.c_str(), O_RDONLY);
    if (dfd >= 0) {
        if (::fsync(dfd) != 0 && errno != EINVAL) { /* dir fsync unsupported; ignore */ }
        ::close(dfd);
    }
}

std::vector<uint8_t> read_whole_file(const std::string& path) {
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) sys_throw("open snapshot");
    struct stat st {};
    if (::fstat(fd, &st) != 0) { ::close(fd); sys_throw("fstat snapshot"); }

    std::vector<uint8_t> buf(static_cast<size_t>(st.st_size));
    size_t rd = 0;
    while (rd < buf.size()) {
        const ssize_t n = ::read(fd, buf.data() + rd, buf.size() - rd);
        if (n <= 0) break;
        rd += static_cast<size_t>(n);
    }
    ::close(fd);
    buf.resize(rd);
    return buf;
}

}  // namespace

void save_snapshot(const VDB& db, const std::string& path, uint64_t lsn) {
    std::vector<uint8_t> buf;
    put<uint32_t>(buf, kSnapMagic);
    put<uint16_t>(buf, kSnapVersion);
    put<uint32_t>(buf, static_cast<uint32_t>(db.config_.dim));
    put<uint8_t>(buf, static_cast<uint8_t>(db.config_.metric));
    put<uint64_t>(buf, lsn);

    put<uint64_t>(buf, db.next_ext_id_);
    put<uint64_t>(buf, db.live_count_);
    put<uint64_t>(buf, db.deleted_count_);

    put<uint64_t>(buf, db.int_to_ext_.size());
    for (ExternalId e : db.int_to_ext_) put<uint64_t>(buf, e);

    put<uint64_t>(buf, db.deleted_.size());
    for (bool d : db.deleted_) put<uint8_t>(buf, d ? 1 : 0);

    put<uint64_t>(buf, db.vectors_.size());
    for (const auto& v : db.vectors_) put_floats(buf, v);

    db.index_->serialize(buf);

    write_file_atomic(path, buf);
}

uint64_t load_snapshot(VDB& db, const std::string& path) {
    const std::vector<uint8_t> bytes = read_whole_file(path);
    Reader r(bytes.data(), bytes.size());

    if (r.get<uint32_t>() != kSnapMagic) throw std::runtime_error("snapshot bad magic");
    if (r.get<uint16_t>() != kSnapVersion) throw std::runtime_error("snapshot bad version");
    const uint32_t dim    = r.get<uint32_t>();
    const uint8_t  metric = r.get<uint8_t>();
    if (dim != db.config_.dim || metric != static_cast<uint8_t>(db.config_.metric))
        throw std::runtime_error("snapshot config mismatch");
    const uint64_t lsn = r.get<uint64_t>();

    db.next_ext_id_   = r.get<uint64_t>();
    db.live_count_    = r.get<uint64_t>();
    db.deleted_count_ = r.get<uint64_t>();

    const uint64_t n_ext = r.get<uint64_t>();
    db.int_to_ext_.resize(n_ext);
    for (uint64_t i = 0; i < n_ext; ++i) db.int_to_ext_[i] = r.get<uint64_t>();

    const uint64_t n_del = r.get<uint64_t>();
    db.deleted_.resize(n_del);
    for (uint64_t i = 0; i < n_del; ++i) db.deleted_[i] = r.get<uint8_t>() != 0;

    const uint64_t n_vec = r.get<uint64_t>();
    db.vectors_.resize(n_vec);
    for (uint64_t i = 0; i < n_vec; ++i) db.vectors_[i] = r.get_floats();

    // ext_to_int_ holds only live ids; rebuild it from the surviving internal nodes.
    db.ext_to_int_.clear();
    for (InternalId i = 0; i < db.int_to_ext_.size(); ++i)
        if (!db.deleted_[i]) db.ext_to_int_[db.int_to_ext_[i]] = i;

    db.index_ = VDB::make_index_(db.config_);
    db.index_->deserialize(r);

    return lsn;
}

}
