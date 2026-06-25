#include <snapshot.h>

#include <serialize.h>

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr uint32_t kMagic   = 0x56444253u;  // "VDBS"
constexpr uint16_t kVersion = 1;
constexpr const char* kPrefix = "snapshot.";

std::string snapshot_name(uint64_t id) {
    // Zero-pad so lexical order matches numeric order (handy when eyeballing a dir).
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%s%020llu", kPrefix,
                  static_cast<unsigned long long>(id));
    return buf;
}

// fsync a file by path (used for the snapshot file and its containing directory —
// fsync'ing the dir is what makes a rename() durable).
void fsync_path(const std::string& path) {
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) return;  // best effort
#ifdef __APPLE__
    if (::fcntl(fd, F_FULLFSYNC) != 0) ::fsync(fd);
#else
    ::fsync(fd);
#endif
    ::close(fd);
}

// Collect snapshot ids present in `dir`, newest first.
std::vector<uint64_t> list_snapshot_ids(const std::string& dir) {
    std::vector<uint64_t> ids;
    if (!fs::exists(dir)) return ids;
    for (const auto& entry : fs::directory_iterator(dir)) {
        const std::string name = entry.path().filename().string();
        if (name.rfind(kPrefix, 0) != 0) continue;            // not a snapshot file
        if (name.find(".tmp") != std::string::npos) continue;  // skip stale temps
        const std::string num = name.substr(std::strlen(kPrefix));
        try {
            ids.push_back(std::stoull(num));
        } catch (...) {
            // ignore malformed names
        }
    }
    std::sort(ids.rbegin(), ids.rend());  // newest (largest id) first
    return ids;
}

}  // namespace

void snapshot::save(const std::string& dir, uint64_t id, uint32_t dim, uint8_t metric,
                    uint64_t lsn_at_snapshot, uint64_t next_ext_id,
                    const std::vector<ExternalId>& int_to_ext, const HNSWIndex& idx) {
    fs::create_directories(dir);

    const std::string final_path = (fs::path(dir) / snapshot_name(id)).string();
    const std::string tmp_path   = final_path + ".tmp";

    {
        std::ofstream os(tmp_path, std::ios::binary | std::ios::trunc);
        if (!os) throw std::runtime_error("snapshot: cannot open temp file");

        // ---- header (32 bytes) ----
        uint8_t hdr[32] = {0};
        std::memcpy(hdr + 0, &kMagic, 4);
        std::memcpy(hdr + 4, &kVersion, 2);
        std::memcpy(hdr + 8, &dim, 4);
        hdr[12] = metric;
        os.write(reinterpret_cast<const char*>(hdr), sizeof(hdr));

        ser::write_pod(os, lsn_at_snapshot);
        ser::write_pod(os, next_ext_id);
        ser::write_pod(os, static_cast<uint64_t>(int_to_ext.size()));

        // ---- identity ----
        os.write(reinterpret_cast<const char*>(int_to_ext.data()),
                 static_cast<std::streamsize>(int_to_ext.size() * sizeof(ExternalId)));

        // ---- graph ----
        idx.save(os);

        if (!os) throw std::runtime_error("snapshot: write failed");
        os.flush();
    }

    // Durability: flush the bytes, then atomically swap into place, then flush the
    // directory entry. rename() is atomic on POSIX, so a reader/recovery never sees
    // a partially written snapshot under the real name.
    fsync_path(tmp_path);
    fs::rename(tmp_path, final_path);
    fsync_path(dir);

    // Reclaim older snapshots — the newest one supersedes them.
    for (uint64_t old : list_snapshot_ids(dir)) {
        if (old < id) {
            std::error_code ec;
            fs::remove(fs::path(dir) / snapshot_name(old), ec);
        }
    }
}

bool snapshot::load_latest(const std::string& dir, uint32_t expect_dim, uint8_t expect_metric,
                           uint64_t& lsn_at_snapshot, uint64_t& next_ext_id,
                           std::vector<ExternalId>& int_to_ext, HNSWIndex& idx,
                           uint64_t& snapshot_id) {
    for (uint64_t id : list_snapshot_ids(dir)) {
        const std::string path = (fs::path(dir) / snapshot_name(id)).string();
        std::ifstream is(path, std::ios::binary);
        if (!is) continue;

        uint8_t hdr[32] = {0};
        is.read(reinterpret_cast<char*>(hdr), sizeof(hdr));
        if (!is) continue;

        uint32_t magic;   std::memcpy(&magic, hdr + 0, 4);
        uint16_t version; std::memcpy(&version, hdr + 4, 2);
        uint32_t fdim;    std::memcpy(&fdim, hdr + 8, 4);
        const uint8_t fmetric = hdr[12];
        if (magic != kMagic || version != kVersion) continue;
        if (fdim != expect_dim || fmetric != expect_metric) {
            throw std::runtime_error("snapshot: dim/metric mismatch with config");
        }

        uint64_t node_count = 0;
        if (!ser::read_pod(is, lsn_at_snapshot) || !ser::read_pod(is, next_ext_id) ||
            !ser::read_pod(is, node_count)) {
            continue;  // corrupt header → try an older snapshot
        }

        int_to_ext.resize(node_count);
        is.read(reinterpret_cast<char*>(int_to_ext.data()),
                static_cast<std::streamsize>(node_count * sizeof(ExternalId)));
        if (!is) continue;

        try {
            idx.load(is);
        } catch (const std::exception&) {
            continue;  // graph blob corrupt → fall back to an older snapshot
        }

        snapshot_id = id;
        return true;
    }
    return false;
}
