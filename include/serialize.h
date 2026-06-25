#pragma once

#include <cstdint>
#include <cstring>
#include <istream>
#include <ostream>
#include <vector>

// Minimal binary (de)serialization helpers shared by the WAL and the snapshot
// writer. These write fixed-width PODs in the host's native byte order.
//
// Endianness note: VDB persists in host-endian for simplicity — a database file
// is not portable across machines of differing endianness. Real engines either
// pin a byte order (RocksDB: little-endian) or tag the file; both the WAL and
// snapshot headers carry a magic number so at minimum a wrong-endian (or wrong
// file) open is detected rather than silently misread.
namespace ser {

// ---- stream helpers (used by the snapshot, which works over std::iostream) ----

template <class T>
void write_pod(std::ostream& os, const T& v) {
    os.write(reinterpret_cast<const char*>(&v), sizeof(T));
}

template <class T>
bool read_pod(std::istream& is, T& v) {
    is.read(reinterpret_cast<char*>(&v), sizeof(T));
    return static_cast<bool>(is) && is.gcount() == static_cast<std::streamsize>(sizeof(T));
}

// ---- buffer helpers (used by the WAL, which frames records into byte buffers
//      so it can checksum and write them with a single syscall) ----

template <class T>
void put_pod(std::vector<uint8_t>& buf, const T& v) {
    const auto* p = reinterpret_cast<const uint8_t*>(&v);
    buf.insert(buf.end(), p, p + sizeof(T));
}

inline void put_bytes(std::vector<uint8_t>& buf, const void* data, size_t len) {
    const auto* p = static_cast<const uint8_t*>(data);
    buf.insert(buf.end(), p, p + len);
}

// Read a POD out of `buf` at `off`, advancing `off`. Returns false if the buffer
// is too short — the caller treats that as a corrupt/torn record.
template <class T>
bool get_pod(const std::vector<uint8_t>& buf, size_t& off, T& v) {
    if (off + sizeof(T) > buf.size()) return false;
    std::memcpy(&v, buf.data() + off, sizeof(T));
    off += sizeof(T);
    return true;
}

}  // namespace ser
