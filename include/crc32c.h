#pragma once

#include <cstddef>
#include <cstdint>

// CRC32C (Castagnoli, polynomial 0x1EDC6F41 / reflected 0x82F63B78).
//
// Why CRC32C and not CRC32 (zlib): CRC32C is the checksum used by storage
// engines (e.g. RocksDB, ext4 metadata, SCTP) precisely because modern x86 and
// ARM expose it as a single hardware instruction. We use a portable software,
// table-driven implementation here — correctness first; the hardware path is a
// drop-in optimization later.
//
// The function is *chainable*: pass the previous return value as `crc` to keep
// accumulating over multiple buffers, i.e.
//     crc32c(crc32c(0, a, la), b, lb) == crc32c(0, a ++ b)
// The internal ~crc at entry/exit is what makes that identity hold (standard
// reflected-CRC convention). Always seed the first call with 0.
namespace crc {

inline uint32_t crc32c(uint32_t crc, const void* data, size_t len) {
    static uint32_t table[256];
    static bool initialized = false;
    if (!initialized) {
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t c = i;
            for (int k = 0; k < 8; ++k) {
                c = (c & 1) ? (0x82F63B78u ^ (c >> 1)) : (c >> 1);
            }
            table[i] = c;
        }
        initialized = true;  // C++11 guarantees thread-safe static init
    }

    const auto* p = static_cast<const uint8_t*>(data);
    crc = ~crc;
    for (size_t i = 0; i < len; ++i) {
        crc = table[(crc ^ p[i]) & 0xFFu] ^ (crc >> 8);
    }
    return ~crc;
}

}  // namespace crc
