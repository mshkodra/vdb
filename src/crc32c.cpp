#include "crc32c.h"

namespace vdb {
namespace {

constexpr uint32_t kPoly = 0x82F63B78u;  // reflected Castagnoli

struct Crc32cTable {
    uint32_t t[256];
    Crc32cTable() {
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t crc = i;
            for (int k = 0; k < 8; ++k)
                crc = (crc & 1u) ? (crc >> 1) ^ kPoly : (crc >> 1);
            t[i] = crc;
        }
    }
};

const Crc32cTable kTable;

}  // namespace

uint32_t crc32c(const void* data, size_t len) {
    const auto* p = static_cast<const unsigned char*>(data);
    uint32_t crc = 0xFFFFFFFFu;
    for (size_t i = 0; i < len; ++i)
        crc = kTable.t[(crc ^ p[i]) & 0xFFu] ^ (crc >> 8);
    return crc ^ 0xFFFFFFFFu;
}

}
