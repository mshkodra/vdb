#pragma once
#include <cstddef>
#include <cstdint>

namespace vdb {

// CRC32C (Castagnoli). Detects torn writes and at-rest corruption in the WAL.
uint32_t crc32c(const void* data, size_t len);

}
