#pragma once
#include <cstddef>
#include <cstdint>
#include <functional>

using InternalId = uint32_t;
using ExternalId = uint64_t;

using DistanceFn = std::function<float(const float*, const float*, size_t)>;
