#pragma once
#include <cstddef>
#include <cstdint>
#include <functional>

// Common vocabulary types shared across the project.
//
// InternalId : a vector's physical position inside an index (an array offset).
//              Indexes speak this internally; it is NOT stable across compaction.
// ExternalId : the stable, user-facing key a caller uses to refer to a vector.
//              The VDB layer maps ExternalId <-> InternalId (see ROADMAP Stage 5).
using InternalId = uint32_t;
using ExternalId = uint64_t;

// A distance/similarity function over two `dim`-length float vectors.
// Smaller = closer (so inner-product/cosine are stored negated, by convention).
using DistanceFn = std::function<float(const float*, const float*, size_t)>;
