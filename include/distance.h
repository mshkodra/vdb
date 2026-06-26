#pragma once
#include <cstddef>

#include "vdb_types.h"

// Stage 1: distance metrics. These are the correctness primitives every index is
// built on. Start with a clear scalar implementation; optimize with SIMD later
// (see ROADMAP Stage 1 "Resources").

namespace vdb {

// Squared Euclidean (L2^2). Monotonic with L2 but avoids the sqrt.
float l2_squared(const float* a, const float* b, size_t dim);

// Negated inner product, so "smaller = closer" holds like the other metrics.
float neg_inner_product(const float* a, const float* b, size_t dim);

// Cosine distance = 1 - cos_similarity. Assumes inputs are NOT pre-normalized.
float cosine_distance(const float* a, const float* b, size_t dim);

enum class Metric { L2, InnerProduct, Cosine };

// Returns the DistanceFn for a metric (used to configure an index).
DistanceFn metric_fn(Metric m);

}  // namespace vdb
