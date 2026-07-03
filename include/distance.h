#pragma once
#include <cstddef>

#include "vdb_types.h"

namespace vdb {

float l2_squared(const float* a, const float* b, size_t dim);

float neg_inner_product(const float* a, const float* b, size_t dim);

float cosine_distance(const float* a, const float* b, size_t dim);

enum class Metric { L2, InnerProduct, Cosine };

DistanceFn metric_fn(Metric m);

}
