#include "distance.h"

namespace vdb {

float l2_squared(const float* /*a*/, const float* /*b*/, size_t /*dim*/) {
    // TODO Stage 1: sum of squared differences.
    return 0.0f;
}

float neg_inner_product(const float* /*a*/, const float* /*b*/, size_t /*dim*/) {
    // TODO Stage 1: -sum(a[i] * b[i]).
    return 0.0f;
}

float cosine_distance(const float* /*a*/, const float* /*b*/, size_t /*dim*/) {
    // TODO Stage 1: 1 - dot(a,b) / (||a|| * ||b||).
    return 0.0f;
}

DistanceFn metric_fn(Metric m) {
    switch (m) {
        case Metric::L2:           return &l2_squared;
        case Metric::InnerProduct: return &neg_inner_product;
        case Metric::Cosine:       return &cosine_distance;
    }
    return &l2_squared;
}

}  // namespace vdb
