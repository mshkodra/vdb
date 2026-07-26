#include "distance.h"

namespace vdb {

// The free functions delegate to the functors so the loop lives in exactly one place
// (the header). They are only used off the hot path, so the extra call is harmless.
float l2_squared(const float* a, const float* b, size_t dim) {
    return L2{}(a, b, dim);
}

float neg_inner_product(const float* a, const float* b, size_t dim) {
    return InnerProduct{}(a, b, dim);
}

float cosine_distance(const float* a, const float* b, size_t dim) {
    return Cosine{}(a, b, dim);
}

DistanceFn metric_fn(Metric m) {
    switch (m) {
        case Metric::L2:           return &l2_squared;
        case Metric::InnerProduct: return &neg_inner_product;
        case Metric::Cosine:       return &cosine_distance;
    }
    return &l2_squared;
}

}
