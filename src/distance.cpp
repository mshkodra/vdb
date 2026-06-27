#include "distance.h"
#include <iostream>
namespace vdb {

float l2_squared(const float* a, const float* b, size_t dim) {
    float total = 0;
    for(size_t i = 0; i < dim; i++) {
        float diff = a[i] - b[i];
        total += (diff * diff);
    }
    return total;
}

float neg_inner_product(const float* a, const float* b, size_t dim) {
    float total = 0;
    for(size_t i = 0; i < dim; i++)
        total += (a[i] * b[i]);
    return -1 * total;
}

float cosine_distance(const float* a, const float* b, size_t dim) {
    auto isZero = [](const float* a, size_t dim) {
        for(size_t i = 0; i < dim; i++) {
            if(a[i] != 0) return false;
        }
        return true;
    };
    if(isZero(a, dim) || isZero(b, dim)) {
        std::cout << "Error\n Division by Zero";
        return 0;
    }

    auto norm = [](const float* a, size_t dim) {
        float total = 0;
        for(size_t i = 0; i < dim; i++) total += (a[i] * a[i]);
        return sqrtf(total);
    };
    float num = neg_inner_product(a, b, dim) / (norm(a, dim) * norm(b, dim));

    return 1 + num;
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
