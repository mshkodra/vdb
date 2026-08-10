#pragma once
#include <cmath>
#include <cstddef>

#include "vdb_types.h"

namespace vdb {

// Stateless distance functors. Their operator() bodies live in this header (not a
// .cpp) on purpose: an index templated on one of these can *inline* the inner loop
// straight into its hot path and let the compiler auto-vectorize it — which a call
// through std::function (an opaque indirect call) blocks entirely. Each carries no
// state and is trivially default-constructible, so an index just holds one by value
// at zero size/lifetime cost.

struct L2 {
    float operator()(const float* a, const float* b, size_t dim) const {
        float total = 0;
        for (size_t i = 0; i < dim; ++i) {
            const float diff = a[i] - b[i];
            total += diff * diff;
        }
        return total;
    }
};

struct InnerProduct {
    float operator()(const float* a, const float* b, size_t dim) const {
        float total = 0;
        for (size_t i = 0; i < dim; ++i) total += a[i] * b[i];
        return -total;  // negated so "smaller is nearer" matches the other metrics
    }
};

// Quantized L2, for HNSWIndex<L2, int8_t> (see hnsw_index.h's `Elem` template
// parameter). Operates on int8_t-quantized vectors that share one caller-supplied
// scale (symmetric quantization: q_i = round(v_i / scale), no zero-point) — so the
// difference (a_i - b_i) in quantized space is exactly (true_a_i - true_b_i)/scale,
// and this returns the *unscaled* sum of squared differences: exactly
// true_squared_l2_distance / scale^2. That's a positive multiple of the real
// distance for every pair, so it preserves nearest-neighbour ranking on its own —
// the caller (HNSWIndex) only needs to multiply by scale^2 at the very end, to
// report a distance comparable in magnitude to plain float L2, not on every
// comparison in the hot loop.
//
// Accumulates in int32_t: each per-dimension term is at most (255)^2 = 65025
// (the largest possible int8 difference, -127 to 127), and SIFT's dim=128 means
// a worst-case sum of ~8.3M — nowhere near int32_t's ~2.1B ceiling, no overflow
// risk for any dimension count this project uses.
struct L2Int8 {
    int32_t operator()(const int8_t* a, const int8_t* b, size_t dim) const {
        int32_t total = 0;
        for (size_t i = 0; i < dim; ++i) {
            const int32_t diff = static_cast<int32_t>(a[i]) - static_cast<int32_t>(b[i]);
            total += diff * diff;
        }
        return total;
    }
};

struct Cosine {
    float operator()(const float* a, const float* b, size_t dim) const {
        auto norm = [](const float* v, size_t n) {
            float t = 0;
            for (size_t i = 0; i < n; ++i) t += v[i] * v[i];
            return std::sqrt(t);
        };
        const float na = norm(a, dim);
        const float nb = norm(b, dim);
        if (na == 0.0f || nb == 0.0f) return 0.0f;  // undefined for a zero vector
        return 1.0f + InnerProduct{}(a, b, dim) / (na * nb);  // 1 - cos_sim
    }
};

enum class Metric { L2, InnerProduct, Cosine };

// Free-function forms + the std::function factory, kept for the microbenchmarks and
// unit tests that time or dispatch a metric directly. These are NOT on any index's
// hot path — the indexes are templated on the functors above — so the indirect call
// they carry is fine here.
float l2_squared(const float* a, const float* b, size_t dim);
float neg_inner_product(const float* a, const float* b, size_t dim);
float cosine_distance(const float* a, const float* b, size_t dim);

DistanceFn metric_fn(Metric m);

}
