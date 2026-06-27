#include "test.h"

#include "distance.h"

#include <cmath>

// Stage 1 — distance metrics. Hand-checked cases (the expected value is computed
// in the comment so you can verify the test itself, not just trust it). Floats
// aren't exact, so non-integer results are compared with a tolerance.

namespace {
bool near(float a, float b, float eps = 1e-4f) {
    return std::fabs(a - b) <= eps;
}
}  // namespace

// ---- l2_squared: sum of squared differences --------------------------------

TEST(l2_identical_is_zero) {
    float a[] = {1.0f, 2.0f, 3.0f};
    EXPECT(near(vdb::l2_squared(a, a, 3), 0.0f));
}

TEST(l2_known_value) {
    float a[] = {1.0f, 2.0f, 3.0f};
    float b[] = {4.0f, 6.0f, 8.0f};
    // diffs (3,4,5) -> 9 + 16 + 25 = 50
    EXPECT(near(vdb::l2_squared(a, b, 3), 50.0f));
}

TEST(l2_one_dimension) {
    float a[] = {3.0f};
    float b[] = {0.0f};
    EXPECT(near(vdb::l2_squared(a, b, 1), 9.0f));  // 3^2
}

TEST(l2_is_symmetric) {
    float a[] = {2.0f, -1.0f, 0.5f, 4.0f};
    float b[] = {-3.0f, 1.0f, 2.0f, 0.0f};
    EXPECT(near(vdb::l2_squared(a, b, 4), vdb::l2_squared(b, a, 4)));
}

TEST(l2_is_nonnegative) {
    float a[] = {-5.0f, 7.0f, -2.0f};
    float b[] = {1.0f, -1.0f, 9.0f};
    EXPECT(vdb::l2_squared(a, b, 3) >= 0.0f);
}

// ---- neg_inner_product: -(a . b), so "smaller = closer" ---------------------

TEST(neg_ip_known_value) {
    float a[] = {1.0f, 2.0f, 3.0f};
    float b[] = {4.0f, 5.0f, 6.0f};
    // dot = 4 + 10 + 18 = 32  ->  negated = -32
    EXPECT(near(vdb::neg_inner_product(a, b, 3), -32.0f));
}

TEST(neg_ip_orthogonal_is_zero) {
    float a[] = {1.0f, 0.0f, 0.0f};
    float b[] = {0.0f, 1.0f, 0.0f};
    EXPECT(near(vdb::neg_inner_product(a, b, 3), 0.0f));
}

TEST(neg_ip_is_symmetric) {
    float a[] = {2.0f, -3.0f, 4.0f};
    float b[] = {1.0f, 0.5f, -2.0f};
    EXPECT(near(vdb::neg_inner_product(a, b, 3),
                vdb::neg_inner_product(b, a, 3)));
}

// ---- cosine_distance: 1 - (a . b)/(|a||b|), inputs NOT pre-normalized -------

TEST(cosine_same_direction_is_zero) {
    float a[] = {1.0f, 0.0f};
    float b[] = {3.0f, 0.0f};  // same direction, different magnitude
    EXPECT(near(vdb::cosine_distance(a, b, 2), 0.0f));
}

TEST(cosine_orthogonal_is_one) {
    float a[] = {1.0f, 0.0f};
    float b[] = {0.0f, 1.0f};
    EXPECT(near(vdb::cosine_distance(a, b, 2), 1.0f));
}

TEST(cosine_opposite_is_two) {
    float a[] = {1.0f, 0.0f};
    float b[] = {-1.0f, 0.0f};
    EXPECT(near(vdb::cosine_distance(a, b, 2), 2.0f));
}

TEST(cosine_is_magnitude_invariant) {
    float a[] = {1.0f, 2.0f};
    float b[] = {2.0f, 4.0f};  // = 2*a, identical direction
    EXPECT(near(vdb::cosine_distance(a, b, 2), 0.0f));
}

TEST(cosine_forty_five_degrees) {
    float a[] = {1.0f, 0.0f};
    float b[] = {1.0f, 1.0f};
    // cos = 1 / (1 * sqrt(2)) = 0.70710678  ->  dist = 0.29289322
    EXPECT(near(vdb::cosine_distance(a, b, 2), 0.29289322f));
}

// ---- metric_fn: the dispatcher returns the matching function ----------------

TEST(metric_fn_dispatches_correctly) {
    float a[] = {1.0f, 2.0f, 3.0f};
    float b[] = {4.0f, 5.0f, 6.0f};

    auto l2 = vdb::metric_fn(vdb::Metric::L2);
    auto ip = vdb::metric_fn(vdb::Metric::InnerProduct);
    auto cos = vdb::metric_fn(vdb::Metric::Cosine);

    EXPECT(near(l2(a, b, 3), vdb::l2_squared(a, b, 3)));
    EXPECT(near(ip(a, b, 3), vdb::neg_inner_product(a, b, 3)));
    EXPECT(near(cos(a, b, 3), vdb::cosine_distance(a, b, 3)));
}
