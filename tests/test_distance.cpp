#include "test.h"

#include "distance.h"

#include <cmath>

namespace {
bool near(float a, float b, float eps = 1e-4f) {
    return std::fabs(a - b) <= eps;
}
}

TEST(l2_identical_is_zero) {
    float a[] = {1.0f, 2.0f, 3.0f};
    EXPECT(near(vdb::l2_squared(a, a, 3), 0.0f));
}

TEST(l2_known_value) {
    float a[] = {1.0f, 2.0f, 3.0f};
    float b[] = {4.0f, 6.0f, 8.0f};
    EXPECT(near(vdb::l2_squared(a, b, 3), 50.0f));
}

TEST(l2_one_dimension) {
    float a[] = {3.0f};
    float b[] = {0.0f};
    EXPECT(near(vdb::l2_squared(a, b, 1), 9.0f));
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

TEST(neg_ip_known_value) {
    float a[] = {1.0f, 2.0f, 3.0f};
    float b[] = {4.0f, 5.0f, 6.0f};
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

TEST(cosine_same_direction_is_zero) {
    float a[] = {1.0f, 0.0f};
    float b[] = {3.0f, 0.0f};
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
    float b[] = {2.0f, 4.0f};
    EXPECT(near(vdb::cosine_distance(a, b, 2), 0.0f));
}

TEST(cosine_forty_five_degrees) {
    float a[] = {1.0f, 0.0f};
    float b[] = {1.0f, 1.0f};
    EXPECT(near(vdb::cosine_distance(a, b, 2), 0.29289322f));
}

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
