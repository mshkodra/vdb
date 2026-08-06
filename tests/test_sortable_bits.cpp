#include "test.h"

#include "sortable_bits.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

using namespace vdb;

namespace {
constexpr int kIterations = 200000;

// This project builds with -ffast-math (-> -ffinite-math-only), under which
// std::numeric_limits<double>::infinity()/quiet_NaN() are undefined behavior — clang
// warns on the infinity case and silently returns garbage for both. sortable_bits()
// itself is pure bit manipulation and doesn't care, but *constructing* an Inf/NaN
// double to test it with has to go around the standard library entirely: build the
// IEEE-754 bit pattern directly and memcpy it in, never asking the FP unit or the
// optimizer to "believe in" the value.
double from_bits(uint64_t b) {
    double v;
    std::memcpy(&v, &b, sizeof(v));
    return v;
}
}

// --- int64_t -----------------------------------------------------------------

TEST(sortable_bits_int64_known_anchors) {
    // The whole point of the transform: signed order becomes unsigned order.
    EXPECT(sortable_bits(std::numeric_limits<int64_t>::min()) == 0ULL);
    EXPECT(sortable_bits(std::numeric_limits<int64_t>::max()) ==
           std::numeric_limits<uint64_t>::max());
    EXPECT(sortable_bits(int64_t{0}) == 0x8000000000000000ULL);
    EXPECT(sortable_bits(int64_t{-1}) == 0x7FFFFFFFFFFFFFFFULL);
}

TEST(sortable_bits_int64_round_trips) {
    std::mt19937_64 rng(1);
    std::uniform_int_distribution<int64_t> dist(std::numeric_limits<int64_t>::min(),
                                                 std::numeric_limits<int64_t>::max());
    for (int i = 0; i < kIterations; ++i) {
        const int64_t v = dist(rng);
        EXPECT(from_sortable_bits<int64_t>(sortable_bits(v)) == v);
    }
}

TEST(sortable_bits_int64_preserves_order) {
    std::mt19937_64 rng(2);
    std::uniform_int_distribution<int64_t> dist(std::numeric_limits<int64_t>::min(),
                                                 std::numeric_limits<int64_t>::max());
    for (int i = 0; i < kIterations; ++i) {
        const int64_t a = dist(rng);
        const int64_t b = dist(rng);
        EXPECT((a < b) == (sortable_bits(a) < sortable_bits(b)));
        EXPECT((a == b) == (sortable_bits(a) == sortable_bits(b)));
    }
}

TEST(sortable_bits_int64_sorted_matches_native_sort) {
    std::mt19937_64 rng(3);
    std::uniform_int_distribution<int64_t> dist(std::numeric_limits<int64_t>::min(),
                                                 std::numeric_limits<int64_t>::max());
    std::vector<int64_t> values;
    for (int i = 0; i < 5000; ++i) values.push_back(dist(rng));
    values.push_back(std::numeric_limits<int64_t>::min());
    values.push_back(std::numeric_limits<int64_t>::max());
    values.push_back(0);

    std::vector<int64_t> by_native = values;
    std::sort(by_native.begin(), by_native.end());

    std::vector<int64_t> by_bits = values;
    std::sort(by_bits.begin(), by_bits.end(),
              [](int64_t x, int64_t y) { return sortable_bits(x) < sortable_bits(y); });

    EXPECT(by_native == by_bits);
}

// --- double --------------------------------------------------------------------

TEST(sortable_bits_double_known_anchors) {
    // Positive zero sits just above negative zero (both compare mathematically
    // equal, but the transform must place every real value on one side of them
    // consistently, and it breaks the -0.0/+0.0 tie the same way every time).
    EXPECT(sortable_bits(-0.0) < sortable_bits(0.0));
    EXPECT(sortable_bits(-1.0) < sortable_bits(-0.0));
    EXPECT(sortable_bits(0.0) < sortable_bits(1.0));
    EXPECT(sortable_bits(-2.0) < sortable_bits(-1.0));
    EXPECT(sortable_bits(1.0) < sortable_bits(2.0));

    const double neg_inf = from_bits(0xFFF0000000000000ULL);
    const double pos_inf = from_bits(0x7FF0000000000000ULL);
    EXPECT(sortable_bits(neg_inf) < sortable_bits(std::numeric_limits<double>::lowest()));
    EXPECT(sortable_bits(std::numeric_limits<double>::max()) < sortable_bits(pos_inf));
}

TEST(sortable_bits_double_round_trips) {
    std::mt19937_64 rng(4);
    std::uniform_real_distribution<double> dist(-1e300, 1e300);
    for (int i = 0; i < kIterations; ++i) {
        const double v = dist(rng);
        const double got = from_sortable_bits<double>(sortable_bits(v));
        // Bit-exact, not just numerically close: this is a bijection on the raw
        // 64-bit pattern, not a rounding-tolerant transform.
        uint64_t want_bits, got_bits;
        std::memcpy(&want_bits, &v, 8);
        std::memcpy(&got_bits, &got, 8);
        EXPECT(want_bits == got_bits);
    }
}

TEST(sortable_bits_double_preserves_order) {
    std::mt19937_64 rng(5);
    std::uniform_real_distribution<double> dist(-1e300, 1e300);
    for (int i = 0; i < kIterations; ++i) {
        const double a = dist(rng);
        const double b = dist(rng);
        EXPECT((a < b) == (sortable_bits(a) < sortable_bits(b)));
        EXPECT((a == b) == (sortable_bits(a) == sortable_bits(b)));
    }
}

TEST(sortable_bits_double_preserves_order_near_zero) {
    // The sign-flip vs. full-complement branch changes right at zero, so this is
    // exactly where a mask-selection bug would show up — worth a dedicated,
    // tightly-clustered random sample rather than trusting the wide-range test to
    // hit it by chance.
    std::mt19937_64 rng(6);
    std::uniform_real_distribution<double> dist(-1e-300, 1e-300);
    for (int i = 0; i < kIterations; ++i) {
        const double a = dist(rng);
        const double b = dist(rng);
        EXPECT((a < b) == (sortable_bits(a) < sortable_bits(b)));
    }
}

TEST(sortable_bits_double_sorted_matches_native_sort) {
    std::mt19937_64 rng(7);
    std::uniform_real_distribution<double> dist(-1e300, 1e300);
    std::vector<double> values;
    for (int i = 0; i < 5000; ++i) values.push_back(dist(rng));
    values.push_back(0.0);
    values.push_back(-0.0);
    values.push_back(std::numeric_limits<double>::lowest());
    values.push_back(std::numeric_limits<double>::max());
    values.push_back(std::numeric_limits<double>::min());  // smallest positive normal
    values.push_back(-std::numeric_limits<double>::min());
    values.push_back(std::numeric_limits<double>::denorm_min());
    values.push_back(-std::numeric_limits<double>::denorm_min());

    std::vector<double> by_native = values;
    std::sort(by_native.begin(), by_native.end());

    std::vector<double> by_bits = values;
    std::sort(by_bits.begin(), by_bits.end(),
              [](double x, double y) { return sortable_bits(x) < sortable_bits(y); });

    EXPECT(by_native == by_bits);
}

TEST(sortable_bits_double_nan_round_trips_without_a_defined_order) {
    // NaN has no total order under IEEE-754 — the transform still has to produce
    // *something* well-defined (a bijection on the bit pattern) rather than
    // undefined behavior; it just isn't claiming that "something" is meaningfully
    // ordered against other values. Assert only the bit-exact round trip, for a
    // couple of distinct NaN bit patterns (quiet and signaling both exist).
    const uint64_t patterns[] = {0x7FF8000000000000ULL,   // a quiet NaN
                                 0xFFF8000000000001ULL,   // a different quiet NaN
                                 0x7FF0000000000001ULL};  // a signaling NaN
    for (uint64_t nan_bits : patterns) {
        const double nan       = from_bits(nan_bits);
        const double got       = from_sortable_bits<double>(sortable_bits(nan));
        uint64_t     got_bits;
        std::memcpy(&got_bits, &got, 8);
        EXPECT(got_bits == nan_bits);
    }
}
