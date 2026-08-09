// PR 14: the closed-form cost-model planner (docs/plans/PR_STACK.md #14) that picks
// post-filter vs pre-filter from a calibration fit against bench/filter.cpp's
// measurements. Unit tests here exercise the model in isolation, on synthetic data —
// the real calibration numbers come from the SIFT1M benchmark run, not from tests.

#include "test.h"

#include "filter_planner.h"

#include <cmath>

using namespace vdb;

TEST(fit_through_origin_recovers_exact_slope_on_noise_free_data) {
    std::vector<CalibrationPoint> pts = {{1.0, 3.0}, {2.0, 6.0}, {10.0, 30.0}, {100.0, 300.0}};
    EXPECT(std::abs(fit_through_origin(pts) - 3.0) < 1e-9);
}

TEST(fit_through_origin_least_squares_on_noisy_data) {
    // y = 2x with noise; least squares through the origin should land close to 2.
    std::vector<CalibrationPoint> pts = {{1.0, 2.1}, {2.0, 3.9}, {4.0, 8.2}, {8.0, 15.8}};
    const double c = fit_through_origin(pts);
    EXPECT(c > 1.8 && c < 2.2);
}

TEST(fit_through_origin_empty_or_zero_x_returns_zero) {
    EXPECT(fit_through_origin({}) == 0.0);
    EXPECT(fit_through_origin({{0.0, 5.0}, {0.0, 9.0}}) == 0.0);
}

TEST(calibrate_fits_post_and_pre_independently) {
    std::vector<CalibrationPoint> post_pts = {{100.0, 500.0}, {200.0, 1000.0}};  // c=5
    std::vector<CalibrationPoint> pre_pts  = {{50.0, 100.0}, {100.0, 200.0}};    // c=2
    FilterCalibration calib = calibrate(post_pts, pre_pts);
    EXPECT(std::abs(calib.c_index - 5.0) < 1e-9);
    EXPECT(std::abs(calib.c_scan - 2.0) < 1e-9);
}

TEST(predicted_costs_match_the_documented_formulas) {
    FilterCalibration calib{/*c_index=*/2.0, /*c_scan=*/3.0};
    const size_t N = 1000, dim = 128, K = 10;
    const double s = 0.1;

    // want = K + N*(1-s) = 10 + 1000*0.9 = 910; post cost = c_index * want
    EXPECT(std::abs(predicted_post_cost_us(calib, N, K, s) - 2.0 * 910.0) < 1e-6);

    // scan = N*s*dim = 1000*0.1*128 = 12800; pre cost = c_scan * scan
    EXPECT(std::abs(predicted_pre_cost_us(calib, N, dim, s) - 3.0 * 12800.0) < 1e-6);
}

TEST(plan_strategy_picks_pre_filter_for_a_selective_predicate_with_cheap_scan) {
    // Index search is expensive per unit of `want`; scanning is cheap per element.
    // At high selectivity (small s), pre-filter's small allowlist should win easily.
    FilterCalibration calib{/*c_index=*/1.0, /*c_scan=*/0.001};
    const size_t N = 1'000'000, dim = 128, K = 10;
    EXPECT(plan_strategy(calib, N, dim, K, 0.001) == FilterStrategy::Pre);
}

TEST(plan_strategy_picks_post_filter_when_the_predicate_barely_excludes_anything) {
    // At s close to 1, post-filter's over-fetch `want` shrinks toward K while
    // pre-filter would still scan almost the whole database.
    FilterCalibration calib{/*c_index=*/1.0, /*c_scan=*/1.0};
    const size_t N = 1'000'000, dim = 128, K = 10;
    EXPECT(plan_strategy(calib, N, dim, K, 0.99) == FilterStrategy::Post);
}

TEST(plan_strategy_is_deterministic_with_zero_calibration) {
    FilterCalibration calib{};  // both predicted costs are 0
    EXPECT(plan_strategy(calib, 1000, 128, 10, 0.5) == FilterStrategy::Pre);
}
