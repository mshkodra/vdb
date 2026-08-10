#include "filter_planner.h"

namespace vdb {

double fit_through_origin(const std::vector<CalibrationPoint>& points) {
    double sum_xy = 0.0, sum_xx = 0.0;
    for (const auto& p : points) {
        sum_xy += p.x * p.y;
        sum_xx += p.x * p.x;
    }
    if (sum_xx == 0.0) return 0.0;
    return sum_xy / sum_xx;
}

FilterCalibration calibrate(const std::vector<CalibrationPoint>& post_points,
                            const std::vector<CalibrationPoint>& pre_points) {
    return FilterCalibration{fit_through_origin(post_points), fit_through_origin(pre_points)};
}

double predicted_post_cost_us(const FilterCalibration& c, size_t N, size_t K, double s) {
    const double want = static_cast<double>(K) + static_cast<double>(N) * (1.0 - s);
    return c.c_index * want;
}

double predicted_pre_cost_us(const FilterCalibration& c, size_t N, size_t dim, double s) {
    const double scan = static_cast<double>(N) * s * static_cast<double>(dim);
    return c.c_scan * scan;
}

FilterStrategy plan_strategy(const FilterCalibration& c, size_t N, size_t dim, size_t K,
                             double s) {
    const double post = predicted_post_cost_us(c, N, K, s);
    const double pre  = predicted_pre_cost_us(c, N, dim, s);
    return post < pre ? FilterStrategy::Post : FilterStrategy::Pre;
}

FilterCalibration default_filter_calibration() {
    return FilterCalibration{1.859288, 0.001214};
}

}
