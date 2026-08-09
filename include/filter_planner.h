#pragma once
#include <cstddef>
#include <vector>

namespace vdb {

enum class FilterStrategy { Post, Pre };

// Closed-form cost model for choosing between post-filter (VDB::search with a
// Predicate) and pre-filter (VDB::search_prefiltered). Post-filter's cost scales
// with the index's over-fetch width `want = K + N(1-s)` (collect_'s generalized
// margin, include/vdb.h); pre-filter's cost scales with the allowlist scan volume
// `N*s*dim` (prefilter_scan_, src/vdb.cpp). Both models are linear in their driving
// variable — exact for pre-filter (a literal linear scan over the allowlist), a
// simplification for post-filter over HNSW, whose graph search cost isn't strictly
// linear in `want`/ef (docs/design/METADATA_DETAILS.md §1.4). Good enough to pick a
// side; not a promise the predicted microsecond figure is exact.
struct FilterCalibration {
    double c_index = 0.0;  // us per unit of `want` (post-filter's over-fetch width)
    double c_scan  = 0.0;  // us per unit of `N*s*dim` (pre-filter's scan volume)
};

// One measured (driving_variable, time_us) sample from a benchmark sweep.
struct CalibrationPoint {
    double x;  // want (post-filter points) or N*s*dim (pre-filter points)
    double y;  // measured mean query latency, microseconds
};

// Least-squares fit of y = c*x through the origin: c = sum(x*y) / sum(x*x). No
// intercept term — zero work should cost zero time, and fitting one is exactly what
// would let a handful of samples produce a fixed cost that dominates at the small-x
// end of the sweep, which is exactly the regime the crossover lives in. Returns 0
// if `points` is empty or all x are 0.
double fit_through_origin(const std::vector<CalibrationPoint>& points);

// Fits c_index from `post_points` and c_scan from `pre_points` independently.
FilterCalibration calibrate(const std::vector<CalibrationPoint>& post_points,
                            const std::vector<CalibrationPoint>& pre_points);

// Predicted mean query cost (us) of each strategy over N live vectors of `dim`
// dimensions, K requested results, at predicate selectivity `s` (fraction matching,
// in [0,1]).
double predicted_post_cost_us(const FilterCalibration& c, size_t N, size_t K, double s);
double predicted_pre_cost_us(const FilterCalibration& c, size_t N, size_t dim, double s);

// Picks whichever strategy the calibrated model predicts is cheaper.
FilterStrategy plan_strategy(const FilterCalibration& c, size_t N, size_t dim, size_t K,
                             double s);

}
