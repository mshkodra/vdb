#pragma once
// Minimal but rigorous microbenchmark helper. The goal is measurements stable
// enough to compare across implementation versions, so it does the things naive
// timing gets wrong:
//   * warm-up reps (discarded) so caches/branch-predictors are hot
//   * many reps, and we report the MIN — the least-disturbed run is the truest
//     signal of the code's cost; mean/median are inflated by OS scheduling noise
//   * a compiler barrier (do_not_optimize) so the work can't be deleted as dead
//   * the caller does a fixed CHUNK of work per rep, large enough to dwarf the
//     clock's resolution; divide the result by the chunk size for per-op cost
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <vector>

namespace bench {

using clk = std::chrono::steady_clock;

// Forces `value` to be treated as observable, so the compiler cannot optimize
// away the computation that produced it. (Google-Benchmark's trick; clang/gcc.)
template <typename T>
inline void do_not_optimize(const T& value) {
    asm volatile("" : : "r,m"(value) : "memory");
}

struct Stats {
    double min_ns;
    double median_ns;
    double mean_ns;
    double stddev_ns;
    size_t reps;
};

// Times `fn` (which should perform a fixed chunk of work) over `reps` samples
// after `warmup` discarded runs. Returns per-REP stats in nanoseconds; divide by
// the chunk size yourself to get per-op cost.
template <typename Fn>
Stats measure(Fn&& fn, int reps = 50, int warmup = 5) {
    for (int i = 0; i < warmup; ++i) fn();

    std::vector<double> s;
    s.reserve(reps);
    for (int i = 0; i < reps; ++i) {
        auto t0 = clk::now();
        fn();
        auto t1 = clk::now();
        s.push_back(std::chrono::duration<double, std::nano>(t1 - t0).count());
    }
    std::sort(s.begin(), s.end());

    double sum = 0.0;
    for (double x : s) sum += x;
    const double mean = sum / s.size();
    double var = 0.0;
    for (double x : s) var += (x - mean) * (x - mean);
    var /= s.size();

    return Stats{
        /*min_ns=*/s.front(),
        /*median_ns=*/s[s.size() / 2],
        /*mean_ns=*/mean,
        /*stddev_ns=*/std::sqrt(var),
        /*reps=*/s.size(),
    };
}

}  // namespace bench
