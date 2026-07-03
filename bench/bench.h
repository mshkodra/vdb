#pragma once
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <vector>

namespace bench {

using clk = std::chrono::steady_clock;

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
        s.front(),
        s[s.size() / 2],
        mean,
        std::sqrt(var),
        s.size(),
    };
}

}
