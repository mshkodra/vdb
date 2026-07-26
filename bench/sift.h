#pragma once

// SIFT1M driver: build brute / IVF / HNSW over the 1M-vector SIFT base set,
// sweep each method's recall/speed knob, score recall@K against the *shipped*
// ground truth (no 1M oracle rebuild), and emit a recall-vs-QPS CSV.
// Invoked as `run_bench sift [data_dir] [methods]`. `methods` is "all" (default)
// or a comma list of {brute,ivf,hnsw}; a subset appends to the existing CSV so
// an interrupted run can be resumed method-by-method without re-paying builds.
void run_sift(const char* data_dir, const char* methods);
