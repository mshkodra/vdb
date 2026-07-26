#pragma once

// SIFT1M driver: build brute / IVF / HNSW over the 1M-vector SIFT base set,
// sweep each method's recall/speed knob, score recall@K against the *shipped*
// ground truth (no 1M oracle rebuild), and emit a recall-vs-QPS CSV.
// Invoked as `run_bench sift [data_dir] [methods] [label]`.
//   methods : "all" (default) or a comma list of {brute,ivf,hnsw}; add the token
//             "rebuild" to bypass the snapshot cache and force a fresh build.
//   label   : tags the run — output goes to sift1m_<label>.csv with a `variant`
//             column, so before/after variants (e.g. stdfn vs inlined) don't clash.
// IVF/HNSW builds are cached to <data_dir>/cache/*.snap and loaded in seconds on
// re-run; brute (trivial) always rebuilds.
void run_sift(const char* data_dir, const char* methods, const char* label);
