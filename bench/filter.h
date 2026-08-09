#pragma once

// PR 14: post- vs pre-filter crossover on SIFT1M, and the cost-model planner that
// picks between them (see docs/plans/PR_STACK.md #14, docs/design/METADATA_DETAILS.md
// §1.4's U-curve). Measures both strategies across a selectivity sweep, under two
// label distributions — uniform-random (independent of embedding position) and
// k-means-cluster-correlated (the realistic case, per §1.2's "correlation footgun") —
// and calibrates vdb::FilterPlanner from the results.
// Invoked as `run_bench filter [data_dir]`.
void run_filter_bench(const char* data_dir);
