#pragma once

// Stage 4 comparative driver: brute vs IVF vs HNSW on one shared
// (data, queries, ground-truth), sweeping each method's recall/speed knob and
// emitting a tidy recall-vs-QPS CSV. Called from bench/main.cpp.
void run_compare();

// Diagnostic: HNSW clustered recall variance across RNG seeds.
void run_seed_study();
