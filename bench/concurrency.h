#pragma once

// Stage 7 concurrency & durability measurements, emitted as CSV:
//   A. group-commit fsync amortization — durable writes/s vs writer threads
//      (PerOpSync vs Periodic), on a cheap index so the fsync is the bottleneck.
//   B. search QPS vs reader threads — read scaling under the shared_mutex.
//   C. mixed read/write — write throughput and read QPS under contention.
// Run via `run_bench concurrency`. Called from bench/main.cpp.
void run_concurrency();
