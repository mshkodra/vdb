#pragma once

// Phase B: hybrid search (BM25 + RRF fusion, docs/plans/HYBRID_SEARCH_PLAN.md, kept
// local) benchmark/demo driver, sibling to filter.cpp (Phase A). Reuses SIFT1M's
// vectors (already downloaded for filter.cpp/sift.cpp — no new dataset) and attaches
// a synthetic Minecraft-themed "description" (Text) + "category" (Tag) to each row,
// the same way filter.cpp attaches a synthetic `rank` label to the same vectors.
// Invoked as `run_bench hybrid [data_dir]`.
void run_hybrid_bench(const char* data_dir);

// Interactive REPL over the corpus run_hybrid_bench() builds and caches — loads the
// cached snapshot (fails with a pointer to `run_bench hybrid` if it doesn't exist
// yet; never builds one itself) and lets you type queries at a terminal prompt.
// Invoked as `run_bench hybrid-repl [data_dir]`.
void run_hybrid_repl(const char* data_dir);
