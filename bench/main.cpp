#include "vdb.h"

#include <cstdio>

// Stage 4 lives here: a comparative harness (brute vs IVF vs HNSW) measuring
// recall@K, QPS, build time, and memory. For now it just confirms the library
// links. See ROADMAP Stage 0 (eval harness) and Stage 4 (analysis).
int main() {
    std::printf("vdb bench skeleton — implement the eval harness in Stage 4.\n");
    return 0;
}
