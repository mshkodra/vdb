#include <vdb.h>

#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

int main(int argc, char** argv) {
    size_t N   = (argc > 1) ? std::atol(argv[1]) : 60000;
    size_t dim = (argc > 2) ? std::atol(argv[2]) : 128;

    VDBConfig cfg;
    cfg.hnsw.dim = dim;
    VDB db(cfg);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    std::vector<float> v(dim);

    for (size_t i = 0; i < N; ++i) {
        for (auto& x : v) x = u(rng);
        db.insert(v.data());
        if ((i + 1) % 5000 == 0) {
            std::printf("  inserted %zu / %zu\n", i + 1, N);
            std::fflush(stdout);
        }
    }

    std::printf("done: %zu vectors of dim %zu (db.size()=%zu)\n", N, dim, db.size());
    return 0;
}
