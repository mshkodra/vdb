#include "concurrency.h"

#include "bench.h"

#include "distance.h"
#include "durable_vdb.h"
#include "vdb.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <unistd.h>

using namespace vdb;
using bench::clk;

namespace {

std::vector<float> random_flat(size_t n, size_t dim, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    std::vector<float> v(n * dim);
    for (auto& x : v) x = u(rng);
    return v;
}

std::string fresh_dir() {
    static std::atomic<int> ctr{0};
    auto dir = std::filesystem::temp_directory_path() /
               ("vdb_bench_" + std::to_string(::getpid()) + "_" +
                std::to_string(ctr.fetch_add(1)));
    std::filesystem::remove_all(dir);
    return dir.string();
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

double secs(clk::time_point a, clk::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}

struct PolicyRow {
    const char*         name;
    DurableVDB::Policy  policy;
};

// Wall-clock durable-insert throughput (writes/s), median over `reps`, for a policy
// and writer-thread count. Fresh data dir each run so no replay carries over.
double durable_insert_tput(DurableVDB::Policy policy, size_t threads,
                           const std::vector<float>& data, size_t N, size_t dim,
                           IndexKind kind, int reps) {
    std::vector<double> tputs;
    for (int r = 0; r < reps; ++r) {
        const std::string dir = fresh_dir();
        VDBConfig cfg;
        cfg.kind   = kind;
        cfg.dim    = dim;
        cfg.metric = Metric::L2;
        DurableVDB::Options opts;
        opts.policy            = policy;
        opts.flush_interval_ms = 10;         // Periodic flush cadence
        opts.checkpoint_ops    = 1u << 30;   // no checkpoint mid-measurement

        double sec;
        {
            DurableVDB          db(cfg, dir, opts);
            std::atomic<size_t> next{0};
            auto worker = [&] {
                size_t i;
                while ((i = next.fetch_add(1)) < N) db.insert(&data[i * dim]);
            };
            const auto t0 = clk::now();
            std::vector<std::thread> ts;
            for (size_t t = 0; t < threads; ++t) ts.emplace_back(worker);
            for (auto& t : ts) t.join();
            sec = secs(t0, clk::now());
        }
        std::filesystem::remove_all(dir);
        tputs.push_back(N / sec);
    }
    return median(std::move(tputs));
}

// Wall-clock search throughput (QPS) over a prebuilt VDB, `readers` threads sharing a
// fixed pool of Q queries, median over `reps`.
double search_qps(const VDB& db, const std::vector<float>& queries, size_t Q,
                  size_t dim, size_t K, size_t readers, int reps) {
    std::vector<double> out;
    for (int r = 0; r < reps; ++r) {
        std::atomic<size_t> next{0};
        auto worker = [&] {
            size_t i;
            while ((i = next.fetch_add(1)) < Q)
                bench::do_not_optimize(db.search(&queries[i * dim], K));
        };
        const auto t0 = clk::now();
        std::vector<std::thread> ts;
        for (size_t t = 0; t < readers; ++t) ts.emplace_back(worker);
        for (auto& t : ts) t.join();
        out.push_back(Q / secs(t0, clk::now()));
    }
    return median(std::move(out));
}

}  // namespace

void run_concurrency() {
    std::printf("\n== Stage 7: concurrency & durability ==\n");
    std::printf("(CSV rows are grep-able: `run_bench concurrency | grep ^CSV`)\n");

    const size_t counts[] = {1, 2, 4, 8};

    // ---- A. Group-commit fsync amortization -------------------------------------
    // Brute keeps per-insert CPU trivial, so the fsync — not graph work — is the
    // bottleneck. This isolates the durability cost. PerOpSync @1 thread is the
    // per-op-fsync floor; the curve rising with threads is one fsync covering many
    // writers (group commit). Periodic never waits per op → the no-durability ceiling.
    {
        const size_t dim = 32, N = 2000;
        auto data = random_flat(N, dim, 1);
        const PolicyRow policies[] = {
            {"PerOpSync", DurableVDB::Policy::PerOpSync},
            {"Periodic", DurableVDB::Policy::Periodic},
        };

        std::printf("\n-- A. durable insert throughput (Brute; fsync-bound; N=%zu) --\n", N);
        std::printf("CSV,section,policy,threads,inserts,writes_per_sec\n");
        for (const auto& p : policies)
            for (size_t th : counts)
                std::printf("CSV,durable_insert,%s,%zu,%zu,%.0f\n", p.name, th, N,
                            durable_insert_tput(p.policy, th, data, N, dim,
                                                IndexKind::Brute, 3));
    }

    // ---- B. Search QPS vs reader threads ----------------------------------------
    // Readers take VDB::mu_ shared; shared locks don't block one another, so QPS
    // should scale roughly with cores until memory-bandwidth bound.
    {
        const size_t dim = 32, N = 20000, Q = 20000, K = 10;
        auto data    = random_flat(N, dim, 2);
        auto queries = random_flat(Q, dim, 3);

        VDBConfig cfg;
        cfg.kind   = IndexKind::HNSW;
        cfg.dim    = dim;
        cfg.metric = Metric::L2;
        VDB db(cfg);
        for (size_t i = 0; i < N; ++i) db.insert(&data[i * dim]);

        std::printf("\n-- B. search QPS vs reader threads (HNSW, N=%zu, K=%zu) --\n", N, K);
        std::printf("CSV,section,threads,queries,qps,speedup\n");
        double base = 0.0;
        for (size_t th : counts) {
            const double qps = search_qps(db, queries, Q, dim, K, th, 3);
            if (th == 1) base = qps;
            std::printf("CSV,search_qps,%zu,%zu,%.0f,%.2f\n", th, Q, qps, qps / base);
        }
    }

    // ---- C. Mixed read/write contention -----------------------------------------
    // One writer stream (Periodic, so writes aren't fsync-bound) against a growing
    // reader pool, to see how the write's brief exclusive phases and the readers'
    // shared holds interact. Reports both write throughput and read QPS.
    {
        const size_t dim = 32, N = 8000, Q = 40000, K = 10;
        auto data    = random_flat(N, dim, 4);
        auto queries = random_flat(Q, dim, 5);

        std::printf("\n-- C. mixed: 2 writers + R readers (HNSW, Periodic) --\n");
        std::printf("CSV,section,readers,writes_per_sec,read_qps\n");
        for (size_t R : counts) {
            const std::string dir = fresh_dir();
            VDBConfig cfg;
            cfg.kind   = IndexKind::HNSW;
            cfg.dim    = dim;
            cfg.metric = Metric::L2;
            DurableVDB::Options opts;
            opts.policy            = DurableVDB::Policy::Periodic;
            opts.flush_interval_ms = 10;
            opts.checkpoint_ops    = 1u << 30;

            double w_tput, r_qps;
            {
                DurableVDB          db(cfg, dir, opts);
                std::atomic<size_t> wnext{0}, rdone{0};
                std::atomic<bool>   writers_done{false};

                auto writer = [&] {
                    size_t i;
                    while ((i = wnext.fetch_add(1)) < N) db.insert(&data[i * dim]);
                };
                auto reader = [&](unsigned seed) {
                    std::mt19937 rng(seed);
                    std::uniform_int_distribution<size_t> pick(0, Q - 1);
                    size_t did = 0;
                    while (!writers_done.load(std::memory_order_relaxed)) {
                        bench::do_not_optimize(db.search(&queries[pick(rng) * dim], K));
                        ++did;
                    }
                    rdone.fetch_add(did);
                };

                const auto t0 = clk::now();
                std::vector<std::thread> ws, rs;
                for (size_t t = 0; t < 2; ++t) ws.emplace_back(writer);
                for (size_t t = 0; t < R; ++t) rs.emplace_back(reader, 100u + (unsigned)t);
                for (auto& t : ws) t.join();
                const double wsec = secs(t0, clk::now());
                writers_done.store(true, std::memory_order_relaxed);
                for (auto& t : rs) t.join();

                w_tput = N / wsec;
                r_qps  = rdone.load() / wsec;  // reader work done during the write window
            }
            std::filesystem::remove_all(dir);
            std::printf("CSV,mixed,%zu,%.0f,%.0f\n", R, w_tput, r_qps);
        }
    }
}
