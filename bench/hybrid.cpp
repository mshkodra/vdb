#include "hybrid.h"

#include "bench.h"
#include "snapshot.h"
#include "vdb.h"

#include <sys/stat.h>  // mkdir for the snapshot cache dir

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

using namespace vdb;

namespace {

constexpr size_t DIM = 128;  // SIFT descriptors are 128-d — reusing filter.cpp's vectors

// ============================================================================
//  EXPERIMENT GRID — this block is yours to design (see CLAUDE.md); filled in
//  here to answer a direct "show me recall and QPS" ask, not because the earlier
//  "yours to design" note stopped applying — change these and predict before
//  re-running, same as filter.cpp's own sweep.
//
//  BM25 (search_text) is exact — MetadataStore::search_text is a closed-form
//  computation over postings, not an approximate search, so it has no recall
//  loss of its own regardless of which vector index the surrounding VDB happens
//  to use. The only place "recall" means anything in this stack is
//  search_hybrid's vector-ranker half: HNSW is approximate, so the RRF-fused
//  result can differ from what an exact vector index would have fused. Recall
//  here is measured the same way sift.cpp/filter.cpp already do it — HNSW's
//  output vs. a Brute-indexed oracle over the *same* corpus content (same seed,
//  same rows, only the index kind differs).
constexpr size_t N_CAP        = 50000;  // corpus size — SIFT1M has 1M, capped for a fast sweep
constexpr unsigned VOCAB_SEED = 11;
constexpr size_t K_DEFAULT    = 10;
constexpr size_t Q_TIMING     = 100;   // SIFT query vectors used for hybrid QPS/recall
constexpr size_t MEASURE_REPS = 30;    // repeated identical calls for the BM25-only timing
constexpr int    TIMING_REPS  = 2;     // best-of-N per distinct query, for the hybrid sweep

// Query terms at three "commonness" tiers, by construction: word-list index 0 is
// the most-frequently-sampled word (Zipfian weight 1/(index+1) in make_doc()), a
// higher index is rarer. VDB has no public term->doc_freq lookup (only
// MetadataStore does, and VDB doesn't expose meta_), so list position stands in
// for an exact document-frequency number here.
struct QueryTier {
    const char* label;
    std::string term;
};
const std::vector<QueryTier> kTextTiers = {
    {"common (kItems[0])", "sword"},
    {"mid (kItems[8])", "book"},
    {"rare (kItems[15])", "totem"},
};

// A mid-frequency adjective, present across every category — representative of
// an ordinary hybrid query rather than a best/worst case.
const std::string kHybridTerm = "legendary";
// depth = mult * K_DEFAULT — RRF's own over-fetch knob (VDB::search_hybrid's
// `depth` parameter). 1x is "no over-fetch beyond K"; 4x checks whether digging
// deeper into each ranker before fusing changes cost or recall materially.
const std::vector<size_t> kHybridDepthMult = {1, 4};
// ============================================================================

// ---- .fvecs reader (mirrors filter.cpp's/sift.cpp's own copy — each driver
//      keeps its own, the established convention here). Format: each record is
//      [int32 dim][dim float32 payload], no file header/count. ----
std::vector<float> read_fvecs(const std::string& path, size_t& n_out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "!! cannot open %s\n", path.c_str()); std::exit(1); }
    f.seekg(0, std::ios::end);
    const std::streamoff bytes = f.tellg();
    f.seekg(0, std::ios::beg);

    int32_t d = 0;
    f.read(reinterpret_cast<char*>(&d), 4);
    if (d <= 0) { std::fprintf(stderr, "!! bad dim in %s\n", path.c_str()); std::exit(1); }
    const size_t dim = static_cast<size_t>(d);
    const size_t rec = 4 + dim * 4;
    const size_t n   = static_cast<size_t>(bytes) / rec;

    std::vector<float> out(n * dim);
    f.seekg(0, std::ios::beg);
    for (size_t i = 0; i < n; ++i) {
        int32_t dd = 0;
        f.read(reinterpret_cast<char*>(&dd), 4);  // skip the repeated dim prefix
        f.read(reinterpret_cast<char*>(out.data() + i * dim),
               static_cast<std::streamsize>(dim) * 4);
    }
    n_out = n;
    return out;
}

// ---- Minecraft-themed synthetic corpus. Each word list below is ordered
//      common-to-rare and sampled with Zipfian weight 1/(index+1) — deliberate
//      term-frequency skew so BM25's idf has something real to differentiate (a
//      uniformly-sampled vocabulary would make every term equally "rare," a
//      degenerate case for testing idf at all). No real semantic tie to the SIFT
//      vectors these get attached to — this is for exercising the search_text/
//      search_hybrid mechanism, not for judging result *relevance*. ----
const std::vector<std::string> kItems = {
    "sword", "pickaxe", "shovel", "axe", "hoe", "bow", "shield", "torch",
    "bucket", "compass", "map", "potion", "book", "trident", "elytra", "totem",
};
const std::vector<std::string> kBlocks = {
    "dirt", "stone", "cobblestone", "sand", "gravel", "oak_log", "spruce_log",
    "glass", "bricks", "wool", "netherrack", "glowstone", "obsidian",
    "diamond_block", "emerald_block", "netherite_block",
};
const std::vector<std::string> kMobs = {
    "zombie", "skeleton", "spider", "creeper", "villager", "cow", "pig",
    "chicken", "wolf", "enderman", "piglin", "blaze", "ghast", "phantom",
    "wither", "ender_dragon",
};
const std::vector<std::string> kBiomes = {
    "plains", "forest", "desert", "mountains", "ocean", "swamp", "taiga",
    "jungle", "savanna", "tundra", "mushroom_fields", "mesa", "nether", "the_end",
};
const std::vector<std::string> kAdjectives = {
    "sturdy", "shiny", "rusty", "common", "golden", "glowing", "ancient",
    "cursed", "enchanted", "mysterious", "legendary", "obsidian", "diamond",
    "netherite",
};
const std::vector<std::string> kActions = {
    "mine", "craft", "build", "collect", "explore", "trade", "smelt", "brew",
    "enchant", "forge", "discover", "harvest",
};

const std::vector<std::string> kCategories = {"item", "block", "mob", "biome"};

// Sentence templates. "{N}" placeholders are filled positionally by make_doc()'s
// slots array: 0=adjective 1=item 2=block 3=mob 4=biome 5=action.
const std::vector<std::string> kTemplates = {
    "a {0} {1} found while exploring the {4}",
    "{5} the {3} near the {4} to get a {1}",
    "a {0} {2} block seen deep in the {4}",
    "{5} {2} to craft a {0} {1}",
    "the {3} dropped a {0} {1}",
    "legends speak of a {0} {1} hidden in the {4}",
};

// Zipfian pick: weight[i] = 1/(i+1), normalized — index 0 is the most common
// entry in `list`. Evokes the classic word-frequency law, not a rigorous fit to
// any real corpus.
size_t zipf_pick(size_t n, std::mt19937& rng) {
    std::vector<double> weights(n);
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        weights[i] = 1.0 / static_cast<double>(i + 1);
        sum += weights[i];
    }
    std::uniform_real_distribution<double> u(0.0, sum);
    double target = u(rng);
    for (size_t i = 0; i < n; ++i) {
        target -= weights[i];
        if (target <= 0.0) return i;
    }
    return n - 1;  // floating-point fallback; weights sum to `sum` exactly by construction
}

const std::string& zipf_word(const std::vector<std::string>& list, std::mt19937& rng) {
    return list[zipf_pick(list.size(), rng)];
}

std::string fill_template(const std::string& tmpl, const std::string slots[6]) {
    std::string out;
    out.reserve(tmpl.size());
    for (size_t i = 0; i < tmpl.size(); ++i) {
        if (tmpl[i] == '{' && i + 2 < tmpl.size() && tmpl[i + 2] == '}') {
            out += slots[tmpl[i + 1] - '0'];
            i += 2;
        } else {
            out += tmpl[i];
        }
    }
    return out;
}

struct MinecraftDoc {
    std::string description;
    std::string category;
};

MinecraftDoc make_doc(std::mt19937& rng) {
    const std::string slots[6] = {
        zipf_word(kAdjectives, rng), zipf_word(kItems, rng),  zipf_word(kBlocks, rng),
        zipf_word(kMobs, rng),       zipf_word(kBiomes, rng), zipf_word(kActions, rng),
    };
    const std::string& tmpl     = kTemplates[zipf_pick(kTemplates.size(), rng)];
    const std::string& category = kCategories[rng() % kCategories.size()];
    return {fill_template(tmpl, slots), category};
}

std::vector<AttrSpec> minecraft_schema() {
    return {{"description", AttrType::Text}, {"category", AttrType::Tag}};
}

// Builds (or loads a size-matching cache of) a corpus over `base`'s first `n`
// vectors, under index kind `kind`. Re-seeds the same RNG (VOCAB_SEED) every
// call, so an HNSW build and a Brute build produce byte-identical row content
// (same vector, same description, same category) — only the index structure
// differs, which is exactly what makes one a valid recall oracle for the other.
std::unique_ptr<VDB> build_or_load_corpus(IndexKind kind, const std::string& cache_path,
                                          const std::vector<float>& base, size_t n) {
    VDBConfig cfg;
    cfg.kind   = kind;
    cfg.dim    = DIM;
    cfg.metric = Metric::L2;
    cfg.schema = minecraft_schema();
    auto db = std::make_unique<VDB>(cfg);

    bool loaded = false;
    try {
        load_snapshot(*db, cache_path);
        // Guards against silently loading a cache built at a different N_CAP —
        // same footgun filter.cpp's own run_distribution() fixed for.
        loaded = (db->size() == n);
    } catch (const std::exception&) {
        // No cache yet, or it doesn't match this VDB's config — build fresh.
    }
    if (!loaded) db = std::make_unique<VDB>(cfg);  // discard a wrong-sized partial load

    if (loaded) {
        std::printf("  loaded cache %s (N=%zu)\n", cache_path.c_str(), db->size());
    } else {
        std::printf("  building (%s) over %zu vectors...\n",
                   kind == IndexKind::HNSW ? "HNSW" : "Brute", n);
        std::mt19937 rng(VOCAB_SEED);
        for (size_t i = 0; i < n; ++i) {
            const MinecraftDoc doc = make_doc(rng);
            Record              r;
            r.attrs = {attr_text(doc.description), attr_tag(doc.category)};
            db->insert(base.data() + i * DIM, r);
            if ((i + 1) % 10000 == 0) std::printf("    ... %zu / %zu inserted\n", i + 1, n);
        }
        save_snapshot(*db, cache_path, 0);
        std::printf("  cached -> %s\n", cache_path.c_str());
    }
    return db;
}

double percentile(std::vector<double>& sorted, double p) {
    if (sorted.empty()) return 0.0;
    size_t idx = static_cast<size_t>(std::ceil(p / 100.0 * sorted.size()));
    if (idx > 0) --idx;
    return sorted[std::min(idx, sorted.size() - 1)];
}

struct QStats {
    double qps, mean_us, p50_us, p95_us;
};

// Best-of-`reps` latency for each of `n` calls to fn(i) (i = 0..n-1) — same
// "min over reps, then aggregate across queries" shape filter.cpp's own
// time_queries() uses. `fn`'s return value only needs a .size() (for
// do_not_optimize, so the compiler can't elide the call); passing the same
// index every time (n identical calls) is exactly how the BM25-only timing
// below reuses this for "repeat one query N times" instead of "N distinct
// queries."
template <typename Fn>
QStats time_over(Fn&& fn, size_t n, int reps) {
    for (size_t i = 0; i < n; ++i) bench::do_not_optimize(fn(i).size());  // warm-up

    std::vector<double> per_us(n);
    for (size_t i = 0; i < n; ++i) {
        double best = std::numeric_limits<double>::max();
        for (int r = 0; r < reps; ++r) {
            auto t0  = bench::clk::now();
            auto res = fn(i);
            auto t1  = bench::clk::now();
            bench::do_not_optimize(res.size());
            best = std::min(best, std::chrono::duration<double, std::micro>(t1 - t0).count());
        }
        per_us[i] = best;
    }
    std::sort(per_us.begin(), per_us.end());
    double sum = 0.0;
    for (double x : per_us) sum += x;
    const double mean = sum / static_cast<double>(n);
    return {1e6 / mean, mean, percentile(per_us, 50), percentile(per_us, 95)};
}

void print_hits(const VDB& db, const char* label, const std::vector<Hit>& hits) {
    std::printf("\n[%s] %zu hits\n", label, hits.size());
    for (const auto& h : hits) {
        Record rec;
        db.get_metadata(h.id, rec);
        std::printf("  id=%-8llu score=%7.4f  category=%-6s  \"%s\"\n",
                   static_cast<unsigned long long>(h.id), h.dist, rec.attrs[1].text.c_str(),
                   rec.attrs[0].text.c_str());
    }
}

}  // namespace

void run_hybrid_bench(const char* data_dir) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);  // live progress even when piped
    const std::string dir       = data_dir;
    const std::string cache_dir = dir + "/cache";
    ::mkdir(cache_dir.c_str(), 0755);  // ignore EEXIST
    const std::string cache = cache_dir + "/hybrid_minecraft.snap";

    std::printf("\n######## Hybrid search demo: SIFT vectors + synthetic Minecraft text "
                "########\n");
    std::printf("loading from %s ...\n", dir.c_str());

    size_t n_base = 0, n_query = 0;
    auto   base    = read_fvecs(dir + "/sift_base.fvecs", n_base);
    auto   queries = read_fvecs(dir + "/sift_query.fvecs", n_query);
    const size_t n = std::min(n_base, N_CAP);
    std::printf("  base=%zu (capped to %zu)  query=%zu  (dim=%zu)\n", n_base, n, n_query, DIM);

    const std::string brute_cache = cache_dir + "/hybrid_minecraft_brute.snap";

    std::printf("\n[HNSW corpus]\n");
    auto hnsw_db = build_or_load_corpus(IndexKind::HNSW, cache, base, n);
    std::printf("\n[Brute corpus — exact oracle for recall, same corpus content]\n");
    auto brute_db = build_or_load_corpus(IndexKind::Brute, brute_cache, base, n);

    const size_t q_timing = std::min(Q_TIMING, n_query);

    // ---- BM25 (search_text) QPS by term commonality, plus a sanity check that
    //      it's exact regardless of index kind (see the EXPERIMENT GRID comment
    //      above — BM25 never touches the vector index at all). ----
    std::printf("\n== BM25 (search_text) QPS by term commonality ==\n");
    std::printf("%-22s %6s %10s %9s %9s %9s  %s\n", "term (tier)", "hits", "qps", "mean_us",
               "p50_us", "p95_us", "index-invariant?");
    for (const auto& t : kTextTiers) {
        const auto hnsw_hits  = hnsw_db->search_text(0, t.term, K_DEFAULT);
        const auto brute_hits = brute_db->search_text(0, t.term, K_DEFAULT);
        bool       exact_match = hnsw_hits.size() == brute_hits.size();
        for (size_t i = 0; exact_match && i < hnsw_hits.size(); ++i)
            if (hnsw_hits[i].id != brute_hits[i].id) exact_match = false;

        const auto stats = time_over(
            [&](size_t) { return hnsw_db->search_text(0, t.term, K_DEFAULT); }, MEASURE_REPS, 1);
        std::printf("%-22s %6zu %10.1f %9.1f %9.1f %9.1f  %s\n", t.label, hnsw_hits.size(),
                   stats.qps, stats.mean_us, stats.p50_us, stats.p95_us,
                   exact_match ? "yes (OK)" : "NO -- MISMATCH");
    }

    // ---- search_hybrid QPS + recall vs the Brute-indexed oracle, across RRF's
    //      depth knob. Recall = fraction of HNSW-hybrid's returned ids that also
    //      appear in the exact oracle's returned set for the same query, summed
    //      over q_timing SIFT query vectors — same definition and same
    //      unordered_set-intersection method filter.cpp's own
    //      post_recall_vs_prefilter uses. ----
    std::printf("\n== search_hybrid QPS + recall vs exact (Brute) oracle, term=\"%s\" ==\n",
               kHybridTerm.c_str());
    std::printf("%6s %10s %10s %10s %10s\n", "depth", "qps", "mean_us", "p95_us", "recall");
    for (size_t mult : kHybridDepthMult) {
        const size_t depth = mult * K_DEFAULT;

        for (size_t q = 0; q < q_timing; ++q)  // warm-up
            bench::do_not_optimize(
                hnsw_db->search_hybrid(queries.data() + q * DIM, 0, kHybridTerm, K_DEFAULT, depth)
                    .size());

        std::vector<double> per_us(q_timing);
        size_t              hits_total = 0, matched = 0;
        for (size_t q = 0; q < q_timing; ++q) {
            const float* qv = queries.data() + q * DIM;
            double       best = std::numeric_limits<double>::max();
            std::vector<Hit> approx;
            for (int r = 0; r < TIMING_REPS; ++r) {
                auto t0  = bench::clk::now();
                auto res = hnsw_db->search_hybrid(qv, 0, kHybridTerm, K_DEFAULT, depth);
                auto t1  = bench::clk::now();
                const double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
                if (us < best) { best = us; approx = std::move(res); }
            }
            per_us[q] = best;

            const auto exact = brute_db->search_hybrid(qv, 0, kHybridTerm, K_DEFAULT, depth);
            std::unordered_set<ExternalId> exact_ids;
            for (const auto& h : exact) exact_ids.insert(h.id);
            for (const auto& h : approx) {
                ++hits_total;
                if (exact_ids.count(h.id)) ++matched;
            }
        }
        std::sort(per_us.begin(), per_us.end());
        double sum = 0.0;
        for (double x : per_us) sum += x;
        const double mean   = sum / static_cast<double>(q_timing);
        const double recall = hits_total ? static_cast<double>(matched) / hits_total : 1.0;
        std::printf("%6zu %10.1f %10.1f %10.1f %10.4f\n", depth, 1e6 / mean, mean,
                   percentile(per_us, 95), recall);
    }

    // A couple of illustrative results, so the numbers above have something
    // concrete to point at.
    print_hits(*hnsw_db, ("search_text: \"" + kHybridTerm + "\"").c_str(),
              hnsw_db->search_text(0, kHybridTerm, 5));
    print_hits(*hnsw_db, ("search_hybrid: query vector 0 + \"" + kHybridTerm + "\"").c_str(),
              hnsw_db->search_hybrid(queries.data(), 0, kHybridTerm, 5));
}

void run_hybrid_repl(const char* data_dir) {
    const std::string dir   = data_dir;
    const std::string cache = dir + "/cache/hybrid_minecraft.snap";

    VDBConfig cfg;
    cfg.kind   = IndexKind::HNSW;
    cfg.dim    = DIM;
    cfg.metric = Metric::L2;
    cfg.schema = minecraft_schema();
    VDB db(cfg);

    std::printf("loading cached corpus %s ...\n", cache.c_str());
    try {
        load_snapshot(db, cache);
    } catch (const std::exception& e) {
        std::fprintf(stderr,
                     "!! no cached corpus at %s (%s)\n!! run `./build/run_bench hybrid %s` "
                     "first to build it\n",
                     cache.c_str(), e.what(), data_dir);
        std::exit(1);
    }
    std::printf("loaded N=%zu\n", db.size());

    size_t n_query = 0;
    const auto queries = read_fvecs(dir + "/sift_query.fvecs", n_query);

    std::printf(
        "\nHybrid search REPL over the synthetic Minecraft corpus.\n"
        "  <words>                              BM25 text search over 'description'\n"
        "  --cat <item|block|mob|biome> <words>  text search filtered to category\n"
        "  --vec <query index, 0-%zu> <words>    hybrid search (vector + text)\n"
        "  --vec <n> --cat <cat> <words>         hybrid search, filtered\n"
        "  --k <n> <words>                       result count (default 5)\n"
        "  help | quit / exit / Ctrl-D\n",
        n_query - 1);

    std::string line;
    for (;;) {
        std::printf("\n> ");
        std::fflush(stdout);
        if (!std::getline(std::cin, line)) break;  // EOF (Ctrl-D)

        const size_t b = line.find_first_not_of(" \t");
        if (b == std::string::npos) continue;  // blank line
        const size_t e = line.find_last_not_of(" \t");
        line           = line.substr(b, e - b + 1);

        if (line == "quit" || line == "exit") break;
        if (line == "help") {
            std::printf(
                "  <words>                              BM25 text search over 'description'\n"
                "  --cat <item|block|mob|biome> <words>  text search filtered to category\n"
                "  --vec <query index, 0-%zu> <words>    hybrid search (vector + text)\n"
                "  --vec <n> --cat <cat> <words>         hybrid search, filtered\n"
                "  --k <n> <words>                       result count (default 5)\n"
                "  quit / exit / Ctrl-D\n",
                n_query - 1);
            continue;
        }

        std::istringstream       iss(line);
        std::string              tok;
        long                     vec_idx = -1;
        std::string              category;
        size_t                   K = 5;
        std::vector<std::string> rest;
        while (iss >> tok) {
            if (tok == "--vec") {
                if (!(iss >> vec_idx)) { std::printf("!! --vec needs a number\n"); iss.clear(); }
            } else if (tok == "--cat") {
                if (!(iss >> category)) { std::printf("!! --cat needs a value\n"); iss.clear(); }
            } else if (tok == "--k") {
                long k = 0;
                if (iss >> k && k > 0) K = static_cast<size_t>(k);
                else { std::printf("!! --k needs a positive number\n"); iss.clear(); }
            } else {
                rest.push_back(tok);
            }
        }
        if (rest.empty()) { std::printf("(empty query — type 'help')\n"); continue; }
        if (vec_idx >= 0 && static_cast<size_t>(vec_idx) >= n_query) {
            std::printf("!! --vec %ld out of range (0-%zu)\n", vec_idx, n_query - 1);
            continue;
        }

        std::string query_text;
        for (size_t i = 0; i < rest.size(); ++i) {
            if (i) query_text += ' ';
            query_text += rest[i];
        }

        try {
            std::vector<Hit> hits;
            if (vec_idx >= 0 && !category.empty()) {
                hits = db.search_hybrid(queries.data() + static_cast<size_t>(vec_idx) * DIM, 0,
                                        query_text, K, pred_eq(1, attr_tag(category)));
            } else if (vec_idx >= 0) {
                hits = db.search_hybrid(queries.data() + static_cast<size_t>(vec_idx) * DIM, 0,
                                        query_text, K);
            } else if (!category.empty()) {
                hits = db.search_text(0, query_text, K, pred_eq(1, attr_tag(category)));
            } else {
                hits = db.search_text(0, query_text, K);
            }
            print_hits(db, query_text.c_str(), hits);
        } catch (const std::exception& ex) {
            std::printf("!! %s\n", ex.what());
        }
    }
    std::printf("\nbye\n");
}
