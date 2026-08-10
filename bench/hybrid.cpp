#include "hybrid.h"

#include "bench.h"
#include "snapshot.h"
#include "vdb.h"

#include <sys/stat.h>  // mkdir for the snapshot cache dir

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace vdb;

namespace {

constexpr size_t DIM = 128;  // SIFT descriptors are 128-d — reusing filter.cpp's vectors

// ============================================================================
//  EXPERIMENT GRID — this block is yours to design (see CLAUDE.md).
//  Everything below it is plumbing.
//
//  What's built right now is a *demo*, not a benchmark sweep: it builds the
//  corpus, caches it, and runs a handful of example queries so the shape of
//  search_text/search_hybrid's output is visible end to end. Swap the demo-query
//  section at the bottom of run_hybrid_bench() out once you've decided what to
//  actually sweep — query selectivity (vocabulary skew controls this), corpus
//  size, RRF depth vs K, k1/b sensitivity, predicate-gated vs ungated hybrid
//  cost — same shape as filter.cpp's LOW_S/HIGH_S sweep design.
constexpr size_t N_CAP        = 50000;  // corpus size — SIFT1M has 1M, capped for a fast demo
constexpr unsigned VOCAB_SEED = 11;
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

    VDBConfig cfg;
    cfg.kind   = IndexKind::HNSW;
    cfg.dim    = DIM;
    cfg.metric = Metric::L2;
    cfg.schema = minecraft_schema();
    auto db = std::make_unique<VDB>(cfg);

    bool loaded = false;
    try {
        load_snapshot(*db, cache);
        // Guards against silently loading a cache built at a different N_CAP — same
        // footgun filter.cpp's own run_distribution() fixed for the same reason.
        loaded = (db->size() == n);
    } catch (const std::exception&) {
        // No cache yet, or it doesn't match this VDB's config — build fresh.
    }
    if (!loaded) db = std::make_unique<VDB>(cfg);  // discard a wrong-sized partial load

    if (loaded) {
        std::printf("loaded cache %s (N=%zu)\n", cache.c_str(), db->size());
    } else {
        std::printf("building corpus over %zu vectors — this is the slow one...\n", n);
        std::mt19937 rng(VOCAB_SEED);
        for (size_t i = 0; i < n; ++i) {
            const MinecraftDoc doc = make_doc(rng);
            Record              r;
            r.attrs = {attr_text(doc.description), attr_tag(doc.category)};
            db->insert(base.data() + i * DIM, r);
            if ((i + 1) % 10000 == 0) std::printf("    ... %zu / %zu inserted\n", i + 1, n);
        }
        save_snapshot(*db, cache, 0);
        std::printf("cached -> %s\n", cache.c_str());
    }

    // ---- Demo queries: shows the shape of search_text/search_hybrid's output.
    //      Not a benchmark sweep — see the EXPERIMENT GRID comment above. ----
    print_hits(*db, "search_text: \"diamond sword\"", db->search_text(0, "diamond sword", 5));
    print_hits(*db, "search_text: \"creeper nether\"", db->search_text(0, "creeper nether", 5));
    print_hits(*db, "search_hybrid: query vector 0 + \"legendary\"",
              db->search_hybrid(queries.data(), 0, "legendary", 5));
    print_hits(*db, "search_hybrid (category=item): query vector 0 + \"legendary\"",
              db->search_hybrid(queries.data(), 0, "legendary", 5, pred_eq(1, attr_tag("item"))));
}
