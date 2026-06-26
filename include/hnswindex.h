#pragma once
#include <cstddef>
#include <cstdint>
#include <functional>
#include <istream>
#include <ostream>
#include <random>
#include <vector>

using InternalId = uint32_t;
using ExternalId = uint64_t;
using DistanceFn = std::function<float(const float*, const float*, size_t)>;

struct HNSWConfig {
    size_t dim;               // vector dimensionality
    size_t M         = 16;    // connections per node per layer
    size_t Mmax      = 16;    // max connections at layers > 0 (usually = M)
    size_t Mmax0     = 32;    // max connections at layer 0  (usually 2*M)
    size_t ef        = 200;   // efConstruction — search width at build time
    float  mL        = 0.0f;  // set in constructor to 1.0f / std::log(M)
};

class HNSWIndex {
    public:
        explicit HNSWIndex(HNSWConfig cfg, DistanceFn dist_fn);

        InternalId insert(const float* vec);
        std::vector<InternalId> search(const float* query, size_t K, size_t ef) const;
        const float* get(InternalId) const;

        // Tombstone a node: it is excluded from search *results* but still traversed
        // through during graph descent, so connectivity is preserved. See Node::deleted.
        void mark_deleted(InternalId id);
        bool is_deleted(InternalId id) const;

        size_t size() const { return node_pool_.size(); }
        size_t dim()  const { return config_.dim; }

        // Serialize / rebuild the raw graph (node vectors, per-layer neighbour
        // lists, tombstone flags, entry point, max layer) to/from a byte stream.
        // This is pure topology + payload — it knows nothing about ExternalIds.
        // Identity (the ext<->int maps) is persisted one layer up, in VDB. See
        // the Phase 2 doc on why the split is preserved through snapshots.
        void save(std::ostream& os) const;
        void load(std::istream& is);

    private:
        struct Node {
            std::vector<float> data;
            std::vector<std::vector<InternalId>> neighbours;
            bool deleted = false;  // tombstone flag (see mark_deleted)
            Node() = default;      // used by load()
            Node(const float* vec, size_t dim, int max_layer, size_t M, size_t Mmax0);
        };

        HNSWConfig  config_;
        DistanceFn  dist_fn_;
        InternalId  entry_point_ = 0;
        int         max_layer_   = -1;
        mutable std::mt19937 rng_;


        std::vector<Node> node_pool_;

        std::vector<InternalId> search_layer(const float* q,
                                          const std::vector<InternalId>& eps,
                                          size_t ef, int layer) const;
        std::vector<InternalId> select_neighbors(const float* q,
                                              const std::vector<InternalId>& candidates,
                                              size_t M, int layer) const;

        int  sample_layer() const;
        float distance(InternalId a, InternalId b) const;
        float distance(const float* q, InternalId b) const;
};
