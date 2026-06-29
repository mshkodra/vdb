#pragma once
#include <vector>

#include "index.h"

namespace vdb {

// Inverted File index (IVF) — ROADMAP Stage 2.
//
// Idea: partition the vector space into `nlist` Voronoi cells via k-means. Each
// cell has an inverted list of the vectors assigned to it. A query probes only
// the `nprobe` nearest cells instead of scanning everything. This trades recall
// for speed: more probes -> higher recall, lower QPS.
struct IVFConfig {
    size_t dim;
    size_t nlist  = 100;  // number of centroids / cells (heuristic: ~sqrt(N))
    size_t nprobe = 8;    // cells probed per query (recall/speed knob)
    size_t kmeans_iters = 25;
};

class IVFIndex : public Index {
public:
    IVFIndex(IVFConfig cfg, DistanceFn dist_fn);

    // Runs k-means over the sample to learn `nlist` centroids. Must be called
    // before add().
    void train(const float* data, size_t n) override;

    InternalId add(const float* vec) override;
    std::vector<std::pair<InternalId, float>> search(const float* query,
                                                     size_t K) const override;

    // nprobe is a search-time recall/speed dial; no retrain needed to change it.
    void set_nprobe(size_t np) { config_.nprobe = np; }
    size_t size() const override;
    size_t dim() const override;

private:
    IVFConfig   config_;
    DistanceFn  dist_fn_;
    bool        trained_ = false;

    // TODO Stage 2:
    std::vector<std::vector<float>>             centroids_;      // nlist x dim
    std::vector<std::vector<InternalId>>        inverted_lists_; // nlist lists
    std::vector<std::vector<float>>             vectors_;        // raw storage

    // Argmin over centroids_ for a single vector (reused by train + add).
    int nearest_centroid(const float* v) const;
};

}  // namespace vdb
