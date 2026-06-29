#include "ivf_index.h"

#include <algorithm>
#include <cassert>
#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <utility>

namespace vdb {

IVFIndex::IVFIndex(IVFConfig cfg, DistanceFn dist_fn)
    : config_(cfg), dist_fn_(std::move(dist_fn)) {}

// Brute-force argmin over the current centroids. Returns the index of the
// nearest centroid to `v`, or -1 if there are no centroids yet.
int IVFIndex::nearest_centroid(const float* v) const {
    int best = -1;
    float best_dist = std::numeric_limits<float>::max();
    for (size_t c = 0; c < centroids_.size(); ++c) {
        float d = dist_fn_(v, centroids_[c].data(), config_.dim);
        if (d < best_dist) {
            best_dist = d;
            best = static_cast<int>(c);
        }
    }
    return best;
}

void IVFIndex::train(const float* data, size_t n) {
    const size_t dim = config_.dim;

    // Can't have more clusters than points.
    const size_t nlist = std::min(config_.nlist, n);
    config_.nlist = nlist;

    // Forgy initilization of centroids.
    std::mt19937 rng(42);
    std::vector<size_t> perm(n);
    std::iota(perm.begin(), perm.end(), 0);
    std::shuffle(perm.begin(), perm.end(), rng);

    centroids_.assign(nlist, std::vector<float>(dim));
    for (size_t i = 0; i < nlist; ++i) {
        const float* src = data + perm[i] * dim;
        std::copy(src, src + dim, centroids_[i].begin());
    }

    // Lloyd's Algorithm for k-means clustering.
    for (size_t iter = 0; iter < config_.kmeans_iters; ++iter) {
        // Accumulate per-cluster sum + count in one pass.
        std::vector<std::vector<float>> sums(nlist, std::vector<float>(dim, 0.0f));
        std::vector<size_t> counts(nlist, 0);

        for (size_t i = 0; i < n; ++i) {
            const float* v = data + i * dim;
            int c = nearest_centroid(v);
            float* s = sums[c].data();
            for (size_t d = 0; d < dim; ++d) s[d] += v[d];
            counts[c]++;
        }

        // Update each centroid to the mean of its assigned vectors.
        for (size_t c = 0; c < nlist; ++c) {
            if (counts[c] == 0) {
                // Empty cluster: leave the centroid put so it can still win
                continue;
            }
            float inv = 1.0f / static_cast<float>(counts[c]);
            for (size_t d = 0; d < dim; ++d) centroids_[c][d] = sums[c][d] * inv;
        }
    }

    // Inverted lists creation.
    inverted_lists_.assign(nlist, {});
    vectors_.assign(n, std::vector<float>(dim));
    for (size_t i = 0; i < n; ++i) {
        const float* v = data + i * dim;
        std::copy(v, v + dim, vectors_[i].begin());
        int c = nearest_centroid(v);
        inverted_lists_[c].push_back(static_cast<InternalId>(i));
    }

    trained_ = true;
}

InternalId IVFIndex::add(const float* vec) {
    assert(trained_ && "IVFIndex::add called before train()");

    InternalId id = static_cast<InternalId>(vectors_.size());
    int closestCentroid = nearest_centroid(vec);

    inverted_lists_[closestCentroid].push_back(id);
    vectors_.push_back(std::vector<float>(vec, vec + config_.dim));
    return id;
}

std::vector<std::pair<InternalId, float>> IVFIndex::search(const float* query,
                                                           size_t K) const {
    assert(trained_ && "IVFIndex::search called before train()");
    if (K == 0 || centroids_.empty()) return {};

    // Pick the nprobe nearest centroids.
    std::vector<std::pair<float, size_t>> cd;
    cd.reserve(centroids_.size());
    for (size_t c = 0; c < centroids_.size(); ++c) {
        cd.emplace_back(dist_fn_(query, centroids_[c].data(), config_.dim), c);
    }
    size_t probe = std::min(config_.nprobe, cd.size());
    std::nth_element(cd.begin(), cd.begin() + probe, cd.end());

    // Stream probed candidates through a bounded max-heap of the K nearest.
    std::priority_queue<std::pair<float, InternalId>> heap;
    for (size_t p = 0; p < probe; ++p) {
        size_t c = cd[p].second;
        for (InternalId id : inverted_lists_[c]) {
            float d = dist_fn_(query, vectors_[id].data(), config_.dim);
            if (heap.size() < K) {
                heap.emplace(d, id);
            } else if (d < heap.top().first) {
                heap.pop();
                heap.emplace(d, id);
            }
        }
    }

    // Drain largest-first into ascending-by-distance results.
    std::vector<std::pair<InternalId, float>> result(heap.size());
    for (size_t i = heap.size(); i > 0; --i) {
        result[i - 1] = {heap.top().second, heap.top().first};
        heap.pop();
    }
    return result;
}

size_t IVFIndex::size() const { return vectors_.size(); }
size_t IVFIndex::dim() const { return config_.dim; }

}  // namespace vdb
