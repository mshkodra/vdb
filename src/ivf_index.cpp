#include "ivf_index.h"

#include <algorithm>
#include <cassert>
#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <utility>

#include "distance.h"

namespace vdb {

template <class Dist>
IVFIndex<Dist>::IVFIndex(IVFConfig cfg) : config_(cfg) {}

template <class Dist>
int IVFIndex<Dist>::nearest_centroid(const float* v) const {
    int best = -1;
    float best_dist = std::numeric_limits<float>::max();
    for (size_t c = 0; c < centroids_.size(); ++c) {
        float d = dist_(v, centroids_[c].data(), config_.dim);
        if (d < best_dist) {
            best_dist = d;
            best = static_cast<int>(c);
        }
    }
    return best;
}

template <class Dist>
void IVFIndex<Dist>::train(const float* data, size_t n) {
    const size_t dim = config_.dim;

    const size_t nlist = std::min(config_.nlist, n);
    config_.nlist = nlist;

    std::mt19937 rng(42);
    std::vector<size_t> perm(n);
    std::iota(perm.begin(), perm.end(), 0);
    std::shuffle(perm.begin(), perm.end(), rng);

    centroids_.assign(nlist, std::vector<float>(dim));
    for (size_t i = 0; i < nlist; ++i) {
        const float* src = data + perm[i] * dim;
        std::copy(src, src + dim, centroids_[i].begin());
    }

    for (size_t iter = 0; iter < config_.kmeans_iters; ++iter) {
        std::vector<std::vector<float>> sums(nlist, std::vector<float>(dim, 0.0f));
        std::vector<size_t> counts(nlist, 0);

        for (size_t i = 0; i < n; ++i) {
            const float* v = data + i * dim;
            int c = nearest_centroid(v);
            float* s = sums[c].data();
            for (size_t d = 0; d < dim; ++d) s[d] += v[d];
            counts[c]++;
        }

        for (size_t c = 0; c < nlist; ++c) {
            if (counts[c] == 0) {
                continue;
            }
            float inv = 1.0f / static_cast<float>(counts[c]);
            for (size_t d = 0; d < dim; ++d) centroids_[c][d] = sums[c][d] * inv;
        }
    }

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

template <class Dist>
InternalId IVFIndex<Dist>::add(const float* vec) {
    assert(trained_ && "IVFIndex::add called before train()");

    InternalId id = static_cast<InternalId>(vectors_.size());
    int closestCentroid = nearest_centroid(vec);

    inverted_lists_[closestCentroid].push_back(id);
    vectors_.push_back(std::vector<float>(vec, vec + config_.dim));
    return id;
}

template <class Dist>
std::vector<std::pair<InternalId, float>> IVFIndex<Dist>::search(const float* query,
                                                                size_t K) const {
    assert(trained_ && "IVFIndex::search called before train()");
    if (K == 0 || centroids_.empty()) return {};

    std::vector<std::pair<float, size_t>> cd;
    cd.reserve(centroids_.size());
    for (size_t c = 0; c < centroids_.size(); ++c) {
        cd.emplace_back(dist_(query, centroids_[c].data(), config_.dim), c);
    }
    size_t probe = std::min(config_.nprobe, cd.size());
    std::nth_element(cd.begin(), cd.begin() + probe, cd.end());

    std::priority_queue<std::pair<float, InternalId>> heap;
    for (size_t p = 0; p < probe; ++p) {
        size_t c = cd[p].second;
        for (InternalId id : inverted_lists_[c]) {
            float d = dist_(query, vectors_[id].data(), config_.dim);
            if (heap.size() < K) {
                heap.emplace(d, id);
            } else if (d < heap.top().first) {
                heap.pop();
                heap.emplace(d, id);
            }
        }
    }

    std::vector<std::pair<InternalId, float>> result(heap.size());
    for (size_t i = heap.size(); i > 0; --i) {
        result[i - 1] = {heap.top().second, heap.top().first};
        heap.pop();
    }
    return result;
}

template <class Dist>
size_t IVFIndex<Dist>::size() const { return vectors_.size(); }

template <class Dist>
size_t IVFIndex<Dist>::dim() const { return config_.dim; }

template <class Dist>
void IVFIndex<Dist>::serialize(std::vector<uint8_t>& out) const {
    put<uint8_t>(out, trained_ ? 1 : 0);
    put<uint64_t>(out, config_.nlist);
    put<uint64_t>(out, config_.nprobe);
    put<uint64_t>(out, centroids_.size());
    for (const auto& c : centroids_) put_floats(out, c);
    put<uint64_t>(out, inverted_lists_.size());
    for (const auto& lst : inverted_lists_) {
        put<uint64_t>(out, lst.size());
        for (InternalId id : lst) put<uint32_t>(out, id);
    }
    put<uint64_t>(out, vectors_.size());
    for (const auto& v : vectors_) put_floats(out, v);
}

template <class Dist>
void IVFIndex<Dist>::deserialize(Reader& r) {
    trained_       = r.get<uint8_t>() != 0;
    config_.nlist  = r.get<uint64_t>();
    config_.nprobe = r.get<uint64_t>();
    const uint64_t nc = r.get<uint64_t>();
    centroids_.resize(nc);
    for (uint64_t i = 0; i < nc; ++i) centroids_[i] = r.get_floats();
    const uint64_t nl = r.get<uint64_t>();
    inverted_lists_.resize(nl);
    for (uint64_t i = 0; i < nl; ++i) {
        const uint64_t m = r.get<uint64_t>();
        inverted_lists_[i].resize(m);
        for (uint64_t j = 0; j < m; ++j) inverted_lists_[i][j] = r.get<uint32_t>();
    }
    const uint64_t nv = r.get<uint64_t>();
    vectors_.resize(nv);
    for (uint64_t i = 0; i < nv; ++i) vectors_[i] = r.get_floats();
}

// One materialized index type per metric functor (see brute_index.cpp).
template class IVFIndex<L2>;
template class IVFIndex<InnerProduct>;
template class IVFIndex<Cosine>;

}
