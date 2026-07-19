#include "brute_index.h"
#include <algorithm>
#include <utility>

namespace vdb {

BruteIndex::BruteIndex(size_t dim, DistanceFn dist_fn)
    : dim_(dim), dist_fn_(std::move(dist_fn)) {}

InternalId BruteIndex::add(const float* vec) {
    data_.emplace_back(vec, vec + dim_);
    return data_.size() - 1;
}

std::vector<std::pair<InternalId, float>> BruteIndex::search(const float* query,
                                                             size_t K) const {
    using Entry = std::pair<InternalId, float>;

    std::vector<Entry> all;
    all.reserve(data_.size());
    for (size_t i = 0; i < data_.size(); ++i)
        all.push_back({static_cast<InternalId>(i),
                       dist_fn_(query, data_[i].data(), dim_)});

    auto by_distance = [](const Entry& a, const Entry& b) {
        return a.second < b.second;
    };

    const size_t k = std::min(K, all.size());

    std::nth_element(all.begin(), all.begin() + k, all.end(), by_distance);

    all.resize(k);
    std::sort(all.begin(), all.end(), by_distance);
    return all;
}

size_t BruteIndex::size() const { return data_.size(); }
size_t BruteIndex::dim() const { return dim_; }

void BruteIndex::serialize(std::vector<uint8_t>& out) const {
    put<uint64_t>(out, dim_);
    put<uint64_t>(out, data_.size());
    for (const auto& v : data_) put_floats(out, v);
}

void BruteIndex::deserialize(Reader& r) {
    dim_ = r.get<uint64_t>();
    const uint64_t n = r.get<uint64_t>();
    data_.resize(n);
    for (uint64_t i = 0; i < n; ++i) data_[i] = r.get_floats();
}

}
