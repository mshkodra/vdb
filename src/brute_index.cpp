#include "brute_index.h"
#include <algorithm>
#include <utility>

#include "distance.h"

namespace vdb {

template <class Dist>
BruteIndex<Dist>::BruteIndex(size_t dim) : dim_(dim) {}

template <class Dist>
InternalId BruteIndex<Dist>::add(const float* vec) {
    data_.emplace_back(vec, vec + dim_);
    return data_.size() - 1;
}

template <class Dist>
std::vector<std::pair<InternalId, float>> BruteIndex<Dist>::search(const float* query,
                                                                   size_t K) const {
    using Entry = std::pair<InternalId, float>;

    std::vector<Entry> all;
    all.reserve(data_.size());
    for (size_t i = 0; i < data_.size(); ++i)
        all.push_back({static_cast<InternalId>(i),
                       dist_(query, data_[i].data(), dim_)});

    auto by_distance = [](const Entry& a, const Entry& b) {
        return a.second < b.second;
    };

    const size_t k = std::min(K, all.size());

    std::nth_element(all.begin(), all.begin() + k, all.end(), by_distance);

    all.resize(k);
    std::sort(all.begin(), all.end(), by_distance);
    return all;
}

template <class Dist>
size_t BruteIndex<Dist>::size() const { return data_.size(); }

template <class Dist>
size_t BruteIndex<Dist>::dim() const { return dim_; }

template <class Dist>
void BruteIndex<Dist>::serialize(std::vector<uint8_t>& out) const {
    put<uint64_t>(out, dim_);
    put<uint64_t>(out, data_.size());
    for (const auto& v : data_) put_floats(out, v);
}

template <class Dist>
void BruteIndex<Dist>::deserialize(Reader& r) {
    dim_ = r.get<uint64_t>();
    const uint64_t n = r.get<uint64_t>();
    data_.resize(n);
    for (uint64_t i = 0; i < n; ++i) data_[i] = r.get_floats();
}

// One materialized index type per metric functor. The definitions above are only
// instantiated here, so they stay in this .cpp (header/impl split preserved) while
// still inlining each functor's loop into the search hot path.
template class BruteIndex<L2>;
template class BruteIndex<InnerProduct>;
template class BruteIndex<Cosine>;

}
