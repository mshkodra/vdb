#pragma once
#include <cstddef>
#include <utility>
#include <vector>

#include "vdb_types.h"

namespace vdb {

class Index {
public:
    virtual ~Index() = default;

    virtual void train(const float* , size_t ) {}

    virtual InternalId add(const float* vec) = 0;

    virtual std::vector<std::pair<InternalId, float>> search(const float* query,
                                                             size_t K) const = 0;

    virtual size_t size() const = 0;
    virtual size_t dim() const = 0;
};

}
