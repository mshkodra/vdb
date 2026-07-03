#pragma once
#include <memory>
#include <vector>

#include "distance.h"
#include "index.h"

namespace vdb {

enum class IndexKind { Brute, IVF, HNSW };

struct VDBConfig {
    IndexKind kind   = IndexKind::HNSW;
    size_t    dim    = 0;
    Metric    metric = Metric::L2;
};

class VDB {
public:
    explicit VDB(VDBConfig cfg);

    ExternalId insert(const float* vec);

    std::vector<ExternalId> search(const float* query, size_t K) const;

    size_t size() const;
    size_t dim() const;

private:
    VDBConfig              config_;
    std::unique_ptr<Index> index_;

};

}
