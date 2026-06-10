#pragma once

#include <hnswindex.h>

#include <memory>
#include <vector>

enum class IndexType {
    HNSW,
};

struct VDBConfig {
    IndexType  type = IndexType::HNSW;
    HNSWConfig hnsw;          // used when type == HNSW
    DistanceFn dist_fn;       // optional; defaults to squared L2 if empty
};

class VDB {
public:
    explicit VDB(VDBConfig cfg);

    InternalId insert(const float* vec);
    std::vector<InternalId> search(const float* query, size_t K, size_t ef) const;
    const float* get(InternalId id) const;

    size_t size() const;
    size_t dim()  const;

private:
    VDBConfig                    config_;
    std::unique_ptr<HNSWIndex>   hnsw_;
};
