#pragma once
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "serialize.h"
#include "vdb_types.h"

namespace vdb {

class Index {
public:
    virtual ~Index() = default;

    virtual void train(const float* , size_t ) {}

    virtual InternalId add(const float* vec) = 0;

    // Two-phase add for concurrent callers (Stage 7). allocate() reserves the node
    // and stores the vector (serial: assigns the internal id); link() wires it into
    // the structure and may run concurrently under the index's own fine-grained
    // locks. The default is a single-phase add() + no-op link(), so brute/IVF — which
    // have no cheap split and get coarse protection from the DB layer — need no change.
    virtual InternalId allocate(const float* vec) { return add(vec); }
    virtual void       link(InternalId /*id*/) {}

    virtual std::vector<std::pair<InternalId, float>> search(const float* query,
                                                             size_t K) const = 0;

    virtual size_t size() const = 0;
    virtual size_t dim() const = 0;

    // Serialize/restore the full materialized index so a snapshot needs no rebuild.
    virtual void serialize(std::vector<uint8_t>& out) const = 0;
    virtual void deserialize(Reader& r) = 0;
};

}
