#include "test.h"

#include "vdb.h"

// Placeholder so the suite compiles from day one. Replace/extend per stage:
// Stage 1 -> distance + brute-force correctness, Stage 2 -> IVF recall, etc.
TEST(constructs) {
    vdb::VDBConfig cfg;
    cfg.kind = vdb::IndexKind::Brute;
    cfg.dim  = 4;
    vdb::VDB db(cfg);
    EXPECT(db.dim() == 4);
    EXPECT(db.size() == 0);
}
