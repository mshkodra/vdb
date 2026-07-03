#include "test.h"

#include "vdb.h"

TEST(constructs) {
    vdb::VDBConfig cfg;
    cfg.kind = vdb::IndexKind::Brute;
    cfg.dim  = 4;
    vdb::VDB db(cfg);
    EXPECT(db.dim() == 4);
    EXPECT(db.size() == 0);
}
