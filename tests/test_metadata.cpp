#include "test.h"

#include "durable_vdb.h"
#include "metadata.h"
#include "snapshot.h"
#include "vdb.h"

#include <filesystem>
#include <string>
#include <vector>

#include <unistd.h>

using namespace vdb;

namespace {

// category (tag) | price (int64) | in_stock (bool) | rating (float64)
std::vector<AttrSpec> demo_schema() {
    return {{"category", AttrType::Tag},
            {"price", AttrType::Int64},
            {"in_stock", AttrType::Bool},
            {"rating", AttrType::Float64}};
}

VDBConfig demo_config(IndexKind kind = IndexKind::Brute) {
    VDBConfig cfg;
    cfg.kind   = kind;
    cfg.dim    = 2;
    cfg.metric = Metric::L2;
    cfg.schema = demo_schema();
    return cfg;
}

Record row(const std::string& cat, int64_t price, bool stock, double rating,
           std::vector<uint8_t> payload = {}) {
    Record r;
    r.attrs = {attr_tag(cat), attr_int(price), attr_bool(stock), attr_float(rating)};
    r.payload = std::move(payload);
    return r;
}

std::vector<uint8_t> bytes(const std::string& s) { return {s.begin(), s.end()}; }

std::string temp_dir(const char* name) {
    auto p = std::filesystem::temp_directory_path() /
             ("vdb_meta_" + std::string(name) + "_" + std::to_string(::getpid()));
    std::filesystem::remove_all(p);
    return p.string();
}

}  // namespace

TEST(metadata_store_round_trips_every_type) {
    MetadataStore m(demo_schema());
    m.append_row(row("shoes", 8999, true, 4.5));
    m.append_row(row("hats", -20, false, -1.25));

    EXPECT(m.rows() == 2);
    EXPECT(m.get(0, 0).text == "shoes");
    EXPECT(m.get(0, 1).as_int() == 8999);
    EXPECT(m.get(0, 2).as_bool() == true);
    EXPECT(m.get(0, 3).as_double() == 4.5);

    EXPECT(m.get(1, 0).text == "hats");
    EXPECT(m.get(1, 1).as_int() == -20);       // negative survives the u64 punning
    EXPECT(m.get(1, 2).as_bool() == false);
    EXPECT(m.get(1, 3).as_double() == -1.25);  // negative double too
}

TEST(metadata_dictionary_dedupes_and_assigns_dense_codes) {
    MetadataStore m(demo_schema());
    m.append_row(row("shoes", 1, true, 0.0));
    m.append_row(row("hats", 2, true, 0.0));
    m.append_row(row("shoes", 3, true, 0.0));

    uint32_t shoes = 999, hats = 999;
    ASSERT(m.tag_code(0, "shoes", shoes));
    ASSERT(m.tag_code(0, "hats", hats));
    EXPECT(shoes == 0);  // first seen
    EXPECT(hats == 1);

    // Row 0 and row 2 share one dictionary entry: the column holds the same code,
    // which is what makes an equality predicate a single integer compare.
    const uint64_t* col = m.column_raw(0);
    EXPECT(col[0] == shoes);
    EXPECT(col[2] == shoes);
    EXPECT(col[1] == hats);

    uint32_t missing = 0;
    EXPECT(!m.tag_code(0, "socks", missing));  // absent value -> no code
}

TEST(metadata_nulls_are_distinct_from_zero) {
    MetadataStore m(demo_schema());
    Record r;
    r.attrs = {attr_null(), attr_int(0), attr_null(), attr_float(0.0)};
    m.append_row(r);

    EXPECT(!m.present(0, 0));
    EXPECT(m.present(0, 1));
    EXPECT(m.get(0, 0).is_null());
    EXPECT(!m.get(0, 1).is_null());
    EXPECT(m.get(0, 1).as_int() == 0);  // a stored zero is not a null

    // An empty attrs vector is the "all null" shorthand a metadata-less insert uses.
    m.append_row(Record{});
    for (size_t a = 0; a < 4; ++a) EXPECT(!m.present(1, a));
}

TEST(indexed_flag_defaults_to_false) {
    MetadataStore m(demo_schema());
    for (const auto& s : m.schema()) EXPECT(!s.indexed);
}

TEST(indexed_allowed_only_on_numeric_columns) {
    // Int64 and Float64 may opt into indexed — no throw.
    { std::vector<AttrSpec> s = {{"price", AttrType::Int64, /*indexed=*/true}};
      MetadataStore m(s);
      EXPECT(m.schema()[0].indexed); }
    { std::vector<AttrSpec> s = {{"rating", AttrType::Float64, /*indexed=*/true}};
      MetadataStore m(s);
      EXPECT(m.schema()[0].indexed); }

    // Tag and Bool get a postings list unconditionally and never need this flag —
    // declaring indexed=true on them is a schema error, not a silent no-op.
    bool threw = false;
    try {
        std::vector<AttrSpec> s = {{"category", AttrType::Tag, /*indexed=*/true}};
        MetadataStore m(s);
    } catch (const std::invalid_argument&) { threw = true; }
    EXPECT(threw);

    threw = false;
    try {
        std::vector<AttrSpec> s = {{"in_stock", AttrType::Bool, /*indexed=*/true}};
        MetadataStore m(s);
    } catch (const std::invalid_argument&) { threw = true; }
    EXPECT(threw);
}

TEST(indexed_flag_changes_the_fingerprint) {
    std::vector<AttrSpec> plain  = {{"price", AttrType::Int64, false}};
    std::vector<AttrSpec> idx    = {{"price", AttrType::Int64, true}};
    MetadataStore m_plain(plain);
    MetadataStore m_idx(idx);
    // Same name, same type, different indexedness: a real layout change (the indexed
    // column carries an extra secondary structure once PR9 wires it in), so the
    // fingerprint must distinguish them the same way it distinguishes a retyped or
    // renamed column.
    EXPECT(m_plain.fingerprint() != m_idx.fingerprint());

    // Re-declaring the identical schema is deterministic and reproducible.
    MetadataStore m_idx2(idx);
    EXPECT(m_idx.fingerprint() == m_idx2.fingerprint());
}

TEST(metadata_rejects_schema_violations) {
    MetadataStore m(demo_schema());

    bool threw = false;
    try {
        Record r;
        r.attrs = {attr_tag("shoes")};  // arity 1 against a 4-attribute schema
        m.append_row(r);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    EXPECT(threw);

    threw = false;
    try {
        Record r;
        r.attrs = {attr_int(5), attr_int(1), attr_bool(true), attr_float(0.0)};
        m.append_row(r);  // attr 0 is a Tag, not an Int64
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    EXPECT(threw);

    EXPECT(m.rows() == 0);  // neither bad row was appended
}

TEST(metadata_survives_insert_and_readback) {
    VDB db(demo_config());
    const float a[2] = {0.0f, 0.0f};
    const float b[2] = {5.0f, 5.0f};

    const ExternalId id_a = db.insert(a, row("shoes", 8999, true, 4.5, bytes("sku-A")));
    const ExternalId id_b = db.insert(b, row("hats", 1200, false, 3.0, bytes("sku-B")));

    Record got;
    ASSERT(db.get_metadata(id_a, got));
    EXPECT(got.attrs[0].text == "shoes");
    EXPECT(got.attrs[1].as_int() == 8999);
    EXPECT(got.payload == bytes("sku-A"));

    ASSERT(db.get_metadata(id_b, got));
    EXPECT(got.attrs[0].text == "hats");
    EXPECT(got.payload == bytes("sku-B"));
}

TEST(metadata_insert_without_schema_still_works) {
    VDBConfig cfg;  // no schema declared at all
    cfg.kind = IndexKind::Brute;
    cfg.dim  = 2;
    VDB db(cfg);

    const float v[2] = {1.0f, 1.0f};
    Record r;
    r.payload = bytes("just-a-payload");
    const ExternalId id = db.insert(v, r);

    Record got;
    ASSERT(db.get_metadata(id, got));
    EXPECT(got.attrs.empty());
    EXPECT(got.payload == bytes("just-a-payload"));
}

TEST(search_hits_carry_payload_in_distance_order) {
    VDB db(demo_config());
    const float near_[2]  = {0.0f, 0.0f};
    const float far_[2]   = {9.0f, 9.0f};
    const float query[2]  = {0.1f, 0.1f};

    db.insert(near_, row("shoes", 1, true, 1.0, bytes("near")));
    db.insert(far_, row("hats", 2, true, 2.0, bytes("far")));

    const auto hits = db.search_hits(query, 2);
    ASSERT(hits.size() == 2);
    EXPECT(hits[0].payload == bytes("near"));
    EXPECT(hits[1].payload == bytes("far"));
    EXPECT(hits[0].dist <= hits[1].dist);

    // The bare search() overload still returns ids only, and agrees on the order.
    const auto ids = db.search(query, 2);
    ASSERT(ids.size() == 2);
    EXPECT(ids[0] == hits[0].id);
}

TEST(set_metadata_replaces_row_without_moving_the_vector) {
    VDB db(demo_config());
    const float v[2] = {1.0f, 2.0f};
    const ExternalId id = db.insert(v, row("shoes", 100, true, 1.0, bytes("v1")));

    const size_t deleted_before = db.deleted_count();
    ASSERT(db.set_metadata(id, row("boots", 200, false, 2.0, bytes("v2"))));

    // No tombstone: unlike update(), a metadata-only write leaves the graph alone.
    EXPECT(db.deleted_count() == deleted_before);
    EXPECT(db.size() == 1);

    Record got;
    ASSERT(db.get_metadata(id, got));
    EXPECT(got.attrs[0].text == "boots");
    EXPECT(got.attrs[1].as_int() == 200);
    EXPECT(got.attrs[2].as_bool() == false);
    EXPECT(got.payload == bytes("v2"));

    EXPECT(!db.set_metadata(9999, row("x", 1, true, 1.0)));  // unknown id
}

TEST(update_carries_metadata_forward_by_default) {
    VDB db(demo_config());
    const float v1[2] = {1.0f, 1.0f};
    const float v2[2] = {7.0f, 7.0f};
    const ExternalId id = db.insert(v1, row("shoes", 100, true, 4.0, bytes("keep")));

    ASSERT(db.update(id, v2));  // vector-only update

    Record got;
    ASSERT(db.get_metadata(id, got));
    EXPECT(got.attrs[0].text == "shoes");
    EXPECT(got.attrs[1].as_int() == 100);
    EXPECT(got.payload == bytes("keep"));

    // The explicit overload replaces the row instead.
    ASSERT(db.update(id, v1, row("clogs", 5, false, 1.0, bytes("new"))));
    ASSERT(db.get_metadata(id, got));
    EXPECT(got.attrs[0].text == "clogs");
    EXPECT(got.payload == bytes("new"));
}

TEST(compact_permutes_metadata_with_internal_ids) {
    VDB db(demo_config(IndexKind::HNSW));
    std::vector<ExternalId> ids;
    for (int i = 0; i < 10; ++i) {
        const float v[2] = {static_cast<float>(i), static_cast<float>(i)};
        ids.push_back(db.insert(v, row("cat" + std::to_string(i), i, i % 2 == 0,
                                       i * 0.5, bytes("p" + std::to_string(i)))));
    }
    // Tombstone the even ones, so the survivors are not a contiguous id range.
    for (int i = 0; i < 10; i += 2) ASSERT(db.remove(ids[i]));
    ASSERT(db.deleted_count() == 5);

    db.compact();
    EXPECT(db.deleted_count() == 0);
    EXPECT(db.size() == 5);

    // Every surviving external id must still see its own row after renumbering.
    for (int i = 1; i < 10; i += 2) {
        Record got;
        ASSERT(db.get_metadata(ids[i], got));
        EXPECT(got.attrs[0].text == "cat" + std::to_string(i));
        EXPECT(got.attrs[1].as_int() == i);
        EXPECT(got.attrs[3].as_double() == i * 0.5);
        EXPECT(got.payload == bytes("p" + std::to_string(i)));
    }
    // And the dead rows are gone with their ids.
    Record dead;
    EXPECT(!db.get_metadata(ids[0], dead));
}

TEST(snapshot_round_trips_metadata) {
    const std::string dir = temp_dir("snap");
    std::filesystem::create_directories(dir);
    const std::string path = dir + "/snapshot";

    std::vector<ExternalId> ids;
    {
        VDB db(demo_config());
        for (int i = 0; i < 5; ++i) {
            const float v[2] = {static_cast<float>(i), 0.0f};
            ids.push_back(db.insert(v, row(i % 2 ? "shoes" : "hats", i * 10, i % 2 == 0,
                                           i * 1.5, bytes("pay" + std::to_string(i)))));
        }
        ASSERT(db.remove(ids[2]));  // keep a tombstone in the image
        save_snapshot(db, path, /*lsn=*/99);
    }

    VDB restored(demo_config());
    EXPECT(load_snapshot(restored, path) == 99);
    EXPECT(restored.size() == 4);

    for (int i = 0; i < 5; ++i) {
        Record got;
        if (i == 2) { EXPECT(!restored.get_metadata(ids[i], got)); continue; }
        ASSERT(restored.get_metadata(ids[i], got));
        EXPECT(got.attrs[0].text == (i % 2 ? "shoes" : "hats"));
        EXPECT(got.attrs[1].as_int() == i * 10);
        EXPECT(got.attrs[3].as_double() == i * 1.5);
        EXPECT(got.payload == bytes("pay" + std::to_string(i)));
    }
    std::filesystem::remove_all(dir);
}

TEST(snapshot_rejects_a_changed_schema) {
    const std::string dir = temp_dir("snapfp");
    std::filesystem::create_directories(dir);
    const std::string path = dir + "/snapshot";
    {
        VDB db(demo_config());
        const float v[2] = {1.0f, 1.0f};
        db.insert(v, row("shoes", 1, true, 1.0));
        save_snapshot(db, path, 1);
    }

    VDBConfig changed = demo_config();
    changed.schema[1].name = "cost";  // same types, different name -> different layout
    VDB other(changed);

    bool threw = false;
    try {
        load_snapshot(other, path);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    EXPECT(threw);
    std::filesystem::remove_all(dir);
}

TEST(snapshot_rejects_a_schema_that_only_flipped_indexed) {
    const std::string dir = temp_dir("snapfpidx");
    std::filesystem::create_directories(dir);
    const std::string path = dir + "/snapshot";
    {
        VDB db(demo_config());  // price (attr 1, Int64) not indexed
        const float v[2] = {1.0f, 1.0f};
        db.insert(v, row("shoes", 1, true, 1.0));
        save_snapshot(db, path, 1);
    }

    VDBConfig changed = demo_config();
    changed.schema[1].indexed = true;  // same name/type, only indexedness differs
    VDB other(changed);

    bool threw = false;
    try {
        load_snapshot(other, path);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    EXPECT(threw);
    std::filesystem::remove_all(dir);
}

TEST(wal_record_round_trips_metadata) {
    WalRecord in{WalRecordType::Insert, /*lsn=*/7, /*ext_id=*/3, {1.0f, 2.0f}, {}};
    in.meta = row("shoes", -8999, true, 4.5, bytes("blob"));

    const auto frame = encode_record(in);
    WalRecord out;
    size_t    consumed = 0;
    ASSERT(decode_record(frame.data(), frame.size(), out, consumed) == DecodeStatus::Ok);
    EXPECT(consumed == frame.size());
    EXPECT(out.vec == in.vec);
    ASSERT(out.meta.attrs.size() == 4);
    EXPECT(out.meta.attrs[0].type == AttrType::Tag);
    EXPECT(out.meta.attrs[0].text == "shoes");
    EXPECT(out.meta.attrs[1].as_int() == -8999);
    EXPECT(out.meta.attrs[2].as_bool() == true);
    EXPECT(out.meta.attrs[3].as_double() == 4.5);
    EXPECT(out.meta.payload == bytes("blob"));
}

TEST(wal_setmeta_record_round_trips_without_a_vector) {
    WalRecord in{WalRecordType::SetMeta, 11, 4, {}, {}};
    in.meta = row("hats", 42, false, 0.25);

    const auto frame = encode_record(in);
    WalRecord out;
    size_t    consumed = 0;
    ASSERT(decode_record(frame.data(), frame.size(), out, consumed) == DecodeStatus::Ok);
    EXPECT(out.type == WalRecordType::SetMeta);
    EXPECT(out.vec.empty());
    EXPECT(out.meta.attrs[0].text == "hats");
    EXPECT(out.meta.attrs[1].as_int() == 42);
}

TEST(wal_null_attributes_round_trip) {
    WalRecord in{WalRecordType::Insert, 1, 1, {3.0f}, {}};
    in.meta.attrs = {attr_null(), attr_int(7), attr_null(), attr_null()};

    const auto frame = encode_record(in);
    WalRecord out;
    size_t    consumed = 0;
    ASSERT(decode_record(frame.data(), frame.size(), out, consumed) == DecodeStatus::Ok);
    ASSERT(out.meta.attrs.size() == 4);
    EXPECT(out.meta.attrs[0].is_null());
    EXPECT(out.meta.attrs[1].as_int() == 7);
    EXPECT(out.meta.attrs[3].is_null());
}

TEST(durable_recovery_restores_metadata) {
    const std::string dir = temp_dir("durable");
    std::vector<ExternalId> ids;
    {
        DurableVDB db(demo_config(), dir);
        for (int i = 0; i < 6; ++i) {
            const float v[2] = {static_cast<float>(i), 0.0f};
            ids.push_back(db.insert(v, row("c" + std::to_string(i), i * 3, i % 2 == 0,
                                           i * 0.25, bytes("b" + std::to_string(i)))));
        }
        // Exercise all three metadata write paths before the crash.
        const float moved[2] = {100.0f, 0.0f};
        ASSERT(db.update(ids[1], moved));                                  // carry forward
        ASSERT(db.set_metadata(ids[2], row("changed", 777, false, 9.5, bytes("new"))));
        ASSERT(db.remove(ids[3]));
        db.sync();
    }  // destructor only best-effort syncs; the WAL is on disk

    DurableVDB reopened(demo_config(), dir);
    EXPECT(reopened.size() == 5);

    Record got;
    ASSERT(reopened.get_metadata(ids[1], got));
    EXPECT(got.attrs[0].text == "c1");        // update() kept the old row
    EXPECT(got.attrs[1].as_int() == 3);
    EXPECT(got.payload == bytes("b1"));

    ASSERT(reopened.get_metadata(ids[2], got));
    EXPECT(got.attrs[0].text == "changed");   // SetMeta replayed
    EXPECT(got.attrs[1].as_int() == 777);
    EXPECT(got.payload == bytes("new"));

    EXPECT(!reopened.get_metadata(ids[3], got));  // deleted

    ASSERT(reopened.get_metadata(ids[5], got));
    EXPECT(got.attrs[0].text == "c5");
    EXPECT(got.attrs[3].as_double() == 5 * 0.25);

    std::filesystem::remove_all(dir);
}

TEST(durable_recovery_across_a_checkpoint_keeps_metadata) {
    const std::string dir = temp_dir("ckpt");
    std::vector<ExternalId> ids;
    {
        DurableVDB db(demo_config(), dir);
        for (int i = 0; i < 4; ++i) {
            const float v[2] = {static_cast<float>(i), 1.0f};
            ids.push_back(db.insert(v, row("pre" + std::to_string(i), i, true, i,
                                           bytes("pre" + std::to_string(i)))));
        }
        db.checkpoint();  // rows 0..3 now live in the snapshot
        for (int i = 4; i < 7; ++i) {
            const float v[2] = {static_cast<float>(i), 1.0f};
            ids.push_back(db.insert(v, row("post" + std::to_string(i), i, false, i,
                                           bytes("post" + std::to_string(i)))));
        }
        db.sync();  // rows 4..6 only in the WAL tail
    }

    DurableVDB reopened(demo_config(), dir);
    EXPECT(reopened.size() == 7);
    for (int i = 0; i < 7; ++i) {
        Record got;
        ASSERT(reopened.get_metadata(ids[i], got));
        const std::string want = (i < 4 ? "pre" : "post") + std::to_string(i);
        EXPECT(got.attrs[0].text == want);
        EXPECT(got.payload == bytes(want));
    }
    std::filesystem::remove_all(dir);
}
