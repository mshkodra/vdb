// gRPC service (Phase B, B6 — docs/plans/HYBRID_SEARCH_PLAN.md, kept local) wrapping
// a DurableVDB fixed to the same schema bench/hybrid.cpp already uses:
// description (Text) + category (Tag), dim=128. Not a general schema-configurable
// service — that's a bigger feature than what's built here.
//
// Insecure channel credentials (no TLS) — this is meant for a trusted local
// network, not the public internet. Binds 0.0.0.0 (not just localhost) so it's
// reachable from other machines on that network, per the "run this on one of my
// local hosts" use case.
#include "vdb.grpc.pb.h"
#include "vdb.pb.h"

#include "durable_vdb.h"

#include <grpcpp/grpcpp.h>

#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr size_t DIM = 128;

std::vector<vdb::AttrSpec> minecraft_schema() {
    return {{"description", vdb::AttrType::Text}, {"category", vdb::AttrType::Tag}};
}

grpc::Status bad_dim_(size_t got, size_t want) {
    return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                        "vector has " + std::to_string(got) + " dims, server expects " +
                            std::to_string(want));
}

class VDBServiceImpl final : public vdbproto::VDBService::Service {
public:
    explicit VDBServiceImpl(vdb::DurableVDB& db) : db_(db) {}

    grpc::Status Insert(grpc::ServerContext*, const vdbproto::InsertRequest* req,
                       vdbproto::InsertResponse* resp) override {
        if (static_cast<size_t>(req->vector_size()) != db_.dim())
            return bad_dim_(req->vector_size(), db_.dim());

        vdb::Record rec;
        rec.attrs = {vdb::attr_text(req->description()), vdb::attr_tag(req->category())};
        const ExternalId id = db_.insert(req->vector().data(), rec);
        resp->set_id(id);
        return grpc::Status::OK;
    }

    grpc::Status Remove(grpc::ServerContext*, const vdbproto::RemoveRequest* req,
                       vdbproto::RemoveResponse* resp) override {
        resp->set_removed(db_.remove(req->id()));
        return grpc::Status::OK;
    }

    grpc::Status SearchVector(grpc::ServerContext*, const vdbproto::SearchVectorRequest* req,
                             vdbproto::SearchResponse* resp) override {
        if (static_cast<size_t>(req->vector_size()) != db_.dim())
            return bad_dim_(req->vector_size(), db_.dim());
        fill_hits_(db_.search_hits(req->vector().data(), req->k()), resp);
        return grpc::Status::OK;
    }

    grpc::Status SearchText(grpc::ServerContext*, const vdbproto::SearchTextRequest* req,
                           vdbproto::SearchResponse* resp) override {
        try {
            const std::vector<vdb::Hit> hits =
                req->category().empty()
                    ? db_.search_text(0, req->query(), req->k())
                    : db_.search_text(0, req->query(), req->k(),
                                      vdb::pred_eq(1, vdb::attr_tag(req->category())));
            fill_hits_(hits, resp);
            return grpc::Status::OK;
        } catch (const std::exception& e) {
            return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, e.what());
        }
    }

    grpc::Status SearchHybrid(grpc::ServerContext*, const vdbproto::SearchHybridRequest* req,
                             vdbproto::SearchResponse* resp) override {
        if (static_cast<size_t>(req->vector_size()) != db_.dim())
            return bad_dim_(req->vector_size(), db_.dim());
        try {
            const std::vector<vdb::Hit> hits =
                req->category().empty()
                    ? db_.search_hybrid(req->vector().data(), 0, req->query(), req->k(),
                                        req->depth())
                    : db_.search_hybrid(req->vector().data(), 0, req->query(), req->k(),
                                        vdb::pred_eq(1, vdb::attr_tag(req->category())),
                                        req->depth());
            fill_hits_(hits, resp);
            return grpc::Status::OK;
        } catch (const std::exception& e) {
            return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, e.what());
        }
    }

    grpc::Status GetMetadata(grpc::ServerContext*, const vdbproto::GetMetadataRequest* req,
                            vdbproto::GetMetadataResponse* resp) override {
        vdb::Record rec;
        const bool  found = db_.get_metadata(req->id(), rec);
        resp->set_found(found);
        if (found) {
            resp->set_description(rec.attrs[0].text);
            resp->set_category(rec.attrs[1].text);
        }
        return grpc::Status::OK;
    }

    grpc::Status Stats(grpc::ServerContext*, const vdbproto::StatsRequest*,
                      vdbproto::StatsResponse* resp) override {
        resp->set_size(db_.size());
        resp->set_deleted_count(db_.deleted_count());
        return grpc::Status::OK;
    }

private:
    // A vdb::Hit only carries a payload (unused by this schema), not attrs — so
    // description/category come from a metadata lookup per hit, same as
    // bench/hybrid.cpp's print_hits already does. Extra cost is K get_metadata()
    // calls per query, not something a scan-touching-every-candidate cost applies
    // to — same reasoning collect_'s own over-fetch margin doesn't apply here.
    void fill_hits_(const std::vector<vdb::Hit>& hits, vdbproto::SearchResponse* resp) {
        for (const auto& h : hits) {
            auto*       out = resp->add_hits();
            vdb::Record rec;
            out->set_id(h.id);
            out->set_score(h.dist);
            if (db_.get_metadata(h.id, rec)) {
                out->set_description(rec.attrs[0].text);
                out->set_category(rec.attrs[1].text);
            }
        }
    }

    vdb::DurableVDB& db_;
};

}  // namespace

int main(int argc, char** argv) {
    const std::string data_dir = argc > 1 ? argv[1] : "data/vdb_server";
    const std::string port     = argc > 2 ? argv[2] : "50051";

    vdb::VDBConfig cfg;
    cfg.kind   = vdb::IndexKind::HNSW;
    cfg.dim    = DIM;
    cfg.metric = vdb::Metric::L2;
    cfg.schema = minecraft_schema();

    std::printf("opening %s (dim=%zu, schema=description:Text,category:Tag)...\n",
               data_dir.c_str(), DIM);
    vdb::DurableVDB db(cfg, data_dir);
    std::printf("recovered N=%zu live rows\n", db.size());

    VDBServiceImpl service(db);
    grpc::ServerBuilder builder;
    const std::string addr = "0.0.0.0:" + port;
    builder.AddListeningPort(addr, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);

    std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
    std::printf("listening on %s (insecure channel — trusted local network only)\n", addr.c_str());
    server->Wait();
    return 0;
}
