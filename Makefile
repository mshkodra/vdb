CXX      ?= clang++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -Wpedantic
# pthreads is linked ahead of Stage 7 (concurrency), where the test suite will
# spin up reader/writer threads. Harmless before then.
CXXFLAGS += -pthread
# Distance metrics are templated functors inlined into each index (no std::function
# on the hot path), so the compiler can see the L2/IP loop at the call site.
# -ffast-math lets it treat FP addition as associative and vectorize the distance
# reduction; -march=native emits the widest SIMD this host supports. NOTE: -ffast-math
# changes distance values in the low bits, which can flip tie-breaks between
# near-equidistant neighbours — re-baseline recall numbers after this change.
CXXFLAGS += -ffast-math -march=native
AR       ?= ar

INCLUDE   := -Iinclude
SRC_DIR   := src
TEST_DIR  := tests
BENCH_DIR := bench
PROTO_DIR := proto
SERVER_DIR := server
BUILD     ?= build
TEST_BLD  := $(BUILD)/tests
BENCH_BLD := $(BUILD)/bench
PROTO_BLD := $(BUILD)/proto
SERVER_BLD := $(BUILD)/server

SRCS     := $(wildcard $(SRC_DIR)/*.cpp)
OBJS     := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD)/%.o,$(SRCS))
DEPS     := $(OBJS:.o=.d)

TEST_SRCS := $(wildcard $(TEST_DIR)/*.cpp)
TEST_OBJS := $(patsubst $(TEST_DIR)/%.cpp,$(TEST_BLD)/%.o,$(TEST_SRCS))
TEST_DEPS := $(TEST_OBJS:.o=.d)

BENCH_SRCS := $(wildcard $(BENCH_DIR)/*.cpp)
BENCH_OBJS := $(patsubst $(BENCH_DIR)/%.cpp,$(BENCH_BLD)/%.o,$(BENCH_SRCS))
BENCH_DEPS := $(BENCH_OBJS:.o=.d)

LIB       := $(BUILD)/libvdb.a
TEST_BIN  := $(BUILD)/run_tests
BENCH_BIN := $(BUILD)/run_bench

# gRPC service (Phase B, B6). Optional: only touched by `make server`, so a build
# without protobuf/grpc++ installed (`test`/`bench`/`all`) is unaffected. Uses
# pkg-config rather than hardcoded -l flags — grpc++ pulls in a long, version-
# dependent chain of abseil libs that isn't worth hand-maintaining here.
PROTOC          ?= protoc
GRPC_CPP_PLUGIN ?= $(shell which grpc_cpp_plugin)
GRPC_LIBS       := $(shell pkg-config --libs protobuf grpc++ 2>/dev/null)
# -isystem, not -I, for the include search path itself: grpc/protobuf's own
# headers trigger hundreds of -Wall -Wextra -Wpedantic warnings that have nothing
# to do with this project's code, and -isystem suppresses warnings *inside* those
# headers. It doesn't reach macros that are defined in a system header but
# expanded in *our* generated .cc file (protoc's own output uses absl's
# nullability-annotated CHECK macros) — Clang attributes those to the expansion
# site, not the definition site — so the two specific warning categories that
# still leak through are named explicitly instead.
GRPC_CFLAGS := $(patsubst -I%,-isystem%,$(shell pkg-config --cflags protobuf grpc++ 2>/dev/null)) \
              -Wno-nullability-extension -Wno-deprecated-declarations

PROTO_OBJS  := $(PROTO_BLD)/vdb.pb.o $(PROTO_BLD)/vdb.grpc.pb.o
PROTO_DEPS  := $(PROTO_OBJS:.o=.d)
SERVER_SRCS := $(wildcard $(SERVER_DIR)/*.cpp)
SERVER_OBJS := $(patsubst $(SERVER_DIR)/%.cpp,$(SERVER_BLD)/%.o,$(SERVER_SRCS))
SERVER_DEPS := $(SERVER_OBJS:.o=.d)
SERVER_BIN  := $(BUILD)/vdb_server

.PHONY: all clean test bench test-tsan test-asan server

# Sanitized test builds (Stage 7). Sanitizers must be present at both compile and
# link, so they ride in CXXFLAGS (both rules use it). Each uses its own BUILD dir so
# artifacts never mix with the -O2 build. -pthread is here because overriding
# CXXFLAGS on the command line suppresses the makefile's own `+= -pthread`.
SAN_CXXFLAGS := -std=c++17 -g -O1 -Wall -Wextra -Wpedantic -pthread -fno-omit-frame-pointer

test-tsan:
	$(MAKE) test BUILD=build-tsan CXXFLAGS="$(SAN_CXXFLAGS) -fsanitize=thread"

test-asan:
	$(MAKE) test BUILD=build-asan CXXFLAGS="$(SAN_CXXFLAGS) -fsanitize=address"

all: $(LIB)

$(LIB): $(OBJS)
	$(AR) rcs $@ $^

$(BUILD)/%.o: $(SRC_DIR)/%.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) $(INCLUDE) -MMD -MP -c $< -o $@

$(TEST_BLD)/%.o: $(TEST_DIR)/%.cpp | $(TEST_BLD)
	$(CXX) $(CXXFLAGS) $(INCLUDE) -I$(TEST_DIR) -MMD -MP -c $< -o $@

$(TEST_BIN): $(TEST_OBJS) $(LIB)
	$(CXX) $(CXXFLAGS) $(TEST_OBJS) $(LIB) -o $@

test: $(TEST_BIN)
	./$(TEST_BIN)

$(BENCH_BLD)/%.o: $(BENCH_DIR)/%.cpp | $(BENCH_BLD)
	$(CXX) $(CXXFLAGS) $(INCLUDE) -MMD -MP -c $< -o $@

$(BENCH_BIN): $(BENCH_OBJS) $(LIB)
	$(CXX) $(CXXFLAGS) $(BENCH_OBJS) $(LIB) -o $@

bench: $(BENCH_BIN)
	./$(BENCH_BIN)

# protoc emits both the message types (vdb.pb.{h,cc}) and the service stubs
# (vdb.grpc.pb.{h,cc}) from one invocation; declaring all four as the rule's
# output means `make` only re-runs protoc once even though two targets need it.
$(PROTO_BLD)/vdb.pb.cc $(PROTO_BLD)/vdb.pb.h $(PROTO_BLD)/vdb.grpc.pb.cc $(PROTO_BLD)/vdb.grpc.pb.h: \
    $(PROTO_DIR)/vdb.proto | $(PROTO_BLD)
	$(PROTOC) -I $(PROTO_DIR) --cpp_out=$(PROTO_BLD) --grpc_out=$(PROTO_BLD) \
	  --plugin=protoc-gen-grpc=$(GRPC_CPP_PLUGIN) $(PROTO_DIR)/vdb.proto

# Generated code, not hand-written — compiled with the same CXXFLAGS as everything
# else (no -Werror anywhere in this build, so any generated-code warnings are
# harmless noise, not a build blocker).
$(PROTO_BLD)/%.o: $(PROTO_BLD)/%.cc | $(PROTO_BLD)
	$(CXX) $(CXXFLAGS) $(GRPC_CFLAGS) -I$(PROTO_BLD) -MMD -MP -c $< -o $@

$(SERVER_BLD)/%.o: $(SERVER_DIR)/%.cpp $(PROTO_BLD)/vdb.grpc.pb.h | $(SERVER_BLD)
	$(CXX) $(CXXFLAGS) $(INCLUDE) $(GRPC_CFLAGS) -I$(PROTO_BLD) -MMD -MP -c $< -o $@

$(SERVER_BIN): $(SERVER_OBJS) $(PROTO_OBJS) $(LIB)
	$(CXX) $(CXXFLAGS) $(SERVER_OBJS) $(PROTO_OBJS) $(LIB) $(GRPC_LIBS) -o $@

server: $(SERVER_BIN)

$(BUILD):
	mkdir -p $@

$(TEST_BLD): | $(BUILD)
	mkdir -p $@

$(BENCH_BLD): | $(BUILD)
	mkdir -p $@

$(PROTO_BLD): | $(BUILD)
	mkdir -p $@

$(SERVER_BLD): | $(BUILD)
	mkdir -p $@

clean:
	rm -rf build build-tsan build-asan

-include $(DEPS)
-include $(TEST_DEPS)
-include $(BENCH_DEPS)
-include $(PROTO_DEPS)
-include $(SERVER_DEPS)
