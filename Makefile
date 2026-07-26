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
BUILD     ?= build
TEST_BLD  := $(BUILD)/tests
BENCH_BLD := $(BUILD)/bench

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

.PHONY: all clean test bench test-tsan test-asan

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

$(BUILD):
	mkdir -p $@

$(TEST_BLD): | $(BUILD)
	mkdir -p $@

$(BENCH_BLD): | $(BUILD)
	mkdir -p $@

clean:
	rm -rf build build-tsan build-asan

-include $(DEPS)
-include $(TEST_DEPS)
-include $(BENCH_DEPS)
