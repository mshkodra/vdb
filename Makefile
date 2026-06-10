CXX      ?= clang++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -Wpedantic
AR       ?= ar

INCLUDE   := -Iinclude
SRC_DIR   := src
TEST_DIR  := tests
BENCH_DIR := bench
BUILD     := build
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

.PHONY: all clean test bench

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
	rm -rf $(BUILD)

-include $(DEPS)
-include $(TEST_DEPS)
-include $(BENCH_DEPS)
