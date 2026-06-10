#pragma once

#include <cstdio>
#include <functional>
#include <string>
#include <vector>

namespace vdb_test {

struct TestCase {
    const char* name;
    std::function<void(int&)> fn;
};

inline std::vector<TestCase>& registry() {
    static std::vector<TestCase> r;
    return r;
}

struct Registrar {
    Registrar(const char* name, std::function<void(int&)> fn) {
        registry().push_back({name, std::move(fn)});
    }
};

inline int run_all() {
    int passed = 0, failed = 0;
    for (const auto& t : registry()) {
        int local_failures = 0;
        std::printf("[ RUN      ] %s\n", t.name);
        try {
            t.fn(local_failures);
        } catch (const std::exception& e) {
            std::printf("    exception: %s\n", e.what());
            ++local_failures;
        } catch (...) {
            std::printf("    unknown exception\n");
            ++local_failures;
        }
        if (local_failures == 0) {
            std::printf("[       OK ] %s\n", t.name);
            ++passed;
        } else {
            std::printf("[  FAILED  ] %s\n", t.name);
            ++failed;
        }
    }
    std::printf("\n%d passed, %d failed\n", passed, failed);
    return failed == 0 ? 0 : 1;
}

}  // namespace vdb_test

#define TEST(name)                                                             \
    static void test_##name(int& _vdb_fail);                                   \
    static ::vdb_test::Registrar registrar_##name(#name, &test_##name);        \
    static void test_##name(int& _vdb_fail)

#define EXPECT(cond)                                                           \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::printf("    %s:%d: EXPECT(%s) failed\n",                      \
                        __FILE__, __LINE__, #cond);                            \
            ++_vdb_fail;                                                       \
        }                                                                      \
    } while (0)

#define ASSERT(cond)                                                           \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::printf("    %s:%d: ASSERT(%s) failed\n",                      \
                        __FILE__, __LINE__, #cond);                            \
            ++_vdb_fail;                                                       \
            return;                                                            \
        }                                                                      \
    } while (0)
