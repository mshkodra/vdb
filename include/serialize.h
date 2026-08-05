#pragma once
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace vdb {

// Minimal little-endian binary serialization used by snapshots and index graphs.

template <typename T>
void put(std::vector<uint8_t>& buf, T v) {
    static_assert(std::is_trivially_copyable<T>::value, "put requires trivial type");
    const auto* p = reinterpret_cast<const uint8_t*>(&v);
    buf.insert(buf.end(), p, p + sizeof(T));
}

inline void put_floats(std::vector<uint8_t>& buf, const std::vector<float>& v) {
    put<uint64_t>(buf, v.size());
    const auto* p = reinterpret_cast<const uint8_t*>(v.data());
    buf.insert(buf.end(), p, p + v.size() * sizeof(float));
}

inline void put_bytes(std::vector<uint8_t>& buf, const std::vector<uint8_t>& v) {
    put<uint64_t>(buf, v.size());
    buf.insert(buf.end(), v.begin(), v.end());
}

inline void put_string(std::vector<uint8_t>& buf, const std::string& s) {
    put<uint64_t>(buf, s.size());
    buf.insert(buf.end(), s.begin(), s.end());
}

// Cursor over a byte buffer. Throws if a read runs past the end (truncated file).
class Reader {
public:
    Reader(const uint8_t* data, size_t len) : p_(data), end_(data + len) {}

    template <typename T>
    T get() {
        static_assert(std::is_trivially_copyable<T>::value, "get requires trivial type");
        if (p_ + sizeof(T) > end_) throw std::runtime_error("snapshot truncated");
        T v;
        std::memcpy(&v, p_, sizeof(T));
        p_ += sizeof(T);
        return v;
    }

    std::vector<float> get_floats() {
        const uint64_t n = get<uint64_t>();
        std::vector<float> v(n);
        if (n) {
            if (p_ + n * sizeof(float) > end_) throw std::runtime_error("snapshot truncated");
            std::memcpy(v.data(), p_, n * sizeof(float));
            p_ += n * sizeof(float);
        }
        return v;
    }

    std::vector<uint8_t> get_bytes() {
        const uint64_t n = get<uint64_t>();
        if (p_ + n > end_) throw std::runtime_error("snapshot truncated");
        std::vector<uint8_t> v(p_, p_ + n);
        p_ += n;
        return v;
    }

    std::string get_string() {
        const uint64_t n = get<uint64_t>();
        if (p_ + n > end_) throw std::runtime_error("snapshot truncated");
        std::string s(reinterpret_cast<const char*>(p_), n);
        p_ += n;
        return s;
    }

private:
    const uint8_t* p_;
    const uint8_t* end_;
};

}
