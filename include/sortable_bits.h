#pragma once
#include <cstdint>
#include <cstring>

namespace vdb {

// Order-preserving bijections int64_t/double <-> uint64_t: unsigned comparison on
// the transformed bits agrees with the natural (signed int / IEEE-754) comparison
// on the original values. This is what lets a future B-tree key on Int64/Float64
// columns be a plain uint64_t compared with `<`, with no per-type comparator and no
// branch in the traversal's hot loop — same "one indexed load, one compare" shape
// MetadataStore's columns already use, just for an ordered index instead of equality.
//
// int64_t: two's complement already puts INT64_MIN..INT64_MAX in ascending order as
// a *signed* comparison; the only mismatch against *unsigned* comparison is the sign
// bit (a negative number has it set, making it look larger than any positive number
// to an unsigned compare). Flipping just that one bit fixes it: negatives drop below
// positives, and two's-complement order within each half is already unsigned-
// monotonic, so nothing else needs to move.
//
// double (IEEE-754): a positive double's raw bits already compare correctly as an
// unsigned integer — exponent then mantissa both increase with magnitude, and the
// sign bit is 0 for all of them. A negative double's raw bits compare *backwards*:
// larger magnitude (more negative) produces a larger exponent/mantissa, so unsigned
// comparison would rank -1.0 above -2.0. The classic fix (Stereopsis' "radix
// tricks", see docs/plans/LEARNING_RESOURCES.md): if the sign bit is 0, flip only
// the sign bit (lifts positives above all negatives); if the sign bit is 1, flip
// every bit (both reverses the backwards order among negatives and drops the sign
// bit to 0, keeping them below all positives).
//
// NaN has no total order under IEEE-754 itself, so "where a NaN sorts" is not a
// question with a correct answer — this transform still produces a well-defined
// uint64_t for it (round-trips bit-for-bit), it just isn't meaningfully ordered
// relative to other values. Don't index a column that can contain NaN and expect
// range queries over it to mean anything.

inline uint64_t sortable_bits(int64_t v) {
    uint64_t u;
    std::memcpy(&u, &v, sizeof(u));
    return u ^ 0x8000000000000000ULL;
}

inline uint64_t sortable_bits(double v) {
    uint64_t u;
    std::memcpy(&u, &v, sizeof(u));
    const uint64_t mask = (u >> 63) ? 0xFFFFFFFFFFFFFFFFULL : 0x8000000000000000ULL;
    return u ^ mask;
}

// Inverse of sortable_bits(): from_sortable_bits<int64_t>(sortable_bits(v)) == v,
// same for double. Explicit specializations rather than a generic body so an
// unsupported T fails to compile instead of silently reinterpreting bits.
template <class T>
T from_sortable_bits(uint64_t bits);

template <>
inline int64_t from_sortable_bits<int64_t>(uint64_t bits) {
    const uint64_t u = bits ^ 0x8000000000000000ULL;
    int64_t        v;
    std::memcpy(&v, &u, sizeof(v));
    return v;
}

template <>
inline double from_sortable_bits<double>(uint64_t bits) {
    // Mirrors sortable_bits(double)'s mask choice, but keyed off the *encoded*
    // sign bit: encoding flips a positive's sign bit to 1 and a negative's to 0
    // (full complement), so the two cases swap on decode.
    const uint64_t mask = (bits >> 63) ? 0x8000000000000000ULL : 0xFFFFFFFFFFFFFFFFULL;
    const uint64_t u    = bits ^ mask;
    double         v;
    std::memcpy(&v, &u, sizeof(v));
    return v;
}

}
