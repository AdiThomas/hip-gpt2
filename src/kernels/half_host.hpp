#pragma once
// Host-only IEEE-754 binary32 <-> binary16 bit conversion (no HIP headers, no hardware
// half type). The bench uses this to build FP16 inputs deterministically on every
// platform, and to convert the *rounded* inputs back to float for the CPU reference,
// so FP16 kernels are judged only on accumulation error, not on input rounding.
#include <cstdint>
#include <cstring>

// float -> half bits, round-to-nearest-even. Handles subnormals, overflow to inf, NaN.
inline uint16_t f32_to_f16_bits(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof x);
    const uint32_t sign = (x >> 16) & 0x8000u;
    const uint32_t absx = x & 0x7FFFFFFFu;

    if (absx >= 0x7F800000u)                                // inf or NaN (keep NaN quiet)
        return static_cast<uint16_t>(sign | 0x7C00u | (absx > 0x7F800000u ? 0x0200u : 0u));
    if (absx >= 0x477FF000u)                                // >= 65520 rounds to inf
        return static_cast<uint16_t>(sign | 0x7C00u);

    if (absx < 0x38800000u) {                               // below 2^-14: half subnormal or 0
        if (absx <= 0x33000000u) return static_cast<uint16_t>(sign);  // <= 2^-25: ties-to-even 0
        const uint32_t e = absx >> 23;                      // biased f32 exponent, 102..112
        const uint32_t m = (absx & 0x7FFFFFu) | 0x800000u;  // implicit 1
        // value = m * 2^(e-150); in units of the half subnormal step 2^-24 that is
        // m * 2^(e-126), i.e. a right shift by (126 - e) in [14, 24].
        const uint32_t shift = 126u - e;
        uint32_t q = m >> shift;
        const uint32_t rem = m & ((1u << shift) - 1u), halfway = 1u << (shift - 1u);
        if (rem > halfway || (rem == halfway && (q & 1u))) ++q;   // may carry into 0x400 (min normal): correct
        return static_cast<uint16_t>(sign | q);
    }

    // Normal range: rebias exponent (127 -> 15), keep 10 of 23 mantissa bits.
    const uint32_t e = (absx >> 23) - 127u + 15u;           // 1..30
    const uint32_t m = absx & 0x7FFFFFu;
    uint32_t q = (e << 10) | (m >> 13);
    const uint32_t rem = m & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (q & 1u))) ++q; // carry into exponent is correct
    return static_cast<uint16_t>(sign | q);
}

// half bits -> float (exact: every binary16 value is representable in binary32).
inline float f16_bits_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    const uint32_t e = (h >> 10) & 0x1Fu;
    uint32_t m = h & 0x3FFu;
    uint32_t x;
    if (e == 0) {
        if (m == 0) {
            x = sign;                                       // +-0
        } else {                                            // subnormal: normalize
            int shift = 0;
            while (!(m & 0x400u)) { m <<= 1; ++shift; }
            m &= 0x3FFu;
            x = sign | ((uint32_t)(127 - 15 + 1 - shift) << 23) | (m << 13);
        }
    } else if (e == 0x1F) {
        x = sign | 0x7F800000u | (m << 13);                 // inf / NaN
    } else {
        x = sign | ((e - 15u + 127u) << 23) | (m << 13);
    }
    float f;
    std::memcpy(&f, &x, sizeof f);
    return f;
}
