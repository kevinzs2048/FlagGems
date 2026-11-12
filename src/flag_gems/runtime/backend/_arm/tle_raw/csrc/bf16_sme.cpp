// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
// BF16 inputs/output, FP32 accumulation. Apple SME2, 512-bit streaming VL.
// Called synchronously by the CPU Triton TLE Raw wrapper.
#include <arm_sme.h>
#include <cstdint>
#include <cstring>
#include <omp.h>
#include <type_traits>
#include <vector>

extern "C" __arm_locally_streaming
int flag_gems_arm_bf16_sme_vector_words() { return svcntsw(); }

template<int MR, int NR, bool F32, bool NSTRIPE>
__arm_new("za") __arm_locally_streaming
static void compute_range(const void* a, const void* w, float* out,
                          int64_t kp, int64_t np, int64_t nt, int64_t mt,
                          int64_t begin, int64_t end) {
    if (svcntsw() != 16) __builtin_trap();
    using Packed = std::conditional_t<F32, float, uint16_t>;
    const auto ph = svptrue_b16();
    const auto pf = svptrue_b32();
    for (int64_t job = begin; job < end; ++job) {
        const auto mi = NSTRIPE ? job % mt : job / nt;
        const auto ni = NSTRIPE ? job / mt : job % nt;
        const auto* ap = static_cast<const Packed*>(a) + mi * kp * MR;
        const auto* wp = static_cast<const Packed*>(w) + ni * kp * NR;
        svzero_za();
        if constexpr (F32) {
            static_assert(MR == 16 && NR == 64);
            for (int64_t k = 0; k < kp; ++k) {
                const auto av = svld1_f32(pf, ap + k * 16);
                const auto* bp = wp + k * 64;
                svmopa_za32_f32_m(0, pf, pf, av, svld1_f32(pf, bp));
                svmopa_za32_f32_m(1, pf, pf, av, svld1_f32(pf, bp + 16));
                svmopa_za32_f32_m(2, pf, pf, av, svld1_f32(pf, bp + 32));
                svmopa_za32_f32_m(3, pf, pf, av, svld1_f32(pf, bp + 48));
            }
        } else {
            for (int64_t pair = 0; pair < kp / 2; ++pair) {
                const auto av = svreinterpret_bf16_u16(svld1_u16(ph, ap + pair * MR * 2));
                const auto* bp = wp + pair * NR * 2;
                const auto b0 = svreinterpret_bf16_u16(svld1_u16(ph, bp));
                const auto b1 = svreinterpret_bf16_u16(svld1_u16(ph, bp + 32));
                svmopa_za32_bf16_m(0, ph, ph, av, b0);
                svmopa_za32_bf16_m(1, ph, ph, av, b1);
                if constexpr (MR == 32) {
                    const auto a1 = svreinterpret_bf16_u16(svld1_u16(ph, ap + pair * 64 + 32));
                    svmopa_za32_bf16_m(2, ph, ph, a1, b0);
                    svmopa_za32_bf16_m(3, ph, ph, a1, b1);
                } else {
                    const auto b2 = svreinterpret_bf16_u16(svld1_u16(ph, bp + 64));
                    const auto b3 = svreinterpret_bf16_u16(svld1_u16(ph, bp + 96));
                    svmopa_za32_bf16_m(2, ph, ph, av, b2);
                    svmopa_za32_bf16_m(3, ph, ph, av, b3);
                }
            }
        }
#define STORE_TILE(T, R, C) \
        for (int row = 0; row < 16; row += 4) { \
            const auto four = svread_hor_za32_f32_vg4(T, row); \
            auto* dst = out + (mi * MR + R + row) * np + ni * NR + C; \
            svst1_f32(pf, dst, svget4_f32(four, 0)); \
            svst1_f32(pf, dst + np, svget4_f32(four, 1)); \
            svst1_f32(pf, dst + 2 * np, svget4_f32(four, 2)); \
            svst1_f32(pf, dst + 3 * np, svget4_f32(four, 3)); \
        }
        STORE_TILE(0, 0, 0)
        STORE_TILE(1, 0, 16)
        if constexpr (MR == 32) {
            STORE_TILE(2, 16, 0)
            STORE_TILE(3, 16, 16)
        } else {
            STORE_TILE(2, 0, 32)
            STORE_TILE(3, 0, 48)
        }
#undef STORE_TILE
    }
}

// Match the installed Torch Arm BF16 dot's 32 independent FP32 sums.
// This is an opt-in numerical compatibility route, not a generic Torch ABI:
// a different framework build can select a different reduction schedule.
__arm_new("za") __arm_locally_streaming
static void compute_torch_order(const float* a, const float* w, float* out,
                                float* partial, int64_t k, int64_t np,
                                int64_t mt, int64_t begin, int64_t end) {
    if (svcntsw() != 16) __builtin_trap();
    constexpr int MR = 16, NR = 64, TILE = MR * NR;
    const int64_t groups = (k + 31) / 32;
    const int64_t kp = groups * 32;
    const int64_t main_groups = k / 32;
    const int64_t tail_vectors = (k % 32) / 8;
    const auto pf = svptrue_b32();
    for (int64_t job = begin; job < end; ++job) {
        const int64_t mi = job % mt, ni = job / mt;
        const float* ap = a + mi * kp * MR;
        const float* wp = w + ni * kp * NR;
        for (int phase = 0; phase < 36; ++phase) {
            svzero_za();
            const int64_t steps = phase < 32 ? main_groups : tail_vectors * 2;
            for (int64_t step = 0; step < steps; ++step) {
                // Main lanes advance by 32. Tail lanes accumulate low four,
                // then high four, in the same register for each group of 8.
                const int64_t col = phase < 32
                    ? phase * groups + step
                    : ((phase - 32) + step * 4) * groups + main_groups;
                const auto av = svld1_f32(pf, ap + col * MR);
                const float* bp = wp + col * NR;
                svmopa_za32_f32_m(0, pf, pf, av, svld1_f32(pf, bp));
                svmopa_za32_f32_m(1, pf, pf, av, svld1_f32(pf, bp + 16));
                svmopa_za32_f32_m(2, pf, pf, av, svld1_f32(pf, bp + 32));
                svmopa_za32_f32_m(3, pf, pf, av, svld1_f32(pf, bp + 48));
            }
#define STORE_PART(T, C) \
            for (int row = 0; row < MR; row += 4) { \
                const auto four = svread_hor_za32_f32_vg4(T, row); \
                float* dst = partial + phase * TILE + row * NR + C; \
                svst1_f32(pf, dst, svget4_f32(four, 0)); \
                svst1_f32(pf, dst + NR, svget4_f32(four, 1)); \
                svst1_f32(pf, dst + 2 * NR, svget4_f32(four, 2)); \
                svst1_f32(pf, dst + 3 * NR, svget4_f32(four, 3)); \
            }
            STORE_PART(0, 0)
            STORE_PART(1, 16)
            STORE_PART(2, 32)
            STORE_PART(3, 48)
#undef STORE_PART
        }
        for (int row = 0; row < MR; ++row) {
            for (int col = 0; col < NR; col += 16) {
                const int offset = row * NR + col;
#define P(L) svld1_f32(pf, partial + (L) * TILE + offset)
#define ADD(A, B) svadd_f32_x(pf, A, B)
#define LANE(L) ADD(ADD(ADD(P(L), P(16 + L)), ADD(P(8 + L), P(24 + L))), \
                    ADD(ADD(P(4 + L), P(20 + L)), ADD(P(12 + L), P(28 + L))))
                const auto l0 = LANE(0), l1 = LANE(1);
                const auto l2 = LANE(2), l3 = LANE(3);
                auto value = ADD(ADD(l0, l1), ADD(l2, l3));
                const auto tail = ADD(ADD(P(32), P(33)), ADD(P(34), P(35)));
                value = ADD(value, tail);
#undef LANE
#undef ADD
#undef P
                for (int64_t rem = k / 8 * 8; rem < k; ++rem) {
                    const int64_t index = (rem % 32) * groups + rem / 32;
                    value = svmla_f32_x(pf, value,
                        svdup_f32(ap[index * MR + row]),
                        svld1_f32(pf, wp + index * NR + col));
                }
                svst1_f32(pf, out + (mi * MR + row) * np + ni * NR + col, value);
            }
        }
    }
}

template<int MR, int NR, bool F32, bool NSTRIPE = false, bool TORCH_ORDER = false>
static void run(uint64_t address) {
    using Packed = std::conditional_t<F32, float, uint16_t>;
    const auto* d = reinterpret_cast<const int64_t*>(address);
    const auto* x = reinterpret_cast<const uint16_t*>(d[0]);
    const auto* w = reinterpret_cast<const Packed*>(d[1]);
    auto* packed_a = reinterpret_cast<Packed*>(d[2]);
    auto* scratch = reinterpret_cast<float*>(d[3]);
    auto* y = reinterpret_cast<uint16_t*>(d[4]);
    const int64_t m = d[5], n = d[6], k = d[7];
    const int threads = int(d[8]);
    const auto* bias = reinterpret_cast<const uint16_t*>(d[9]);
    const int64_t mt = (m + MR - 1) / MR, nt = (n + NR - 1) / NR;
    const int64_t kp = TORCH_ORDER ? (k + 31) / 32 * 32 : F32 ? k : (k + 1) / 2 * 2;
    const int64_t np = nt * NR;
#pragma omp parallel num_threads(threads)
    {
#pragma omp for schedule(static)
        for (int64_t tile = 0; tile < mt; ++tile) {
            if constexpr (F32) {
                for (int64_t col = 0; col < kp; ++col) {
                    for (int64_t row = 0; row < MR; ++row) {
                        const auto r = tile * MR + row;
                        uint32_t bits = r < m && col < k ? uint32_t(x[r * k + col]) << 16 : 0;
                        const int64_t packed_col = TORCH_ORDER ? (col % 32) * (kp / 32) + col / 32 : col;
                        std::memcpy(packed_a + tile * kp * MR + packed_col * MR + row, &bits, sizeof(bits));
                    }
                }
            } else {
                for (int64_t pair = 0; pair < kp / 2; ++pair) {
                    auto* dst = packed_a + tile * kp * MR + pair * MR * 2;
                    for (int64_t row = 0; row < MR; ++row) {
                        const int64_t r = tile * MR + row;
                        dst[row * 2] = r < m ? x[r * k + pair * 2] : 0;
                        dst[row * 2 + 1] = r < m && pair * 2 + 1 < k ? x[r * k + pair * 2 + 1] : 0;
                    }
                }
            }
        }
        const int64_t workers = omp_get_num_threads(), tid = omp_get_thread_num();
        if constexpr (TORCH_ORDER) {
            std::vector<float> partial(36 * 16 * 64);
            compute_torch_order(packed_a, w, scratch, partial.data(), k, np, mt,
                               mt * nt * tid / workers, mt * nt * (tid + 1) / workers);
        } else {
            compute_range<MR, NR, F32, NSTRIPE>(packed_a, w, scratch, kp, np, nt, mt,
                                      mt * nt * tid / workers, mt * nt * (tid + 1) / workers);
        }
#pragma omp barrier
#pragma omp for schedule(static)
        for (int64_t row = 0; row < m; ++row) {
            for (int64_t col = 0; col < n; ++col) {
                uint32_t bits;
                float value = scratch[row * np + col];
                if (bias) {
                    const uint32_t bias_bits = uint32_t(bias[col]) << 16;
                    float b;
                    std::memcpy(&b, &bias_bits, sizeof(b));
                    value += b;
                }
                std::memcpy(&bits, &value, sizeof(bits));
                const auto rounded = bits + 0x7fffu + ((bits >> 16) & 1u);
                y[row * n + col] = (bits & 0x7fffffffu) > 0x7f800000u
                    ? uint16_t((bits >> 16) | 0x40u) : uint16_t(rounded >> 16);
            }
        }
    }
}

extern "C" void flag_gems_arm_bf16_sme(uint64_t address) { run<32, 32, false>(address); }
extern "C" void flag_gems_arm_bf16_sme_16x64(uint64_t address) { run<16, 64, false>(address); }
extern "C" void flag_gems_arm_bf16_sme_16x64_striped(uint64_t address) { run<16, 64, false, true>(address); }
// Converts BF16 operands exactly to FP32; no FP16 conversion or quantization.
extern "C" void flag_gems_arm_bf16_sme_f32_16x64(uint64_t address) { run<16, 64, true>(address); }

extern "C" void flag_gems_arm_bf16_sme_torch_16x64(uint64_t address) { run<16, 64, true, true, true>(address); }

// Precision-neutral Q/K/V packing for equal-length vision chunks.
// Each work item copies one image/head of one tensor, keeping row copies
// contiguous and avoiding masked gathers. Called through CPU Triton Raw.
template<int FIXED_D>
static void vision_pack(uint64_t address) {
    const auto* d = reinterpret_cast<const int64_t*>(address);
    const int64_t sequence = d[6], heads = d[7], dim = d[8], images = d[9];
    const int threads = int(d[16]);
    const int64_t width = FIXED_D ? FIXED_D : dim;
#pragma omp parallel for num_threads(threads) schedule(static)
    for (int64_t work = 0; work < 3 * images * ((sequence + 15) / 16); ++work) {
        const int64_t blocks = (sequence + 15) / 16;
        const int64_t tensor = work / (images * blocks);
        const int64_t image = work / blocks % images;
        const int64_t begin = work % blocks * 16;
        const int64_t end = begin + 16 < sequence ? begin + 16 : sequence;
        const auto* source = reinterpret_cast<const uint16_t*>(d[tensor]);
        auto* output = reinterpret_cast<uint16_t*>(d[3 + tensor]);
        const int64_t head_stride = d[10 + tensor * 2];
        const int64_t token_stride = d[11 + tensor * 2];
        for (int64_t head = 0; head < heads; ++head) {
            const auto* src = source + head * head_stride + image * sequence * token_stride;
            auto* dst = output + (image * heads + head) * sequence * width;
            for (int64_t row = begin; row < end; ++row) {
                std::memcpy(dst + row * width, src + row * token_stride,
                            width * sizeof(uint16_t));
            }
        }
    }
}

extern "C" void flag_gems_arm_vision_pack(uint64_t address) {
    const int64_t dim = reinterpret_cast<const int64_t*>(address)[8];
    if (dim == 72) vision_pack<72>(address);
    else if (dim == 64) vision_pack<64>(address);
    else if (dim == 128) vision_pack<128>(address);
    else vision_pack<0>(address);
}
