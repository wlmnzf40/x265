/*****************************************************************************
 * Copyright (C) 2024 MulticoreWare, Inc
 *
 * Authors: Optimization for ARM SVE vectorization
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02111, USA.
 *
 * This program is also available under a commercial proprietary license.
 * For more information, contact us at license @ x265.com.
 *****************************************************************************/

/*
 * SVE vertical luma filter (luma_vsp / second pass of luma_hvpp).
 *
 * Design rationale
 * ----------------
 * The horizontal pass is already well-optimised by the i8mm code path.
 * The bottleneck is the vertical (SP) pass on int16_t intermediate values.
 *
 * At VL=256 (svcntw()=8), svld1sh_s32 loads 8 int16→int32 in one instruction,
 * eliminating NEON's lo/hi split.  The inner loop is unrolled to 8 output rows
 * per iteration (vs NEON's 4) to expose 8 independent accumulator chains to the
 * CPU's out-of-order engine, better hiding the 7-instruction MLA dependency
 * chain (~28-cycle critical path per output row).
 *
 * Register usage (8-row block, VL=256): window w0..w14 = 15 Z-regs,
 * accumulators acc0..acc7 = 8 Z-regs, hoisted constants = 7 Z-regs → 30 total,
 * within the 32 Z-register architectural limit.
 *
 * The VL check in asm-primitives.cpp ensures this code is only registered when
 * svcntw() > 4 (VL > 128-bit), where the wider registers justify the overhead.
 */

#if defined(HAVE_SVE) && HAVE_SVE_BRIDGE

#include "filter-prim-sve.h"
#include "primitives.h"
#include "constants.h"

#if !HIGH_BIT_DEPTH

#include <arm_sve.h>
#include <arm_neon.h>

namespace X265_NS {

/* External declaration: horizontal luma PS filter from filter-neon-i8mm.cpp. */
#if defined(HAVE_NEON_I8MM)
template<int width, int height>
void interp8_horiz_ps_i8mm(const uint8_t *src, intptr_t srcStride,
                            int16_t *dst, intptr_t dstStride,
                            int coeffIdx, int isRowExt);
#endif

/* External declaration: horizontal luma PS filter from filter-neon-dotprod.cpp. */
#if defined(HAVE_NEON_DOTPROD)
template<int width, int height>
void interp8_horiz_ps_dotprod(const uint8_t *src, intptr_t srcStride,
                               int16_t *dst, intptr_t dstStride,
                               int coeffIdx, int isRowExt);
#endif

namespace {

/*
 * Non-template SVE filter helpers — one per coeffIdx.
 * Non-template avoids GCC two-phase name-lookup restriction for SVE intrinsics.
 */

/* coeffIdx == 1: { -1, 4, -10, 58, 17, -5, 1, 0 } */
static inline svint32_t compute_coeff1(svbool_t pg,
    svint32_t r0, svint32_t r1, svint32_t r2, svint32_t r3,
    svint32_t r4, svint32_t r5, svint32_t r6, svint32_t /*r7*/,
    int32_t offset)
{
    svint32_t diff = svsub_s32_x(pg, r6, r0);
    svint32_t acc  = svadd_s32_x(pg, diff, svdup_n_s32(offset));
    acc = svmla_s32_x(pg, acc, r1, svdup_n_s32( 4));
    acc = svmls_s32_x(pg, acc, r2, svdup_n_s32(10));
    acc = svmla_s32_x(pg, acc, r3, svdup_n_s32(58));
    acc = svmla_s32_x(pg, acc, r4, svdup_n_s32(17));
    acc = svmls_s32_x(pg, acc, r5, svdup_n_s32( 5));
    return acc;
}

/* coeffIdx == 2: { -1, 4, -11, 40, 40, -11, 4, -1 } (symmetric) */
static inline svint32_t compute_coeff2(svbool_t pg,
    svint32_t r0, svint32_t r1, svint32_t r2, svint32_t r3,
    svint32_t r4, svint32_t r5, svint32_t r6, svint32_t r7,
    int32_t offset)
{
    svint32_t t0 = svadd_s32_x(pg, r3, r4);
    svint32_t t1 = svadd_s32_x(pg, r2, r5);
    svint32_t t2 = svadd_s32_x(pg, r1, r6);
    svint32_t t3 = svadd_s32_x(pg, r0, r7);
    svint32_t acc = svdup_n_s32(offset);
    acc = svmla_s32_x(pg, acc, t0, svdup_n_s32(40));
    acc = svmls_s32_x(pg, acc, t1, svdup_n_s32(11));
    acc = svmla_s32_x(pg, acc, t2, svdup_n_s32( 4));
    acc = svmls_s32_x(pg, acc, t3, svdup_n_s32( 1));
    return acc;
}

/* coeffIdx == 3: { 0, 1, -5, 17, 58, -10, 4, -1 } (mirror of coeff1) */
static inline svint32_t compute_coeff3(svbool_t pg,
    svint32_t r0, svint32_t r1, svint32_t r2, svint32_t r3,
    svint32_t r4, svint32_t r5, svint32_t r6, svint32_t r7,
    int32_t offset)
{
    svint32_t diff = svsub_s32_x(pg, r1, r7);
    svint32_t acc  = svadd_s32_x(pg, diff, svdup_n_s32(offset));
    acc = svmla_s32_x(pg, acc, r6, svdup_n_s32( 4));
    acc = svmls_s32_x(pg, acc, r5, svdup_n_s32(10));
    acc = svmla_s32_x(pg, acc, r4, svdup_n_s32(58));
    acc = svmla_s32_x(pg, acc, r3, svdup_n_s32(17));
    acc = svmls_s32_x(pg, acc, r2, svdup_n_s32( 5));
    return acc;
}

/* Arithmetic right-shift then saturate to [0, 255]. */
static inline svint32_t saturate_sve(svbool_t pg, svint32_t acc, int shift)
{
    svuint32_t sh = svdup_n_u32((uint32_t)shift);
    acc = svasr_s32_x(pg, acc, sh);
    acc = svmax_s32_x(pg, acc, svdup_n_s32(  0));
    acc = svmin_s32_x(pg, acc, svdup_n_s32(255));
    return acc;
}

/* Store and advance dst pointer by dstStride. */
static inline void store_row(svbool_t pg, uint8_t *&d, intptr_t dstStride,
                              svint32_t acc, int shift)
{
    svst1b_s32(pg, reinterpret_cast<int8_t *>(d),
               saturate_sve(pg, acc, shift));
    d += dstStride;
}

/*
 * Macro to call the right compute helper at compile time.
 * Avoids a runtime switch inside the hot loop.
 */
#define COMPUTE(ci, pg, r0,r1,r2,r3,r4,r5,r6,r7, off) \
    ((ci) == 1 ? compute_coeff1(pg,r0,r1,r2,r3,r4,r5,r6,r7,off) : \
     (ci) == 2 ? compute_coeff2(pg,r0,r1,r2,r3,r4,r5,r6,r7,off) : \
                 compute_coeff3(pg,r0,r1,r2,r3,r4,r5,r6,r7,off))

/*
 * SVE vertical luma filter: int16 input -> uint8 output.
 *
 * Inner loop unrolled to 8 rows per iteration to expose 8 independent
 * accumulator chains to the CPU's out-of-order engine, hiding the MLA
 * dependency-chain latency (~28 cycles per row).  A 4-row tail handles
 * heights that are not multiples of 8 (e.g. 16×4, 16×12).
 */
template<int coeffIdx, int width, int height>
static void interp8_vert_sp_sve(const int16_t *src, intptr_t srcStride,
                                 uint8_t *dst, intptr_t dstStride)
{
    const int headRoom = IF_INTERNAL_PREC - X265_DEPTH;
    const int shift    = IF_FILTER_PREC + headRoom;
    const int32_t offset = (int32_t)((1u << (shift - 1))
                           + ((uint32_t)IF_INTERNAL_OFFS << IF_FILTER_PREC));

    src -= (8 / 2 - 1) * srcStride;

    for (int col = 0; col < width; col += (int)svcntw())
    {
        svbool_t pg = svwhilelt_b32(col, width);

        const int16_t *s = src;
        uint8_t       *d = dst + col;

        /* Preamble: load 7 rows into the sliding window. */
        svint32_t w0 = svld1sh_s32(pg, s + col); s += srcStride;
        svint32_t w1 = svld1sh_s32(pg, s + col); s += srcStride;
        svint32_t w2 = svld1sh_s32(pg, s + col); s += srcStride;
        svint32_t w3 = svld1sh_s32(pg, s + col); s += srcStride;
        svint32_t w4 = svld1sh_s32(pg, s + col); s += srcStride;
        svint32_t w5 = svld1sh_s32(pg, s + col); s += srcStride;
        svint32_t w6 = svld1sh_s32(pg, s + col); s += srcStride;

        /*
         * 8-row primary loop.
         * Loads 8 new rows (w7..w14), computes 8 independent output rows,
         * then rotates the window by 8.
         */
        int row = 0;
        for (; row + 7 < height; row += 8)
        {
            svint32_t w7  = svld1sh_s32(pg, s + col);
            svint32_t w8  = svld1sh_s32(pg, s + 1 * srcStride + col);
            svint32_t w9  = svld1sh_s32(pg, s + 2 * srcStride + col);
            svint32_t w10 = svld1sh_s32(pg, s + 3 * srcStride + col);
            svint32_t w11 = svld1sh_s32(pg, s + 4 * srcStride + col);
            svint32_t w12 = svld1sh_s32(pg, s + 5 * srcStride + col);
            svint32_t w13 = svld1sh_s32(pg, s + 6 * srcStride + col);
            svint32_t w14 = svld1sh_s32(pg, s + 7 * srcStride + col);
            s += 8 * srcStride;

            svint32_t acc0 = COMPUTE(coeffIdx, pg, w0, w1, w2, w3, w4, w5, w6, w7,   offset);
            svint32_t acc1 = COMPUTE(coeffIdx, pg, w1, w2, w3, w4, w5, w6, w7, w8,   offset);
            svint32_t acc2 = COMPUTE(coeffIdx, pg, w2, w3, w4, w5, w6, w7, w8, w9,   offset);
            svint32_t acc3 = COMPUTE(coeffIdx, pg, w3, w4, w5, w6, w7, w8, w9, w10,  offset);
            svint32_t acc4 = COMPUTE(coeffIdx, pg, w4, w5, w6, w7, w8, w9, w10,w11,  offset);
            svint32_t acc5 = COMPUTE(coeffIdx, pg, w5, w6, w7, w8, w9, w10,w11,w12,  offset);
            svint32_t acc6 = COMPUTE(coeffIdx, pg, w6, w7, w8, w9, w10,w11,w12,w13,  offset);
            svint32_t acc7 = COMPUTE(coeffIdx, pg, w7, w8, w9, w10,w11,w12,w13,w14,  offset);

            store_row(pg, d, dstStride, acc0, shift);
            store_row(pg, d, dstStride, acc1, shift);
            store_row(pg, d, dstStride, acc2, shift);
            store_row(pg, d, dstStride, acc3, shift);
            store_row(pg, d, dstStride, acc4, shift);
            store_row(pg, d, dstStride, acc5, shift);
            store_row(pg, d, dstStride, acc6, shift);
            store_row(pg, d, dstStride, acc7, shift);

            /* Rotate window by 8. */
            w0 = w8;  w1 = w9;  w2 = w10; w3 = w11;
            w4 = w12; w5 = w13; w6 = w14;
        }

        /*
         * 4-row tail (handles heights not divisible by 8, e.g. 4 and 12).
         * For heights that are multiples of 8 this loop body never executes.
         */
        for (; row < height; row += 4)
        {
            svint32_t w7  = svld1sh_s32(pg, s + col);
            svint32_t w8  = svld1sh_s32(pg, s + 1 * srcStride + col);
            svint32_t w9  = svld1sh_s32(pg, s + 2 * srcStride + col);
            svint32_t w10 = svld1sh_s32(pg, s + 3 * srcStride + col);
            s += 4 * srcStride;

            svint32_t acc0 = COMPUTE(coeffIdx, pg, w0, w1, w2, w3, w4, w5, w6, w7,  offset);
            svint32_t acc1 = COMPUTE(coeffIdx, pg, w1, w2, w3, w4, w5, w6, w7, w8,  offset);
            svint32_t acc2 = COMPUTE(coeffIdx, pg, w2, w3, w4, w5, w6, w7, w8, w9,  offset);
            svint32_t acc3 = COMPUTE(coeffIdx, pg, w3, w4, w5, w6, w7, w8, w9, w10, offset);

            store_row(pg, d, dstStride, acc0, shift);
            store_row(pg, d, dstStride, acc1, shift);
            store_row(pg, d, dstStride, acc2, shift);
            store_row(pg, d, dstStride, acc3, shift);

            w0 = w4; w1 = w5; w2 = w6; w3 = w7;
            w4 = w8; w5 = w9; w6 = w10;
        }
    }
}

#undef COMPUTE

/* Dispatcher: maps run-time coeffIdx -> compile-time template parameter. */
template<int width, int height>
static void interp_vert_sp_sve(const int16_t *src, intptr_t srcStride,
                                uint8_t *dst, intptr_t dstStride, int coeffIdx)
{
    switch (coeffIdx)
    {
    case 1:
        interp8_vert_sp_sve<1, width, height>(src, srcStride, dst, dstStride);
        break;
    case 2:
        interp8_vert_sp_sve<2, width, height>(src, srcStride, dst, dstStride);
        break;
    case 3:
        interp8_vert_sp_sve<3, width, height>(src, srcStride, dst, dstStride);
        break;
    default:
        break;
    }
}

/*
 * Combined HV luma filter: i8mm/dotprod horizontal + SVE vertical.
 */
#if defined(HAVE_NEON_I8MM) || defined(HAVE_NEON_DOTPROD)
template<int width, int height>
static void interp_hv_pp_sve(const pixel *src, intptr_t srcStride,
                              pixel *dst, intptr_t dstStride,
                              int idxX, int idxY)
{
    const int N_TAPS = 8;
    ALIGN_VAR_32(int16_t, immed[width * (height + N_TAPS - 1)]);

#if defined(HAVE_NEON_I8MM)
    interp8_horiz_ps_i8mm<width, height>(src, srcStride, immed, width,
                                          idxX, 1);
#else
    interp8_horiz_ps_dotprod<width, height>(src, srcStride, immed, width,
                                             idxX, 1);
#endif

    interp_vert_sp_sve<width, height>(immed + (N_TAPS / 2 - 1) * width,
                                      width, dst, dstStride, idxY);
}
#endif /* HAVE_NEON_I8MM || HAVE_NEON_DOTPROD */

} /* anonymous namespace */

void setupFilterPrimitives_sve(EncoderPrimitives &p)
{
#if defined(HAVE_NEON_I8MM) || defined(HAVE_NEON_DOTPROD)
#define LUMA_SVE(W, H) \
    p.pu[LUMA_ ## W ## x ## H].luma_vsp  = interp_vert_sp_sve<W, H>; \
    p.pu[LUMA_ ## W ## x ## H].luma_hvpp = interp_hv_pp_sve<W, H>

    LUMA_SVE(32, 32);
    LUMA_SVE(64, 64);

    LUMA_SVE(32,  8);
    LUMA_SVE(32, 16);
    LUMA_SVE(32, 24);
    LUMA_SVE(32, 64);
    LUMA_SVE(64, 16);
    LUMA_SVE(64, 32);
    LUMA_SVE(64, 48);

    LUMA_SVE(16,  4);
    LUMA_SVE(16,  8);
    LUMA_SVE(16, 12);
    LUMA_SVE(16, 16);
    LUMA_SVE(16, 32);
    LUMA_SVE(16, 64);
    LUMA_SVE(24, 32);
    LUMA_SVE(48, 64);

#undef LUMA_SVE
#endif /* HAVE_NEON_I8MM || HAVE_NEON_DOTPROD */
}

} /* namespace X265_NS */

#endif /* !HIGH_BIT_DEPTH */
#endif /* HAVE_SVE && HAVE_SVE_BRIDGE */
