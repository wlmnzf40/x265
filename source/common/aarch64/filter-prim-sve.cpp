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
 * The horizontal pass is already well-optimised by the i8mm code path (using
 * matrix-multiply accumulate).  The bottleneck in the combined HV filter is
 * the vertical (SP - short-to-pixel) pass, which works on 14-bit int16_t
 * intermediate values.
 *
 * NEON limitation: each vmlal_n_s16 / vaddl_s16 instruction processes only
 * 4 int32 lanes.  To handle an 8-column group the code must maintain separate
 * "lo" and "hi" int32x4 accumulators and split every int16x8 load with
 * vget_low/vget_high.  This doubles the multiply-accumulate instruction count.
 *
 * SVE solution: svld1sh_s32 loads int16 values from memory and sign-extends
 * them directly to int32 inside the vector register.  At VL=128 this is
 * identical to NEON; at VL=256 (AWS Graviton3 / Neoverse V1) it loads 8 int32
 * per instruction; at VL=512 it loads 16.  Combined with svmla_s32_x (no
 * separate lo/hi split needed) the inner loop uses roughly half the
 * multiply-accumulate instructions that NEON requires for the same output
 * count.  The final saturation and pack to uint8 is done with svmax, svmin
 * and svst1b_s32 (store lowest byte of each active int32 lane) - all SVE1
 * operations, no SVE2 required.
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
 *
 * Using non-template functions avoids the GCC two-phase name-lookup
 * restriction: SVE intrinsic names must be visible at template-definition
 * time when they appear in a dependent context.  By moving every SVE call
 * out of the template the issue disappears.
 *
 * Parameters r0..r7: sign-extended int32 vectors for the 8 filter taps.
 * offset: rounding bias = (1 << (shift-1)) + IF_INTERNAL_OFFS * 64.
 */

/* coeffIdx == 1: { -1, 4, -10, 58, 17, -5, 1, 0 } */
static inline svint32_t compute_coeff1(svbool_t pg,
    svint32_t r0, svint32_t r1, svint32_t r2, svint32_t r3,
    svint32_t r4, svint32_t r5, svint32_t r6, svint32_t /*r7*/,
    int32_t offset)
{
    /* acc = offset + (r6 - r0) + 4*r1 - 10*r2 + 58*r3 + 17*r4 - 5*r5 */
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
    /* acc = offset + (r1 - r7) + 4*r6 - 10*r5 + 58*r4 + 17*r3 - 5*r2 */
    svint32_t diff = svsub_s32_x(pg, r1, r7);
    svint32_t acc  = svadd_s32_x(pg, diff, svdup_n_s32(offset));
    acc = svmla_s32_x(pg, acc, r6, svdup_n_s32( 4));
    acc = svmls_s32_x(pg, acc, r5, svdup_n_s32(10));
    acc = svmla_s32_x(pg, acc, r4, svdup_n_s32(58));
    acc = svmla_s32_x(pg, acc, r3, svdup_n_s32(17));
    acc = svmls_s32_x(pg, acc, r2, svdup_n_s32( 5));
    return acc;
}

/* Arithmetic right-shift then unsigned saturate to [0, 255]. */
static inline svint32_t saturate_sve(svbool_t pg, svint32_t acc, int shift)
{
    svuint32_t sh = svdup_n_u32((uint32_t)shift);
    acc = svasr_s32_x(pg, acc, sh);
    acc = svmax_s32_x(pg, acc, svdup_n_s32(  0));
    acc = svmin_s32_x(pg, acc, svdup_n_s32(255));
    return acc;
}

/*
 * SVE vertical luma filter: int16 input (intermediate) -> uint8 output.
 *
 * Column loop advances by svcntw() per iteration so the same binary handles
 * VL=128 (4 int32/iter), VL=256 (8/iter), VL=512 (16/iter).
 *
 * svld1sh_s32 reads 16-bit values and sign-extends to 32-bit, eliminating the
 * NEON lo/hi split.  svst1b_s32 packs the lowest byte of every active int32
 * lane into consecutive memory bytes.
 *
 * SVE types (svint32_t) have unknown size, so pointer arithmetic on them is
 * illegal.  The sliding window uses 11 named scalar variables (w0..w10) and
 * explicit assignment to rotate them.
 */
template<int coeffIdx, int width, int height>
static void interp8_vert_sp_sve(const int16_t *src, intptr_t srcStride,
                                 uint8_t *dst, intptr_t dstStride)
{
    const int headRoom = IF_INTERNAL_PREC - X265_DEPTH;
    const int shift    = IF_FILTER_PREC + headRoom;
    const int32_t offset = (int32_t)((1u << (shift - 1))
                           + ((uint32_t)IF_INTERNAL_OFFS << IF_FILTER_PREC));

    /* Point src at first tap row (3 rows before first output row). */
    src -= (8 / 2 - 1) * srcStride;

    for (int col = 0; col < width; col += (int)svcntw())
    {
        svbool_t pg = svwhilelt_b32(col, width);

        const int16_t *s = src;
        uint8_t       *d = dst;

        /* Load sliding-window preamble: rows 0..6 (7 rows). */
        svint32_t w0 = svld1sh_s32(pg, s + col); s += srcStride;
        svint32_t w1 = svld1sh_s32(pg, s + col); s += srcStride;
        svint32_t w2 = svld1sh_s32(pg, s + col); s += srcStride;
        svint32_t w3 = svld1sh_s32(pg, s + col); s += srcStride;
        svint32_t w4 = svld1sh_s32(pg, s + col); s += srcStride;
        svint32_t w5 = svld1sh_s32(pg, s + col); s += srcStride;
        svint32_t w6 = svld1sh_s32(pg, s + col); s += srcStride;

        for (int row = 0; row < height; row += 4)
        {
            /* Append 4 new rows to the sliding window. */
            svint32_t w7  = svld1sh_s32(pg, s + col);
            svint32_t w8  = svld1sh_s32(pg, s + 1 * srcStride + col);
            svint32_t w9  = svld1sh_s32(pg, s + 2 * srcStride + col);
            svint32_t w10 = svld1sh_s32(pg, s + 3 * srcStride + col);
            s += 4 * srcStride;

            /* Compute and store 4 output rows. */
            svint32_t acc0, acc1, acc2, acc3;
            if (coeffIdx == 1)
            {
                acc0 = compute_coeff1(pg, w0, w1, w2, w3, w4, w5, w6, w7, offset);
                acc1 = compute_coeff1(pg, w1, w2, w3, w4, w5, w6, w7, w8, offset);
                acc2 = compute_coeff1(pg, w2, w3, w4, w5, w6, w7, w8, w9, offset);
                acc3 = compute_coeff1(pg, w3, w4, w5, w6, w7, w8, w9, w10, offset);
            }
            else if (coeffIdx == 2)
            {
                acc0 = compute_coeff2(pg, w0, w1, w2, w3, w4, w5, w6, w7, offset);
                acc1 = compute_coeff2(pg, w1, w2, w3, w4, w5, w6, w7, w8, offset);
                acc2 = compute_coeff2(pg, w2, w3, w4, w5, w6, w7, w8, w9, offset);
                acc3 = compute_coeff2(pg, w3, w4, w5, w6, w7, w8, w9, w10, offset);
            }
            else /* coeffIdx == 3 */
            {
                acc0 = compute_coeff3(pg, w0, w1, w2, w3, w4, w5, w6, w7, offset);
                acc1 = compute_coeff3(pg, w1, w2, w3, w4, w5, w6, w7, w8, offset);
                acc2 = compute_coeff3(pg, w2, w3, w4, w5, w6, w7, w8, w9, offset);
                acc3 = compute_coeff3(pg, w3, w4, w5, w6, w7, w8, w9, w10, offset);
            }

            svst1b_s32(pg, reinterpret_cast<int8_t *>(d + col),
                       saturate_sve(pg, acc0, shift));
            d += dstStride;
            svst1b_s32(pg, reinterpret_cast<int8_t *>(d + col),
                       saturate_sve(pg, acc1, shift));
            d += dstStride;
            svst1b_s32(pg, reinterpret_cast<int8_t *>(d + col),
                       saturate_sve(pg, acc2, shift));
            d += dstStride;
            svst1b_s32(pg, reinterpret_cast<int8_t *>(d + col),
                       saturate_sve(pg, acc3, shift));
            d += dstStride;

            /* Slide the window: rotate w0..w6 = w4..w10. */
            w0 = w4; w1 = w5; w2 = w6; w3 = w7;
            w4 = w8; w5 = w9; w6 = w10;
        }
    }
}

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
 * Combined horizontal+vertical luma HV filter (luma_hvpp).
 *
 * Horizontal pass: i8mm matrix-multiply accumulate (best available).
 * Vertical   pass: SVE scalable vertical filter (this file).
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

    /* Large square blocks. */
    LUMA_SVE(32, 32);
    LUMA_SVE(64, 64);

    /* Rectangular variants involving a 32- or 64-wide dimension. */
    LUMA_SVE(32,  8);
    LUMA_SVE(32, 16);
    LUMA_SVE(32, 24);
    LUMA_SVE(32, 64);
    LUMA_SVE(64, 16);
    LUMA_SVE(64, 32);
    LUMA_SVE(64, 48);

    /* Smaller blocks that also benefit from the SVE vertical filter. */
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
