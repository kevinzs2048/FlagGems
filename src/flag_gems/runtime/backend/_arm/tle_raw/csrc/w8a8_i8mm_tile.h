// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <arm_neon.h>
#include <cstdint>
#include <cstring>

static inline uint16x4_t w8a8_i8mm_bf16(float32x4_t value) {
    const auto bits=vreinterpretq_u32_f32(value);
    const auto rounding=vaddq_u32(vdupq_n_u32(0x7fff),vandq_u32(vshrq_n_u32(bits,16),vdupq_n_u32(1)));
    const auto rounded=vshrq_n_u32(vaddq_u32(bits,rounding),16);
    const auto nan=vcgtq_u32(vandq_u32(bits,vdupq_n_u32(0x7fffffff)),vdupq_n_u32(0x7f800000));
    return vmovn_u32(vbslq_u32(nan,vorrq_u32(vshrq_n_u32(bits,16),vdupq_n_u32(0x40)),rounded));
}


template<bool ACTIVATE>
static inline void w8a8_i8mm_store(uint16_t* out,uint16x8_t bits,const uint16_t* lut){
    if constexpr(ACTIVATE){
        uint16_t values[8];vst1q_u16(values,bits);
#pragma GCC unroll 8
        for(int i=0;i<8;++i)out[i]=lut[values[i]];
    }else vst1q_u16(out,bits);
}

// Partial M tiles keep the same row-pair layout, while avoiding padded
// row computations and the scalar N-tail store for complete eight-column tiles.
template<int ROW_PAIRS,bool ACTIVATE=false,int A_ROWS=8>
static inline void w8a8_i8mm_row_tail(const int8_t* a,const int8_t* w,
        const float* xs,const float* ws,const uint16_t* bias,uint16_t* y,
        int64_t row,int64_t col,int64_t m,int64_t n,int64_t k,const uint16_t* lut=nullptr){
    int32x4_t dot[ROW_PAIRS*4];
    for(int i=0;i<ROW_PAIRS*4;++i)dot[i]=vdupq_n_s32(0);
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC unroll 4
#endif
    for(int64_t g=0;g<k/8;++g){
        const auto b0=vld1q_s8(w+g*64),b1=vld1q_s8(w+g*64+16);
        const auto b2=vld1q_s8(w+g*64+32),b3=vld1q_s8(w+g*64+48);
#pragma GCC unroll 4
        for(int r=0;r<ROW_PAIRS;++r){
            const auto av=vld1q_s8(a+(r/(A_ROWS/2))*k*A_ROWS+g*(A_ROWS*8)+(r%(A_ROWS/2))*16);
            dot[r*4]=vmmlaq_s32(dot[r*4],av,b0);dot[r*4+1]=vmmlaq_s32(dot[r*4+1],av,b1);
            dot[r*4+2]=vmmlaq_s32(dot[r*4+2],av,b2);dot[r*4+3]=vmmlaq_s32(dot[r*4+3],av,b3);
        }
    }
    const auto w0=vld1q_f32(ws+col),w1=vld1q_f32(ws+col+4);
    auto b0=vdupq_n_f32(0),b1=vdupq_n_f32(0);
    if(bias){const auto b=vld1q_u16(bias+col);b0=vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(b),16));b1=vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(b),16));}
#pragma GCC unroll 4
    for(int r=0;r<ROW_PAIRS;++r){
        const auto a0=vcvtq_f32_s32(dot[r*4]),a1=vcvtq_f32_s32(dot[r*4+1]);
        const auto a2=vcvtq_f32_s32(dot[r*4+2]),a3=vcvtq_f32_s32(dot[r*4+3]);
#pragma GCC unroll 2
        for(int rr=0;rr<2;++rr)if(row+r*2+rr<m){
            const auto low=rr?vcombine_f32(vget_high_f32(a0),vget_high_f32(a1)):vcombine_f32(vget_low_f32(a0),vget_low_f32(a1));
            const auto high=rr?vcombine_f32(vget_high_f32(a2),vget_high_f32(a3)):vcombine_f32(vget_low_f32(a2),vget_low_f32(a3));
            const auto scale=vdupq_n_f32(xs[row+r*2+rr]);
            auto y0=vmulq_f32(vmulq_f32(low,scale),w0),y1=vmulq_f32(vmulq_f32(high,scale),w1);
            if(bias){y0=vaddq_f32(y0,b0);y1=vaddq_f32(y1,b1);}
            w8a8_i8mm_store<ACTIVATE>(y+(row+r*2+rr)*n+col,vcombine_u16(w8a8_i8mm_bf16(y0),w8a8_i8mm_bf16(y1)),lut);
        }
    }
}

// A: eight-row panels with A_ROWS-row contiguous K streams (default8).
// W: [N/8,K/8,8 columns,8 K]. Existing SME callers keep A_ROWS=8.
// The caller owns layouts and assigns disjoint output tiles.
template<bool ACTIVATE=false,int A_ROWS=8,int PREFETCH=0>
static inline void w8a8_i8mm_tile8(const int8_t* a,const int8_t* w,
        const float* xs,const float* ws,const uint16_t* bias,uint16_t* y,
        int64_t row,int64_t col,int64_t m,int64_t n,int64_t k,const uint16_t* lut=nullptr) {
    if(row+8>m && col+8<=n){
        const auto left=m-row;
        if(left<=2)w8a8_i8mm_row_tail<1,ACTIVATE,A_ROWS>(a,w,xs,ws,bias,y,row,col,m,n,k,lut);
        else if(left<=4)w8a8_i8mm_row_tail<2,ACTIVATE,A_ROWS>(a,w,xs,ws,bias,y,row,col,m,n,k,lut);
        else if(left<=6)w8a8_i8mm_row_tail<3,ACTIVATE,A_ROWS>(a,w,xs,ws,bias,y,row,col,m,n,k,lut);
        else w8a8_i8mm_row_tail<4,ACTIVATE,A_ROWS>(a,w,xs,ws,bias,y,row,col,m,n,k,lut);
        return;
    }
    int32x4_t dot[16];
    for (int i=0;i<16;++i) dot[i]=vdupq_n_s32(0);
#if defined(__GNUC__) && !defined(__clang__)
#if FLAGGEMS_I8MM_K_UNROLL == 2
#pragma GCC unroll 2
#elif FLAGGEMS_I8MM_K_UNROLL == 4
#pragma GCC unroll 4
#endif
#endif
    for (int64_t g=0;g<k/8;++g) {
        if constexpr(PREFETCH>0) {
            const int64_t ahead=g+PREFETCH<k/8 ? g+PREFETCH : k/8-1;
            __builtin_prefetch(w+ahead*64,0,3);
        }
        const auto b0=vld1q_s8(w+g*64);
        const auto b1=vld1q_s8(w+g*64+16);
        const auto b2=vld1q_s8(w+g*64+32);
        const auto b3=vld1q_s8(w+g*64+48);
#pragma unroll
        for (int r=0;r<4;++r) {
            const auto av=vld1q_s8(a+(r/(A_ROWS/2))*k*A_ROWS+g*(A_ROWS*8)+(r%(A_ROWS/2))*16);
            dot[r*4]=vmmlaq_s32(dot[r*4],av,b0);
            dot[r*4+1]=vmmlaq_s32(dot[r*4+1],av,b1);
            dot[r*4+2]=vmmlaq_s32(dot[r*4+2],av,b2);
            dot[r*4+3]=vmmlaq_s32(dot[r*4+3],av,b3);
        }
    }
    if (row+8<=m && col+8<=n) {
        const auto w0=vld1q_f32(ws+col),w1=vld1q_f32(ws+col+4);
        auto b0=vdupq_n_f32(0),b1=vdupq_n_f32(0);
        if (bias) {
            const auto b=vld1q_u16(bias+col);
            b0=vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(b),16));
            b1=vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(b),16));
        }
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC unroll 4
#else
#pragma unroll
#endif
        for (int r=0;r<4;++r) {
            const auto a0=vcvtq_f32_s32(dot[r*4]),a1=vcvtq_f32_s32(dot[r*4+1]);
            const auto a2=vcvtq_f32_s32(dot[r*4+2]),a3=vcvtq_f32_s32(dot[r*4+3]);
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC unroll 2
#else
#pragma unroll
#endif
            for (int rr=0;rr<2;++rr) {
                const auto low=rr ? vcombine_f32(vget_high_f32(a0),vget_high_f32(a1))
                                  : vcombine_f32(vget_low_f32(a0),vget_low_f32(a1));
                const auto high=rr ? vcombine_f32(vget_high_f32(a2),vget_high_f32(a3))
                                   : vcombine_f32(vget_low_f32(a2),vget_low_f32(a3));
                const auto scale=vdupq_n_f32(xs[row+r*2+rr]);
                auto y0=vmulq_f32(vmulq_f32(low,scale),w0);
                auto y1=vmulq_f32(vmulq_f32(high,scale),w1);
                if (bias) {y0=vaddq_f32(y0,b0);y1=vaddq_f32(y1,b1);}
                w8a8_i8mm_store<ACTIVATE>(y+(row+r*2+rr)*n+col,vcombine_u16(w8a8_i8mm_bf16(y0),w8a8_i8mm_bf16(y1)),lut);
            }
        }
        return;
    }
    for (int r=0;r<4;++r) for (int c=0;c<4;++c) {
        float values[4];vst1q_f32(values,vcvtq_f32_s32(dot[r*4+c]));
        for (int rr=0;rr<2;++rr) for (int cc=0;cc<2;++cc) {
            const int64_t i=row+r*2+rr,j=col+c*2+cc;
            if (i<m && j<n) {
                float value=(values[rr*2+cc]*xs[i])*ws[j];
                if (bias) {uint32_t b=uint32_t(bias[j])<<16;float bf;std::memcpy(&bf,&b,4);value+=bf;}
                uint32_t bits;std::memcpy(&bits,&value,4);
                const auto rounded=(bits&0x7fffffffu)>0x7f800000u ? uint16_t((bits>>16)|0x40u)
                    : uint16_t((bits+0x7fffu+((bits>>16)&1u))>>16);
                y[i*n+j]=ACTIVATE?lut[rounded]:rounded;
            }
        }
    }
}
