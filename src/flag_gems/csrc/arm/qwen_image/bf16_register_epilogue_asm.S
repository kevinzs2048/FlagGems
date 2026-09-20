//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
// Local experimental derivative: output epilogue only. Full-K arithmetic retained.
// Source: KleidiAI v1.29.0, original filename recorded in generator review/report.
//

#if defined(_MSC_VER)
    #define KAI_ASM_GLOBAL(name) GLOBAL name
    #define KAI_ASM_FUNCTION_TYPE(name)
    #define KAI_ASM_FUNCTION_LABEL(name) name PROC
    #define KAI_ASM_FUNCTION_END(name) ENDP

    #define KAI_ASM_CODE(name) AREA name, CODE, READONLY
    #define KAI_ASM_ALIGN
    #define KAI_ASM_LABEL(name) name
    #define KAI_ASM_INST(hex) DCD hex
    #define KAI_ASM_END END
#else
    #if defined(__APPLE__)
        #define KAI_ASM_GLOBAL(name) .globl _##name
        #define KAI_ASM_FUNCTION_TYPE(name)
        #define KAI_ASM_FUNCTION_LABEL(name) _##name:
        #define KAI_ASM_FUNCTION_END(name)
    #else
        #define KAI_ASM_GLOBAL(name) .global name
        #define KAI_ASM_FUNCTION_TYPE(name) .type name, %function
        #define KAI_ASM_FUNCTION_LABEL(name) name:
        #define KAI_ASM_FUNCTION_END(name) .size name, .-name
    #endif

    #define KAI_ASM_CODE(name) .text
    #define KAI_ASM_ALIGN .p2align 4,,11
    #define KAI_ASM_LABEL(name) name:
    #define KAI_ASM_INST(hex) .inst hex
    #define KAI_ASM_END
#endif

    KAI_ASM_CODE(matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(qwen21_bf16_register_epilogue_kernel)

KAI_ASM_FUNCTION_TYPE(qwen21_bf16_register_epilogue_kernel)
KAI_ASM_FUNCTION_LABEL(qwen21_bf16_register_epilogue_kernel)
    stp x20, x21, [sp, -144]!
    stp x22, x23, [sp, 16]
    stp x24, x25, [sp, 32]
    stp x26, x27, [sp, 48]
    str x28, [sp, 64]
    stp d8, d9, [sp, 72]
    stp d10, d11, [sp, 88]
    stp d12, d13, [sp, 104]
    stp d14, d15, [sp, 120]
    KAI_ASM_INST(0xd503477f)  // SMSTART ZA
    mov x15, #0x0
    ldr x14, [x0, #0x30]
    ptrue p0.b
    KAI_ASM_INST(0x25207811)  // ptrue pn9.b
    ldr w13, [x0, #0x20]
    mov x11, #0x0
    ldr w10, [x0, #0x28]
    add x14, x14, #0x1
    ldr x9, [x0, #0x0]
    lsr x14, x14, #0x1
KAI_ASM_LABEL(label_1)  // M loop
    ldr x28, [x0, #0x8]
KAI_ASM_LABEL(label_2)  // N loop
    fmov z22.s, #1.0
    KAI_ASM_INST(0xa040478c)  // ld1w { z12.s-z13.s }, pn9.b/Z, [x28]
    KAI_ASM_INST(0xc00800ff)  // zero { zad0, zad1, zad2, zad3, zad4, zad5, zad6, zad7 }
    mov x27, x9
    KAI_ASM_INST(0x25aa4570)  // whilelt pn8.s, x11, x10, VLx2
    addvl x28, x28, #2
    KAI_ASM_INST(0x808c02c0)  // fmopa za0.s, p0/M, p0/M, z22.s, z12.s
    KAI_ASM_INST(0x808d02c1)  // fmopa za1.s, p0/M, p0/M, z22.s, z13.s
    KAI_ASM_INST(0x808c02c2)  // fmopa za2.s, p0/M, p0/M, z22.s, z12.s
    KAI_ASM_INST(0x808d02c3)  // fmopa za3.s, p0/M, p0/M, z22.s, z13.s
    lsr x21, x14, #0x2
    and x20, x14, #0x3
    cbz x21, label_6
    subs x21, x21, #0x1
    KAI_ASM_INST(0xa040a764)  // ld1h { z4.h-z7.h }, pn9.b/Z, [x27]
    KAI_ASM_INST(0xa041a768)  // ld1h { z8.h-z11.h }, pn9.b/Z, [x27, #0x4, MUL VL]
    addvl x27, x27, #8
    KAI_ASM_INST(0xa040a794)  // ld1h { z20.h-z23.h }, pn9.b/Z, [x28]
    KAI_ASM_INST(0xa041a78c)  // ld1h { z12.h-z15.h }, pn9.b/Z, [x28, #0x4, MUL VL]
    addvl x28, x28, #8
    ble label_5
KAI_ASM_LABEL(label_4)  // K loop
    KAI_ASM_INST(0x81940080)  // bfmopa za0.s, p0/M, p0/M, z4.h, z20.h
    subs x21, x21, #0x1
    KAI_ASM_INST(0x81950081)  // bfmopa za1.s, p0/M, p0/M, z4.h, z21.h
    KAI_ASM_INST(0x819400a2)  // bfmopa za2.s, p0/M, p0/M, z5.h, z20.h
    KAI_ASM_INST(0x819500a3)  // bfmopa za3.s, p0/M, p0/M, z5.h, z21.h
    KAI_ASM_INST(0x819600c0)  // bfmopa za0.s, p0/M, p0/M, z6.h, z22.h
    KAI_ASM_INST(0x819700c1)  // bfmopa za1.s, p0/M, p0/M, z6.h, z23.h
    KAI_ASM_INST(0x819600e2)  // bfmopa za2.s, p0/M, p0/M, z7.h, z22.h
    KAI_ASM_INST(0x819700e3)  // bfmopa za3.s, p0/M, p0/M, z7.h, z23.h
    KAI_ASM_INST(0xa040a764)  // ld1h { z4.h-z7.h }, pn9.b/Z, [x27]
    KAI_ASM_INST(0x818c0100)  // bfmopa za0.s, p0/M, p0/M, z8.h, z12.h
    KAI_ASM_INST(0xa040a794)  // ld1h { z20.h-z23.h }, pn9.b/Z, [x28]
    KAI_ASM_INST(0x818d0101)  // bfmopa za1.s, p0/M, p0/M, z8.h, z13.h
    KAI_ASM_INST(0x818c0122)  // bfmopa za2.s, p0/M, p0/M, z9.h, z12.h
    KAI_ASM_INST(0x818d0123)  // bfmopa za3.s, p0/M, p0/M, z9.h, z13.h
    KAI_ASM_INST(0x818e0140)  // bfmopa za0.s, p0/M, p0/M, z10.h, z14.h
    KAI_ASM_INST(0x818f0141)  // bfmopa za1.s, p0/M, p0/M, z10.h, z15.h
    KAI_ASM_INST(0x818e0162)  // bfmopa za2.s, p0/M, p0/M, z11.h, z14.h
    KAI_ASM_INST(0x818f0163)  // bfmopa za3.s, p0/M, p0/M, z11.h, z15.h
    KAI_ASM_INST(0xa041a768)  // ld1h { z8.h-z11.h }, pn9.b/Z, [x27, #0x4, MUL VL]
    addvl x27, x27, #8
    KAI_ASM_INST(0xa041a78c)  // ld1h { z12.h-z15.h }, pn9.b/Z, [x28, #0x4, MUL VL]
    addvl x28, x28, #8
    bgt label_4
KAI_ASM_LABEL(label_5)  // K loop tail
    KAI_ASM_INST(0x81940080)  // bfmopa za0.s, p0/M, p0/M, z4.h, z20.h
    KAI_ASM_INST(0x81950081)  // bfmopa za1.s, p0/M, p0/M, z4.h, z21.h
    KAI_ASM_INST(0x819400a2)  // bfmopa za2.s, p0/M, p0/M, z5.h, z20.h
    KAI_ASM_INST(0x819500a3)  // bfmopa za3.s, p0/M, p0/M, z5.h, z21.h
    KAI_ASM_INST(0x819600c0)  // bfmopa za0.s, p0/M, p0/M, z6.h, z22.h
    KAI_ASM_INST(0x819700c1)  // bfmopa za1.s, p0/M, p0/M, z6.h, z23.h
    KAI_ASM_INST(0x819600e2)  // bfmopa za2.s, p0/M, p0/M, z7.h, z22.h
    KAI_ASM_INST(0x819700e3)  // bfmopa za3.s, p0/M, p0/M, z7.h, z23.h
    KAI_ASM_INST(0x818c0100)  // bfmopa za0.s, p0/M, p0/M, z8.h, z12.h
    KAI_ASM_INST(0x818d0101)  // bfmopa za1.s, p0/M, p0/M, z8.h, z13.h
    KAI_ASM_INST(0x818c0122)  // bfmopa za2.s, p0/M, p0/M, z9.h, z12.h
    KAI_ASM_INST(0x818d0123)  // bfmopa za3.s, p0/M, p0/M, z9.h, z13.h
    KAI_ASM_INST(0x818e0140)  // bfmopa za0.s, p0/M, p0/M, z10.h, z14.h
    KAI_ASM_INST(0x818f0141)  // bfmopa za1.s, p0/M, p0/M, z10.h, z15.h
    KAI_ASM_INST(0x818e0162)  // bfmopa za2.s, p0/M, p0/M, z11.h, z14.h
    KAI_ASM_INST(0x818f0163)  // bfmopa za3.s, p0/M, p0/M, z11.h, z15.h
KAI_ASM_LABEL(label_6)  // K oddments
    cbz x20, label_8
KAI_ASM_LABEL(label_7)  // K oddments: Loop
    KAI_ASM_INST(0xa040277c)  // ld1h { z28.h-z29.h }, pn9.b/Z, [x27]
    subs x20, x20, #0x1
    addvl x27, x27, #2
    KAI_ASM_INST(0xa1402787)  // ld1h { z7.h, z15.h }, pn9.b/Z, [x28]
    addvl x28, x28, #2
    KAI_ASM_INST(0x81870380)  // bfmopa za0.s, p0/M, p0/M, z28.h, z7.h
    KAI_ASM_INST(0x818f0381)  // bfmopa za1.s, p0/M, p0/M, z28.h, z15.h
    KAI_ASM_INST(0x818703a2)  // bfmopa za2.s, p0/M, p0/M, z29.h, z7.h
    KAI_ASM_INST(0x818f03a3)  // bfmopa za3.s, p0/M, p0/M, z29.h, z15.h
    bgt label_7
KAI_ASM_LABEL(label_8)  // K oddments: End
    // Modified epilogue: integer RNE with canonical NaN, matching ATen BF16.
    mov w8, #0x7fff
    dup z17.s, w8
    mov w8, #0x7fc0
    dup z18.s, w8
    whilelt p1.h, x11, x10
    ldr x26, [x0, #0x10]
    sub x25, x13, x15
    cntw x24
    KAI_ASM_INST(0x854ec01a)  // ld1rw { z26.s }, p0/Z, [x0, #56]
    ldr x23, [x0, #0x18]
    cmp x25, x24
    KAI_ASM_INST(0x854fc018)  // ld1rw { z24.s }, p0/Z, [x0, #60]
    mov x12, #0x0
    csel x22, x25, x24, LT
    add x26, x26, x11, LSL #1  // C += n (BF16 output)
    lsr x21, x22, #0x2
    madd x26, x15, x23, x26  // C += m * ldc
    and x20, x22, #0x3
    cbz x21, label_11
KAI_ASM_LABEL(label_10)  // Store to output array: Accumulator row 0 loop
    KAI_ASM_INST(0xc0860404)  // mova { z4.s-z7.s }, za0h.s[x12]
    KAI_ASM_INST(0xc086042c)  // mova { z12.s-z15.s }, za1h.s[x12]
    KAI_ASM_INST(0xc1b8cb44)  // fclamp { z4.s-z7.s }, z26.s, z24.s
    KAI_ASM_INST(0xc1b8cb4c)  // fclamp { z12.s-z15.s }, z26.s, z24.s
    add x12, x12, #0x4
    cmp x12, x21, LSL #2
    fcmuo p2.s, p0/z, z4.s, z4.s
    lsr z16.s, z4.s, #16
    and z16.s, z16.s, #1
    add z16.s, z16.s, z17.s
    add z4.s, z4.s, z16.s
    lsr z4.s, z4.s, #16
    sel z4.s, p2, z18.s, z4.s
    fcmuo p2.s, p0/z, z12.s, z12.s
    lsr z16.s, z12.s, #16
    and z16.s, z16.s, #1
    add z16.s, z16.s, z17.s
    add z12.s, z12.s, z16.s
    lsr z12.s, z12.s, #16
    sel z12.s, p2, z18.s, z12.s
    uzp1 z19.h, z4.h, z12.h
    st1h { z19.h }, p1, [x26]
    add x26, x26, x23
    fcmuo p2.s, p0/z, z5.s, z5.s
    lsr z16.s, z5.s, #16
    and z16.s, z16.s, #1
    add z16.s, z16.s, z17.s
    add z5.s, z5.s, z16.s
    lsr z5.s, z5.s, #16
    sel z5.s, p2, z18.s, z5.s
    fcmuo p2.s, p0/z, z13.s, z13.s
    lsr z16.s, z13.s, #16
    and z16.s, z16.s, #1
    add z16.s, z16.s, z17.s
    add z13.s, z13.s, z16.s
    lsr z13.s, z13.s, #16
    sel z13.s, p2, z18.s, z13.s
    uzp1 z19.h, z5.h, z13.h
    st1h { z19.h }, p1, [x26]
    add x26, x26, x23
    fcmuo p2.s, p0/z, z6.s, z6.s
    lsr z16.s, z6.s, #16
    and z16.s, z16.s, #1
    add z16.s, z16.s, z17.s
    add z6.s, z6.s, z16.s
    lsr z6.s, z6.s, #16
    sel z6.s, p2, z18.s, z6.s
    fcmuo p2.s, p0/z, z14.s, z14.s
    lsr z16.s, z14.s, #16
    and z16.s, z16.s, #1
    add z16.s, z16.s, z17.s
    add z14.s, z14.s, z16.s
    lsr z14.s, z14.s, #16
    sel z14.s, p2, z18.s, z14.s
    uzp1 z19.h, z6.h, z14.h
    st1h { z19.h }, p1, [x26]
    add x26, x26, x23
    fcmuo p2.s, p0/z, z7.s, z7.s
    lsr z16.s, z7.s, #16
    and z16.s, z16.s, #1
    add z16.s, z16.s, z17.s
    add z7.s, z7.s, z16.s
    lsr z7.s, z7.s, #16
    sel z7.s, p2, z18.s, z7.s
    fcmuo p2.s, p0/z, z15.s, z15.s
    lsr z16.s, z15.s, #16
    and z16.s, z16.s, #1
    add z16.s, z16.s, z17.s
    add z15.s, z15.s, z16.s
    lsr z15.s, z15.s, #16
    sel z15.s, p2, z18.s, z15.s
    uzp1 z19.h, z7.h, z15.h
    st1h { z19.h }, p1, [x26]
    add x26, x26, x23
    blt label_10
KAI_ASM_LABEL(label_11)  // Store to output array: Accumulator row 0 oddments
    cbz x20, label_12
    KAI_ASM_INST(0xc0860400)  // mova { z0.s-z3.s }, za0h.s[x12]
    KAI_ASM_INST(0xc0860428)  // mova { z8.s-z11.s }, za1h.s[x12]
    subs x20, x20, #0x1
    KAI_ASM_INST(0xc1b8cb40)  // fclamp { z0.s-z3.s }, z26.s, z24.s
    KAI_ASM_INST(0xc1b8cb48)  // fclamp { z8.s-z11.s }, z26.s, z24.s
    fcmuo p2.s, p0/z, z0.s, z0.s
    lsr z16.s, z0.s, #16
    and z16.s, z16.s, #1
    add z16.s, z16.s, z17.s
    add z0.s, z0.s, z16.s
    lsr z0.s, z0.s, #16
    sel z0.s, p2, z18.s, z0.s
    fcmuo p2.s, p0/z, z8.s, z8.s
    lsr z16.s, z8.s, #16
    and z16.s, z16.s, #1
    add z16.s, z16.s, z17.s
    add z8.s, z8.s, z16.s
    lsr z8.s, z8.s, #16
    sel z8.s, p2, z18.s, z8.s
    uzp1 z19.h, z0.h, z8.h
    st1h { z19.h }, p1, [x26]
    add x26, x26, x23
    beq label_12
    subs x20, x20, #0x1
    fcmuo p2.s, p0/z, z1.s, z1.s
    lsr z16.s, z1.s, #16
    and z16.s, z16.s, #1
    add z16.s, z16.s, z17.s
    add z1.s, z1.s, z16.s
    lsr z1.s, z1.s, #16
    sel z1.s, p2, z18.s, z1.s
    fcmuo p2.s, p0/z, z9.s, z9.s
    lsr z16.s, z9.s, #16
    and z16.s, z16.s, #1
    add z16.s, z16.s, z17.s
    add z9.s, z9.s, z16.s
    lsr z9.s, z9.s, #16
    sel z9.s, p2, z18.s, z9.s
    uzp1 z19.h, z1.h, z9.h
    st1h { z19.h }, p1, [x26]
    add x26, x26, x23
    beq label_12
    fcmuo p2.s, p0/z, z2.s, z2.s
    lsr z16.s, z2.s, #16
    and z16.s, z16.s, #1
    add z16.s, z16.s, z17.s
    add z2.s, z2.s, z16.s
    lsr z2.s, z2.s, #16
    sel z2.s, p2, z18.s, z2.s
    fcmuo p2.s, p0/z, z10.s, z10.s
    lsr z16.s, z10.s, #16
    and z16.s, z16.s, #1
    add z16.s, z16.s, z17.s
    add z10.s, z10.s, z16.s
    lsr z10.s, z10.s, #16
    sel z10.s, p2, z18.s, z10.s
    uzp1 z19.h, z2.h, z10.h
    st1h { z19.h }, p1, [x26]
    add x26, x26, x23
KAI_ASM_LABEL(label_12)  // Store to output array: Accumulator row 0 oddments: End
    subs x25, x25, x22
    beq label_16
    cmp x25, x24
    mov x12, #0x0
    csel x20, x25, x24, LT
    lsr x21, x20, #0x2
    and x20, x20, #0x3
    cbz x21, label_14
KAI_ASM_LABEL(label_13)  // Store to output array: Accumulator row 1 loop
    KAI_ASM_INST(0xc0860454)  // mova { z20.s-z23.s }, za2h.s[x12]
    KAI_ASM_INST(0xc086047c)  // mova { z28.s-z31.s }, za3h.s[x12]
    KAI_ASM_INST(0xc1b8cb54)  // fclamp { z20.s-z23.s }, z26.s, z24.s
    KAI_ASM_INST(0xc1b8cb5c)  // fclamp { z28.s-z31.s }, z26.s, z24.s
    add x12, x12, #0x4
    cmp x12, x21, LSL #2
    fcmuo p2.s, p0/z, z20.s, z20.s
    lsr z16.s, z20.s, #16
    and z16.s, z16.s, #1
    add z16.s, z16.s, z17.s
    add z20.s, z20.s, z16.s
    lsr z20.s, z20.s, #16
    sel z20.s, p2, z18.s, z20.s
    fcmuo p2.s, p0/z, z28.s, z28.s
    lsr z16.s, z28.s, #16
    and z16.s, z16.s, #1
    add z16.s, z16.s, z17.s
    add z28.s, z28.s, z16.s
    lsr z28.s, z28.s, #16
    sel z28.s, p2, z18.s, z28.s
    uzp1 z19.h, z20.h, z28.h
    st1h { z19.h }, p1, [x26]
    add x26, x26, x23
    fcmuo p2.s, p0/z, z21.s, z21.s
    lsr z16.s, z21.s, #16
    and z16.s, z16.s, #1
    add z16.s, z16.s, z17.s
    add z21.s, z21.s, z16.s
    lsr z21.s, z21.s, #16
    sel z21.s, p2, z18.s, z21.s
    fcmuo p2.s, p0/z, z29.s, z29.s
    lsr z16.s, z29.s, #16
    and z16.s, z16.s, #1
    add z16.s, z16.s, z17.s
    add z29.s, z29.s, z16.s
    lsr z29.s, z29.s, #16
    sel z29.s, p2, z18.s, z29.s
    uzp1 z19.h, z21.h, z29.h
    st1h { z19.h }, p1, [x26]
    add x26, x26, x23
    fcmuo p2.s, p0/z, z22.s, z22.s
    lsr z16.s, z22.s, #16
    and z16.s, z16.s, #1
    add z16.s, z16.s, z17.s
    add z22.s, z22.s, z16.s
    lsr z22.s, z22.s, #16
    sel z22.s, p2, z18.s, z22.s
    fcmuo p2.s, p0/z, z30.s, z30.s
    lsr z16.s, z30.s, #16
    and z16.s, z16.s, #1
    add z16.s, z16.s, z17.s
    add z30.s, z30.s, z16.s
    lsr z30.s, z30.s, #16
    sel z30.s, p2, z18.s, z30.s
    uzp1 z19.h, z22.h, z30.h
    st1h { z19.h }, p1, [x26]
    add x26, x26, x23
    fcmuo p2.s, p0/z, z23.s, z23.s
    lsr z16.s, z23.s, #16
    and z16.s, z16.s, #1
    add z16.s, z16.s, z17.s
    add z23.s, z23.s, z16.s
    lsr z23.s, z23.s, #16
    sel z23.s, p2, z18.s, z23.s
    fcmuo p2.s, p0/z, z31.s, z31.s
    lsr z16.s, z31.s, #16
    and z16.s, z16.s, #1
    add z16.s, z16.s, z17.s
    add z31.s, z31.s, z16.s
    lsr z31.s, z31.s, #16
    sel z31.s, p2, z18.s, z31.s
    uzp1 z19.h, z23.h, z31.h
    st1h { z19.h }, p1, [x26]
    add x26, x26, x23
    blt label_13
KAI_ASM_LABEL(label_14)  // Store to output array: Accumulator row 1 oddments
    cbz x20, label_15
    KAI_ASM_INST(0xc0860444)  // mova { z4.s-z7.s }, za2h.s[x12]
    KAI_ASM_INST(0xc086046c)  // mova { z12.s-z15.s }, za3h.s[x12]
    subs x20, x20, #0x1
    KAI_ASM_INST(0xc1b8cb44)  // fclamp { z4.s-z7.s }, z26.s, z24.s
    KAI_ASM_INST(0xc1b8cb4c)  // fclamp { z12.s-z15.s }, z26.s, z24.s
    fcmuo p2.s, p0/z, z4.s, z4.s
    lsr z16.s, z4.s, #16
    and z16.s, z16.s, #1
    add z16.s, z16.s, z17.s
    add z4.s, z4.s, z16.s
    lsr z4.s, z4.s, #16
    sel z4.s, p2, z18.s, z4.s
    fcmuo p2.s, p0/z, z12.s, z12.s
    lsr z16.s, z12.s, #16
    and z16.s, z16.s, #1
    add z16.s, z16.s, z17.s
    add z12.s, z12.s, z16.s
    lsr z12.s, z12.s, #16
    sel z12.s, p2, z18.s, z12.s
    uzp1 z19.h, z4.h, z12.h
    st1h { z19.h }, p1, [x26]
    add x26, x26, x23
    beq label_15
    subs x20, x20, #0x1
    fcmuo p2.s, p0/z, z5.s, z5.s
    lsr z16.s, z5.s, #16
    and z16.s, z16.s, #1
    add z16.s, z16.s, z17.s
    add z5.s, z5.s, z16.s
    lsr z5.s, z5.s, #16
    sel z5.s, p2, z18.s, z5.s
    fcmuo p2.s, p0/z, z13.s, z13.s
    lsr z16.s, z13.s, #16
    and z16.s, z16.s, #1
    add z16.s, z16.s, z17.s
    add z13.s, z13.s, z16.s
    lsr z13.s, z13.s, #16
    sel z13.s, p2, z18.s, z13.s
    uzp1 z19.h, z5.h, z13.h
    st1h { z19.h }, p1, [x26]
    add x26, x26, x23
    beq label_15
    fcmuo p2.s, p0/z, z6.s, z6.s
    lsr z16.s, z6.s, #16
    and z16.s, z16.s, #1
    add z16.s, z16.s, z17.s
    add z6.s, z6.s, z16.s
    lsr z6.s, z6.s, #16
    sel z6.s, p2, z18.s, z6.s
    fcmuo p2.s, p0/z, z14.s, z14.s
    lsr z16.s, z14.s, #16
    and z16.s, z16.s, #1
    add z16.s, z16.s, z17.s
    add z14.s, z14.s, z16.s
    lsr z14.s, z14.s, #16
    sel z14.s, p2, z18.s, z14.s
    uzp1 z19.h, z6.h, z14.h
    st1h { z19.h }, p1, [x26]
KAI_ASM_LABEL(label_15)  // Store to output array: Accumulator row 1 oddments: End
KAI_ASM_LABEL(label_16)  // Store to output array: End
    incw x11, ALL, MUL #2
    cmp x11, x10
    blt label_2
    incw x15, ALL, MUL #2
    mov x11, #0x0
    cmp x15, x13
    mov x9, x27
    blt label_1
    KAI_ASM_INST(0xd503467f)  // SMSTOP
    ldp x22, x23, [sp, 16]
    ldp x24, x25, [sp, 32]
    ldp x26, x27, [sp, 48]
    ldr x28, [sp, 64]
    ldp d8, d9, [sp, 72]
    ldp d10, d11, [sp, 88]
    ldp d12, d13, [sp, 104]
    ldp d14, d15, [sp, 120]
    ldp x20, x21, [sp], 144
    ret
    KAI_ASM_FUNCTION_END(qwen21_bf16_register_epilogue_kernel)

    KAI_ASM_END
