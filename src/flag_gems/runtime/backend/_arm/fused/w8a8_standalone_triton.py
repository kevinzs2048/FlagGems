# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Ordinary CPU Triton W8A8 reference for the independent N4 weight layout.

No external integer microkernel. Keep this path available for measured
comparisons; the independently implemented TLE Raw path is the tuned route.
"""
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

@triton.jit
def _quantize(x,q,z,s,K:tl.constexpr,B:tl.constexpr):
    r=tl.program_id(0);j=tl.arange(0,B)
    v=tl.load(x+r*K+j,j<K,0.)
    lo=tl.minimum(tl.min(tl.where(j<K,v,float('inf')),0),0.)
    hi=tl.maximum(tl.max(tl.where(j<K,v,float('-inf')),0),0.)
    mult=tl.where(lo==hi,1.,255./(hi-lo))
    dmin=lo*mult;dmax=hi*mult
    zp=tl.where((-128.+dmin)+(127.+dmax)>0,-128.-dmin,127.-dmax)
    zero=libdevice.rint(tl.minimum(tl.maximum(zp,-128.),127.)).to(tl.int32)
    val=v*mult
    rounded=tl.where(val>=0,tl.floor(val+0.5),-tl.floor(-val+0.5)).to(tl.int32)
    quant=tl.minimum(tl.maximum(rounded+zero,-128),127).to(tl.int8)
    tl.store(q+r*K+j,quant,j<K);tl.store(z+r,zero);tl.store(s+r,tl.where(mult!=0,1./mult,0.))

@triton.jit
def _gemv(q,z,s,w,y,N:tl.constexpr,K:tl.constexpr,B:tl.constexpr):
    b=tl.program_id(0);r=tl.program_id(1)
    j=tl.arange(0,B);c=tl.arange(0,4)
    weight=tl.load(w+b*(4*K+48)+(j[None,:]//16)*64+c[:,None]*16+j[None,:]%16,j[None,:]<K,0).to(tl.int8).to(tl.int32)
    act=tl.load(q+r*K+j,j<K,0).to(tl.int32)
    dot=tl.sum(weight*act[None,:],1)
    sums=tl.load((w+b*(4*K+48)+4*K).to(tl.pointer_type(tl.int32))+c)
    scales=tl.load((w+b*(4*K+48)+4*K+16).to(tl.pointer_type(tl.float32))+c)
    out=(dot-tl.load(z+r)*sums).to(tl.float32)*(scales*tl.load(s+r))
    tl.store(y+r*N+b*4+c,out)

def linear_triton(x,packed,n):
    if x.device.type!='cpu' or x.dtype!=torch.float32 or not x.is_contiguous() or x.ndim!=2:
        raise ValueError('Expected contiguous CPU FP32 [M,K]')
    m,k=x.shape
    if not m or n%4 or k%32 or packed.numel()!=n//4*(4*k+48):
        raise ValueError('Independent W8 layout requires N%4=0 and K%32=0')
    q=torch.empty((m,k),dtype=torch.int8);z=torch.empty(m,dtype=torch.int32);s=torch.empty(m)
    output=torch.empty((m,n))
    _quantize[(m,)](x,q,z,s,k,triton.next_power_of_2(k),enable_fp_fusion=False)
    _gemv[(n//4,m)](q,z,s,packed,output,n,k,triton.next_power_of_2(k),enable_fp_fusion=False)
    return output
