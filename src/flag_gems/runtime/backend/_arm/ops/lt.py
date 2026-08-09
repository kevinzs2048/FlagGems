import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

from ..vector_config import ELEMENTWISE_ROLLED_TILE, SINGLE_PROGRAM_MAX_ELEMENTS


@pointwise_dynamic(promotion_methods=[(0, 1, "ALWAYS_BOOL")])
@triton.jit
def lt_func(x, y):
    return x.to(tl.float32) < y


@triton.jit
def lt_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr = 16,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    lanes = tl.arange(0, BLOCK_SIZE)
    full_elements = (n_elements // BLOCK_SIZE) * BLOCK_SIZE
    for block_start in range(
        pid * BLOCK_SIZE, full_elements, num_programs * BLOCK_SIZE
    ):
        offsets = block_start + lanes
        x_vals = tl.load(x_ptr + offsets)
        y_vals = tl.load(y_ptr + offsets)
        tl.store(out_ptr + offsets, x_vals < y_vals)

    # Keep the predicate out of the hot loop.  Generic masked loads currently
    # lower to per-lane scalar guards in triton-cpu; only the final partial
    # vector needs those semantics.
    if pid == 0 and full_elements < n_elements:
        offsets = full_elements + lanes
        mask = offsets < n_elements
        x_vals = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        y_vals = tl.load(y_ptr + offsets, mask=mask, other=0.0)
        tl.store(out_ptr + offsets, x_vals < y_vals, mask=mask)


def lt_block(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.device == y.device, "Tensors must be on the same device"
    # Do not cast BF16/FP16 tensors to a temporary FP32 tensor.  Triton/LLVM
    # performs the comparison in the appropriate compute type and the direct
    # loads keep this path bandwidth-bound like the ATen implementation.
    x_b, y_b = torch.broadcast_tensors(x, y)
    out = torch.empty_like(x_b, dtype=torch.bool)

    # Flatten tensors for Triton kernel
    x_b_flat = x_b.contiguous().view(-1)
    y_b_flat = y_b.contiguous().view(-1)
    out_flat = out.view(-1)

    n_elements = out_flat.numel()

    programs = (
        1
        if n_elements <= SINGLE_PROGRAM_MAX_ELEMENTS
        else min(
            max(1, torch.get_num_threads()),
            triton.cdiv(n_elements, ELEMENTWISE_ROLLED_TILE),
        )
    )
    lt_kernel[(programs,)](
        x_b_flat,
        y_b_flat,
        out_flat,
        n_elements,
        BLOCK_SIZE=ELEMENTWISE_ROLLED_TILE,
        num_warps=1,
        num_stages=1,
    )

    return out


def lt(A, B):
    logging.debug("GEMS LT")
    return lt_block(A, B)
    # return lt_func(A, B)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "ALWAYS_BOOL")])
@triton.jit
def lt_func_scalar(x, y):
    return x.to(tl.float32) < y


def lt_scalar(A, B):
    logging.debug("GEMS LT SCALAR")
    return lt_func_scalar(A, B)
