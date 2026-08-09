import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.utils import dim_compress, libentry
from flag_gems.utils import triton_lang_extension as tle

from ..vector_config import ELEMENTWISE_ROLLED_TILE

logger = logging.getLogger(__name__)

# torch.any: Tests if any elements in input evaluate to True. If the dtype of input
#            is not BOOL, then test if any elements in input evaluate to non-zero value
# In triton function, test if any elements in input evaluate to non-zero value is ok.


@triton.jit
def reduce_any(a, b):
    return a or b


@libentry()
@triton.autotune(configs=runtime.get_tuned_config("any"), key=["M", "N"])
@triton.jit
def any_kernel_dim(
    inp,
    out,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Map the program id to the row of inp it should compute.
    pid = tle.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    inp = inp + rows * N
    out = out + rows
    row_mask = rows < M

    _any = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.int1)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask and col_mask

        a = tl.load(inp + cols, mask, other=0.0)
        _any = _any or (a != 0)
    any = tl.reduce(_any, axis=1, combine_fn=reduce_any)
    tl.store(out, any[:, None], row_mask)


# @libentry()
@triton.jit
def any_kernel_1(
    inp,
    mid,
    n_elements,
    mid_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    inp_ptrs = inp + offset
    mask = offset < n_elements
    inp_val = tl.load(inp_ptrs, mask=mask, other=0.0)
    any_val = tl.reduce(inp_val != 0, axis=0, combine_fn=reduce_any)
    mid_ptr = mid + pid
    tl.store(mid_ptr, any_val)


# @libentry()
@triton.jit
def any_kernel_2(mid, out, MID_SIZE, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    mid_ptrs = mid + offset
    mask = offset < MID_SIZE
    mid_val = tl.load(mid_ptrs, mask=mask, other=0).to(tl.int1)
    any_val = tl.reduce(mid_val, axis=0, combine_fn=reduce_any)
    tl.store(out, any_val)


@triton.jit(do_not_specialize=["n_elements"])
def any_rolled_kernel(inp, out, n_elements, BLOCK_SIZE: tl.constexpr):
    lanes = tl.arange(0, BLOCK_SIZE)
    state = 0
    full_elements = (n_elements // BLOCK_SIZE) * BLOCK_SIZE
    for base in range(0, full_elements, BLOCK_SIZE):
        tile = (tl.load(inp + base + lanes) != 0).to(tl.int32)
        state |= tl.max(tile, axis=0)
    if full_elements < n_elements:
        idx = full_elements + lanes
        mask = idx < n_elements
        tile = (tl.load(inp + idx, mask=mask, other=0) != 0).to(tl.int32)
        state |= tl.max(tile, axis=0)
    tl.store(out, state != 0)


def any(inp):
    logger.debug("GEMS ANY")
    n_elements = inp.numel()
    out = torch.empty([], dtype=torch.bool, device=inp.device)
    if n_elements == 0:
        out.fill_(False)
        return out
    any_rolled_kernel[(1,)](
        inp.contiguous().reshape(-1),
        out,
        n_elements,
        BLOCK_SIZE=ELEMENTWISE_ROLLED_TILE,
        num_warps=1,
        num_stages=1,
    )
    return out


def any_dim(inp, dim=None, keepdim=False):
    logger.debug("GEMS ANY DIM")
    shape = list(inp.shape)
    if dim is None:
        out = any(inp)
        if keepdim:
            out = torch.reshape(out, [1] * inp.ndim)
    else:
        assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
        dim = dim % inp.ndim
        inp = dim_compress(inp, dim)
        N = shape[dim]
        shape[dim] = 1
        M = inp.numel() // N

        out = torch.empty(shape, dtype=torch.bool, device=inp.device)

        grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
        any_kernel_dim[grid](inp, out, M, N)
        if not keepdim:
            out = out.squeeze(dim=dim)
    return out


def any_dims(inp, dim=None, keepdim=False):
    logger.debug("GEMS ANY DIMS")

    if dim is None or isinstance(dim, int):
        return any_dim(inp, dim=dim, keepdim=keepdim)
    assert ((i >= -inp.ndim and i < inp.ndim) for i in dim), "Invalid dim"

    shape = list(inp.shape)
    dim = [d % inp.ndim for d in dim]
    inp = dim_compress(inp, dim)
    N = 1
    for i in dim:
        N *= shape[i]
        shape[i] = 1
    M = inp.numel() // N

    out = torch.empty(shape, dtype=torch.bool, device=inp.device)

    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
    any_kernel_dim[grid](inp, out, M, N)
    if not keepdim:
        out = out.squeeze(dim=dim)
    return out
