import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems import runtime

# from ..runtime import torch_device_fn
# from ..utils import libentry
from flag_gems.utils import triton_lang_extension as tle
from ..profile_range import profile_range
from ..vector_config import REDUCTION_TILE

_VOCAB_ASSUME_FINITE = False


def set_argmax_vocab_assume_finite(enabled: bool) -> None:
    """Select the finite-logits fast path for model-owned decode routing."""
    global _VOCAB_ASSUME_FINITE
    _VOCAB_ASSUME_FINITE = bool(enabled)


# @libentry()
@triton.jit
def argmax_kernel_1(
    inp,
    mid_value,
    mid_index,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    inp_ptrs = inp + offset
    mask = offset < M
    inp_val = tl.load(inp_ptrs, mask=mask, other=-float("inf"))
    max_val, max_index = tl.max(inp_val, axis=0, return_indices=True)
    max_index = max_index + pid * BLOCK_SIZE
    mid_value_ptr = mid_value + pid
    max_index_ptr = mid_index + pid
    tl.store(mid_value_ptr, max_val)
    tl.store(max_index_ptr, max_index)


# @libentry()
@triton.jit
def argmax_kernel_2(mid_value, mid_index, out, mid_size, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    mid_ptrs = mid_value + offset
    mask = offset < mid_size
    mid_val = tl.load(mid_ptrs, mask=mask, other=-float("inf"))
    index_val = tl.argmax(mid_val, axis=0)
    mid_index_ptrs = mid_index + index_val
    out_val = tl.load(mid_index_ptrs)
    tl.store(out, out_val)


@triton.jit
def argmax_vocab_rolled_kernel(
    inp,
    out,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    ASSUME_FINITE: tl.constexpr,
):
    """CPU vocabulary argmax with a small vector carried through a rolled loop.

    The GPU-oriented two-stage implementation materializes vector<512> values
    and expands the value/index reduction into a large shuffle network.  This
    form keeps one winner per native-width lane, then performs the two
    horizontal reductions only once after scanning the vocabulary.
    """
    lanes = tl.arange(0, BLOCK_SIZE)
    best_values = tl.full((BLOCK_SIZE,), -float("inf"), tl.float32)
    # Vocabulary sizes are i32.  Keeping the lane state in i32 halves index
    # register pressure; only widen the final scalar for aten::argmax's ABI.
    best_indices = tl.full((BLOCK_SIZE,), 0, tl.int32)
    if not ASSUME_FINITE:
        first_nan = tl.full((BLOCK_SIZE,), n_elements, tl.int32)

    for off in range(0, n_elements, BLOCK_SIZE):
        offsets = off + lanes
        values = tl.load(
            inp + offsets,
            mask=offsets < n_elements,
            other=-float("inf"),
        ).to(tl.float32)
        if not ASSUME_FINITE:
            is_nan = values != values
            first_nan = tl.minimum(
                first_nan, tl.where(is_nan, offsets, n_elements)
            )
        update = values > best_values
        best_values = tl.where(update, values, best_values)
        best_indices = tl.where(update, offsets, best_indices)

    max_value = tl.max(best_values, axis=0)
    value_candidate = tl.where(
        best_values == max_value, best_indices, n_elements
    )
    value_index = tl.min(value_candidate, axis=0)
    if ASSUME_FINITE:
        result = value_index
    else:
        nan_index = tl.min(first_nan, axis=0)
        result = tl.where(nan_index < n_elements, nan_index, value_index)
    tl.store(out, result.to(tl.int64))


# @libentry()
@triton.heuristics(runtime.get_heuristic_config("argmax"))
@triton.jit
def argmax_kernel(
    inp,
    out_index,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # set offset
    pid_m = tle.program_id(0)
    pid_k = tle.program_id(1)
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    max_values = tl.full([BLOCK_M], dtype=tl.float32, value=float("-inf"))
    argmax_values = tl.full([BLOCK_M], dtype=tl.int64, value=0)
    for start_n in range(0, N, BLOCK_N):
        n_offset = start_n + tl.arange(0, BLOCK_N)
        offset = m_offset[:, None] * N * K + n_offset[None, :] * K + pid_k
        mask = (m_offset[:, None] < M) & (n_offset[None, :] < N)
        inp_ptrs = inp + offset
        inp_vals = tl.load(inp_ptrs, mask=mask, other=-float("inf")).to(tl.float32)
        local_max, local_argmax = tl.max(
            inp_vals, 1, return_indices=True, return_indices_tie_break_left=True
        )
        # if return indices is not supported, call a tl.argmax in addition
        # local_argmax = tl.argmax(inp_vals, 1)
        update = local_max > max_values
        max_values = tl.where(update, local_max, max_values)
        argmax_values = tl.where(update, start_n + local_argmax, argmax_values)

    offset_index = m_offset * K + pid_k
    out_index_ptrs = out_index + offset_index
    mask1 = m_offset < M
    tl.store(out_index_ptrs, argmax_values, mask=mask1)


def argmax(inp, dim=None, keepdim=False, *, dtype=None):
    logging.debug("GEMS ARGMAX")
    if dim is None:
        M = inp.numel()
        if dtype is None:
            dtype = inp.dtype
        block_size = triton.next_power_of_2(math.ceil(math.sqrt(M)))
        mid_size = triton.cdiv(M, block_size)
        block_mid = triton.next_power_of_2(mid_size)

        mid_value = torch.empty((mid_size,), dtype=dtype, device=inp.device)
        mid_index = torch.empty((mid_size,), dtype=torch.int64, device=inp.device)
        if keepdim:
            shape = list(inp.shape)
            for i in range(0, inp.dim()):
                shape[i] = 1
            out = torch.empty(shape, dtype=torch.int64, device=inp.device)
        else:
            out = torch.empty([], dtype=torch.int64, device=inp.device)

        # with torch_device_fn.device(inp.device):
        argmax_kernel_1[(mid_size, 1, 1)](
            inp,
            mid_value,
            mid_index,
            M,
            block_size,
        )
        argmax_kernel_2[(1, 1, 1)](mid_value, mid_index, out, mid_size, block_mid)
        return out
    else:
        assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
        shape = inp.shape
        dim = dim % inp.ndim
        if inp.numel() == 0:
            out_shape = list(shape)
            if keepdim:
                out_shape[dim] = 1
            else:
                del out_shape[dim]
            return torch.zeros(out_shape, dtype=torch.int64, device=inp.device)
        N = shape[dim]
        M = math.prod(shape[:dim])
        K = inp.numel() // M // N

        inp = inp.contiguous()

        shape_list = list(shape)
        shape_list[dim] = 1
        out_index = torch.empty(shape_list, dtype=torch.int64, device=inp.device)
        if not keepdim:
            out_index = torch.squeeze(out_index, dim)

        # Decode-heavy path frequently reduces a single row over vocab; use
        # a single rolled program on CPU.  Restrict the specialized path to
        # floating logits because it promotes BF16/FP16 comparisons to FP32.
        if M == 1 and K == 1 and inp.dtype in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
        ):
            flat_out = out_index.reshape(-1)
            with profile_range("triton::argmax_vocab"):
                argmax_vocab_rolled_kernel[(1, 1, 1)](
                    inp.reshape(-1),
                    flat_out,
                    N,
                    BLOCK_SIZE=REDUCTION_TILE,
                    ASSUME_FINITE=_VOCAB_ASSUME_FINITE,
                    num_warps=1,
                    num_stages=1,
                )
            return out_index

        grid = lambda meta: (
            triton.cdiv(M, meta["BLOCK_M"]),
            K,
        )
        # with torch_device_fn.device(inp.device):
        argmax_kernel[grid](
            inp,
            out_index,
            M,
            N,
            K,
        )

        return out_index
