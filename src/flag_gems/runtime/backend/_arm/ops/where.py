import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

from ..vector_config import ELEMENTWISE_ROLLED_TILE, SINGLE_PROGRAM_MAX_ELEMENTS


@pointwise_dynamic(
    is_tensor=[True, True, True],
    promotion_methods=[(1, 2, "NO_OPMATH")],
)
@triton.jit
def where_inner(condition, self, other):
    return tl.where(condition, self, other)


@triton.jit(do_not_specialize=["n_elements"])
def _where_tensor_tensor_kernel(
    condition_ptr,
    self_ptr,
    other_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    lanes = tl.arange(0, BLOCK_SIZE)
    full_elements = (n_elements // BLOCK_SIZE) * BLOCK_SIZE
    for base in range(
        pid * BLOCK_SIZE, full_elements, num_programs * BLOCK_SIZE
    ):
        idx = base + lanes
        cond = tl.load(condition_ptr + idx).to(tl.int1)
        lhs = tl.load(self_ptr + idx)
        rhs = tl.load(other_ptr + idx)
        tl.store(out_ptr + idx, tl.where(cond, lhs, rhs))
    if pid == 0 and full_elements < n_elements:
        idx = full_elements + lanes
        mask = idx < n_elements
        cond = tl.load(condition_ptr + idx, mask=mask, other=0).to(tl.int1)
        lhs = tl.load(self_ptr + idx, mask=mask, other=0.0)
        rhs = tl.load(other_ptr + idx, mask=mask, other=0.0)
        tl.store(out_ptr + idx, tl.where(cond, lhs, rhs), mask=mask)


@triton.jit(do_not_specialize=["scalar", "n_elements"])
def _where_scalar_self_kernel(
    condition_ptr,
    other_ptr,
    out_ptr,
    scalar,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    cond = tl.load(condition_ptr + offs, mask=mask, other=0).to(tl.int1)
    other = tl.load(other_ptr + offs, mask=mask, other=0.0)
    out = tl.where(cond, scalar, other)
    tl.store(out_ptr + offs, out, mask=mask)


@triton.jit(do_not_specialize=["scalar", "n_elements"])
def _where_scalar_other_kernel(
    condition_ptr,
    self_ptr,
    out_ptr,
    scalar,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    cond = tl.load(condition_ptr + offs, mask=mask, other=0).to(tl.int1)
    self_tensor = tl.load(self_ptr + offs, mask=mask, other=0.0)
    out = tl.where(cond, self_tensor, scalar)
    tl.store(out_ptr + offs, out, mask=mask)


@triton.jit(do_not_specialize=["scalar", "n_elements"])
def _where_scalar_self_single_program_kernel(
    condition_ptr,
    other_ptr,
    out_ptr,
    scalar,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    offs = tl.arange(0, BLOCK_SIZE)
    full_elements = (n_elements // BLOCK_SIZE) * BLOCK_SIZE
    for base in range(0, full_elements, BLOCK_SIZE):
        idx = base + offs
        cond = tl.load(condition_ptr + idx).to(tl.int1)
        other = tl.load(other_ptr + idx)
        tl.store(out_ptr + idx, tl.where(cond, scalar, other))
    if full_elements < n_elements:
        idx = full_elements + offs
        mask = idx < n_elements
        cond = tl.load(condition_ptr + idx, mask=mask, other=0).to(tl.int1)
        other = tl.load(other_ptr + idx, mask=mask, other=0.0)
        tl.store(out_ptr + idx, tl.where(cond, scalar, other), mask=mask)


@triton.jit(do_not_specialize=["scalar", "n_elements"])
def _where_scalar_other_single_program_kernel(
    condition_ptr,
    self_ptr,
    out_ptr,
    scalar,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    offs = tl.arange(0, BLOCK_SIZE)
    full_elements = (n_elements // BLOCK_SIZE) * BLOCK_SIZE
    for base in range(0, full_elements, BLOCK_SIZE):
        idx = base + offs
        cond = tl.load(condition_ptr + idx).to(tl.int1)
        self_tensor = tl.load(self_ptr + idx)
        tl.store(out_ptr + idx, tl.where(cond, self_tensor, scalar))
    if full_elements < n_elements:
        idx = full_elements + offs
        mask = idx < n_elements
        cond = tl.load(condition_ptr + idx, mask=mask, other=0).to(tl.int1)
        self_tensor = tl.load(self_ptr + idx, mask=mask, other=0.0)
        tl.store(out_ptr + idx, tl.where(cond, self_tensor, scalar), mask=mask)


def _as_scalar(v):
    if isinstance(v, torch.Tensor):
        if v.numel() != 1:
            return None
        return v.item()
    if isinstance(v, (int, float, bool)):
        return v
    return None


def _where_scalar_tensor_fastpath(condition, self, other, out):
    if not isinstance(condition, torch.Tensor) or condition.dtype is not torch.bool:
        return False
    if condition.device.type != "cpu":
        return False
    if not condition.is_contiguous() or not out.is_contiguous():
        return False

    self_scalar = _as_scalar(self)
    other_scalar = _as_scalar(other)
    self_tensor = self if isinstance(self, torch.Tensor) else None
    other_tensor = other if isinstance(other, torch.Tensor) else None

    # Only specialize one-scalar + one-tensor, contiguous, same flattened size.
    if (
        self_scalar is not None
        and other_tensor is not None
        and other_tensor.is_contiguous()
    ):
        if other_tensor.numel() != condition.numel():
            return False
        if other_tensor.dtype != out.dtype:
            return False
        cond_flat = condition.view(-1)
        other_flat = other_tensor.view(-1)
        out_flat = out.view(-1)
        n = cond_flat.numel()
        if n > 1 and (
            torch.get_num_threads() == 1
            or n <= SINGLE_PROGRAM_MAX_ELEMENTS
        ):
            block_size = min(
                ELEMENTWISE_ROLLED_TILE,
                triton.next_power_of_2(n),
            )
            _where_scalar_self_single_program_kernel[(1,)](
                cond_flat,
                other_flat,
                out_flat,
                float(self_scalar),
                n,
                BLOCK_SIZE=block_size,
                num_warps=1,
                num_stages=1,
            )
        else:
            grid = (triton.cdiv(n, 256),)
            _where_scalar_self_kernel[grid](
                cond_flat,
                other_flat,
                out_flat,
                float(self_scalar),
                n,
                BLOCK_SIZE=256,
                num_warps=1,
                num_stages=1,
            )
        return True

    if (
        other_scalar is not None
        and self_tensor is not None
        and self_tensor.is_contiguous()
    ):
        if self_tensor.numel() != condition.numel():
            return False
        if self_tensor.dtype != out.dtype:
            return False
        cond_flat = condition.view(-1)
        self_flat = self_tensor.view(-1)
        out_flat = out.view(-1)
        n = cond_flat.numel()
        if n > 1 and (
            torch.get_num_threads() == 1
            or n <= SINGLE_PROGRAM_MAX_ELEMENTS
        ):
            block_size = min(
                ELEMENTWISE_ROLLED_TILE,
                triton.next_power_of_2(n),
            )
            _where_scalar_other_single_program_kernel[(1,)](
                cond_flat,
                self_flat,
                out_flat,
                float(other_scalar),
                n,
                BLOCK_SIZE=block_size,
                num_warps=1,
                num_stages=1,
            )
        else:
            grid = (triton.cdiv(n, 256),)
            _where_scalar_other_kernel[grid](
                cond_flat,
                self_flat,
                out_flat,
                float(other_scalar),
                n,
                BLOCK_SIZE=256,
                num_warps=1,
                num_stages=1,
            )
        return True

    return False


def _where_tensor_tensor_fastpath(condition, self, other, out):
    if not all(
        isinstance(value, torch.Tensor)
        for value in (condition, self, other, out)
    ):
        return False
    if condition.dtype is not torch.bool or condition.device.type != "cpu":
        return False
    if self.dtype != other.dtype or self.dtype != out.dtype:
        return False
    if not all(value.is_contiguous() for value in (condition, self, other, out)):
        return False
    if not (
        condition.shape == self.shape == other.shape == out.shape
    ):
        return False

    n_elements = out.numel()
    if n_elements == 0:
        return True
    block_size = min(
        ELEMENTWISE_ROLLED_TILE,
        triton.next_power_of_2(max(n_elements, 1)),
    )
    programs = (
        1
        if n_elements <= SINGLE_PROGRAM_MAX_ELEMENTS
        else min(
            max(1, torch.get_num_threads()),
            triton.cdiv(n_elements, block_size),
        )
    )
    _where_tensor_tensor_kernel[(programs,)](
        condition.view(-1),
        self.view(-1),
        other.view(-1),
        out.view(-1),
        n_elements,
        BLOCK_SIZE=block_size,
        num_warps=1,
        num_stages=1,
    )
    return True


def where_self_out(condition, self, other, out=None):
    logging.debug("GEMS WHERE_SELF_OUT")
    result_type = torch.result_type(self, other)
    if out is not None:
        assert (
            out.dtype == result_type
        ), f"Expected out type to be {result_type}, but got {out.dtype}."

    c, a, b = list(
        map(
            lambda x: x if isinstance(x, torch.Tensor) else torch.tensor(x),
            (condition, self, other),
        )
    )

    if a.dtype != result_type:
        a = a.to(result_type)
    if b.dtype != result_type:
        b = b.to(result_type)

    devices = map(lambda x: x.device, (c, a, b))
    devices = list(filter(lambda k: k.type != "cpu", devices))

    # assert len(devices), "CPU only. There seems a mistake to dispatch to here."

    # device = devices[0]
    # if c.device != device and c.ndim == 0:
    #     c = c.to(device)
    # if a.device != device and a.ndim == 0:
    #     a = a.to(device)
    # if b.device != device and b.ndim == 0:
    #     b = b.to(device)

    # assert (
    #     len(set(devices)) == 1
    # ), f"Expected all tensors to be on the same device, but found at least two devices, {devices}"
    assert (
        c.dtype == torch.bool
    ), f"where expected condition to be a boolean tensor, but got a tensor with dtype {condition.dtype}"

    if out is None:
        out_shape = torch.broadcast_shapes(c.shape, a.shape, b.shape)
        out = torch.empty(out_shape, dtype=result_type, device=c.device)

    if _where_scalar_tensor_fastpath(c, a, b, out):
        return out

    if _where_tensor_tensor_fastpath(c, a, b, out):
        return out

    ndim = max(c.ndim, a.ndim, b.ndim)
    where_inner.instantiate(ndim)
    where_inner(c, a, b, out0=out)
    return out


def where_self(condition, self, other):
    logging.debug("GEMS WHERE_SELF")
    return where_self_out(condition, self, other)


def where_scalar_self(condition, self, other):
    logging.debug("GEMS WHERE_SCALAR_SELF")
    return where_self_out(condition, self, other)


def where_scalar_other(condition, self, other):
    logging.debug("GEMS WHERE_SCALAR_OTHER")
    return where_self_out(condition, self, other)
