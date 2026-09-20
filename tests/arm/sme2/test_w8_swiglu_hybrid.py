"""Mixed GEMM must preserve rounded SwiGLU bytes and downstream BF16 output."""

import os

import pytest
import torch

from flag_gems.runtime.backend._arm.quantized_linear.image_sme import load_ops
from flag_gems.runtime.backend._arm.quantized_linear.sme2.bf16_sme import (
    ops as bf16_ops,
)
from flag_gems.runtime.backend._arm.quantized_linear.sme2.bf16_variants import (
    ops as matrix_ops,
)
from flag_gems.runtime.backend._arm.quantized_linear.sme2.pointwise import silu_table
from flag_gems.runtime.backend._arm.quantized_linear.sme2.w8a8_sme import (
    ops,
    quantize_weight,
)


@pytest.mark.parametrize(
    "m,n,k,split",
    [
        (0, 129, 32, 64),
        (1, 129, 8, 64),
        (37, 257, 264, 192),
        (129, 513, 256, 320),
        (128, 1024, 4096, 512),
    ],
)
@pytest.mark.parametrize("workers", [1, 4])
@torch.inference_mode()
def test_hybrid_packed_swiglu_and_consumer(m, n, k, split, workers):
    torch.set_num_threads(4)
    torch.manual_seed(921)
    w8, bf16, fused = ops(), bf16_ops(), load_ops("w8_swiglu")
    x = torch.randn(k, m).bfloat16().t()
    weights = [quantize_weight(torch.randn(n, k).bfloat16()) for _ in range(2)]
    full = [w8.pack(q, s) for q, s in weights]
    sme = [w8.pack(q[:split], s[:split]) for q, s in weights]
    neon = [w8.pack_neon(q[split:], s[split:]) for q, s in weights]
    lhs = w8.quant_pack_bf16(x)
    neon_lhs = w8.repack_neon_lhs(lhs, m, k)
    gate, up = [w8.linear_bf16(x, w, n, 4, 0, True).bfloat16() for w in full]
    expected = bf16.pack_lhs_reference(torch.nn.functional.silu(gate) * up)
    assert torch.equal(
        fused.run(lhs, *full, silu_table(), m, n, k, 4, 64, 256), expected
    )
    for mapping in [
        torch.zeros(os.cpu_count(), dtype=torch.long),
        torch.arange(os.cpu_count()) // 6,
    ]:
        for _ in range(2):
            actual = fused.run_hybrid(
                lhs,
                neon_lhs,
                *sme,
                *neon,
                silu_table(),
                m,
                split,
                n - split,
                k,
                workers,
                mapping,
                64,
                256,
                32,
                64
            )
            assert torch.equal(actual, expected)
        rhs = bf16.pack_direct(torch.randn(17, n).bfloat16())
        matrix = matrix_ops()
        assert torch.equal(
            matrix.matmul(actual, rhs, m, 17, n, 4, 32, 256, 2, 0),
            matrix.matmul(expected, rhs, m, 17, n, 4, 32, 256, 2, 0),
        )


@torch.inference_mode()
def test_invalid_hybrid_layout_rejected():
    torch.set_num_threads(4)
    w8, fused = ops(), load_ops("w8_swiglu")
    x = torch.zeros(32, 32, dtype=torch.bfloat16)
    q, s = quantize_weight(torch.ones(128, 32).bfloat16())
    lhs = w8.quant_pack_bf16(x)
    nl = w8.repack_neon_lhs(lhs, 32, 32)
    ps, pn = w8.pack(q[:64], s[:64]), w8.pack_neon(q[64:], s[64:])
    base = [
        lhs,
        nl,
        ps,
        ps,
        pn,
        pn,
        silu_table(),
        32,
        64,
        64,
        32,
        4,
        torch.zeros(os.cpu_count(), dtype=torch.long),
        64,
        256,
        32,
        64,
    ]
    for index, value in [(1, nl[:-1]), (12, torch.tensor([32])), (15, 16), (11, 5)]:
        args = list(base)
        args[index] = value
        with pytest.raises(RuntimeError):
            fused.run_hybrid(*args)


@torch.no_grad()
def test_projection_pack_cache_invalidates_on_updates():
    from flag_gems.runtime.backend._arm.quantized_linear.sme2.hybrid_swiglu import (
        prepare_weights,
    )
    from flag_gems.runtime.backend._arm.quantized_linear.sme2.w8a8_sme import (
        DynamicW8SMELinear,
    )

    torch.set_num_threads(4)
    layers = [
        DynamicW8SMELinear(*quantize_weight(torch.randn(256, 64).bfloat16()))
        for _ in range(2)
    ]
    a = prepare_weights(*layers, 0.5)
    assert prepare_weights(*layers, 0.5) is a
    assert prepare_weights(*layers, 0.75)[0] == 192
    layers[1].weight_scale.mul_(1.01)
    b = prepare_weights(*layers, 0.5)
    assert b is not a and not torch.equal(a[2], b[2])
    for fraction in (0, 1, float("nan"), 0.01):
        with pytest.raises(ValueError):
            prepare_weights(*layers, fraction)
