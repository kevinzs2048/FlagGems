# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Integer oracle and staged/fused equivalence for the CPU I8MM TLE route."""

import os
from pathlib import Path
import unittest
import torch

from flag_gems.runtime.backend._arm.fused.prepared_w8a8_i8mm import PreparedW8A8I8MM


@unittest.skipUnless(
    Path(os.environ.get("FLAGGEMS_ARM_W8G32_I8MM_LIBRARY", "/missing")).is_file(),
    "Requires the built Arm I8MM TLE library",
)
class TestArmW8A8I8MM(unittest.TestCase):
    def test_integer_oracle_bias_smoothing_and_empty_tails(self):
        torch.manual_seed(1030)
        with torch.inference_mode():
            for m, n, k in [
                (0, 1, 32),
                (1, 3, 32),
                (2, 67, 64),
                (3, 67, 64),
                (4, 70, 128),
                (5, 70, 128),
                (6, 70, 128),
                (7, 9, 64),
                (8, 128, 256),
                (16, 67, 128),
                (25, 70, 128),
                (31, 67, 128),
                (32, 70, 128),
                (63, 67, 128),
                (64, 70, 128),
                (128, 67, 128),
                (17, 67, 128),
                (65, 70, 128),
                (129, 67, 128),
                (257, 128, 256),
                (9, 70, 4096),
            ]:
                for smooth in [False, True]:
                    weight = torch.randn(n, k).bfloat16()
                    x = torch.randn(m, k).bfloat16()
                    if m:
                        x[0].zero_()
                    bias = torch.randn(n).bfloat16() if smooth else None
                    smoothing = torch.exp(torch.randn(k)) if smooth else None
                    wf = (
                        weight.float()
                        if smoothing is None
                        else weight.float() * smoothing
                    )
                    xf = (
                        x.float()
                        if smoothing is None
                        else x.float() * smoothing.reciprocal()
                    )
                    ws = (wf.abs().amax(1) / 127).clamp_min(1e-8)
                    qw = (wf / ws[:, None]).round().clamp(-127, 127).int()
                    peak = xf.abs().amax(1)
                    xs = torch.where(peak > 0, peak / 127, torch.ones_like(peak))
                    qx = (xf / xs[:, None]).round().clamp(-127, 127).int()
                    expected = (qx @ qw.T).float() * xs[:, None] * ws[None, :]
                    if bias is not None:
                        expected += bias.float()
                    expected = expected.bfloat16()
                    for kernel, order in [
                        ("staged", "n"),
                        ("fused", "n"),
                        ("fused", "m"),
                        ("fused", "auto"),
                    ]:
                        with self.subTest(
                            m=m, n=n, k=k, smooth=smooth, kernel=kernel, order=order
                        ):
                            op = PreparedW8A8I8MM(
                                weight,
                                8,
                                smoothing=smoothing,
                                kernel=kernel,
                                order=order,
                            )
                            actual = op(x, bias)
                            torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_fused_norm_integer_reference_rounding_and_ownership(self):
        from flag_gems.runtime.backend._arm.fused.prepared_w8a8_i8mm import (
            PreparedW8A8I8MMAddNorm,
        )
        import torch.nn.functional as F

        torch.manual_seed(1102)
        cases = [
            (0, 67, 32),
            (1, 1, 32),
            (16, 67, 64),
            (25, 67, 64),
            (17, 67, 64),
            (31, 67, 64),
            (32, 1024, 256),
            (33, 67, 64),
            (49, 67, 64),
            (63, 67, 64),
            (64, 1024, 256),
            (65, 67, 64),
            (127, 67, 64),
            (128, 1024, 256),
            (32, 1024, 4096),
        ]
        with torch.inference_mode():
            for m, n, k in cases:
                for smooth in [False, True]:
                    w = torch.randn(n, k).bfloat16()
                    x = torch.randn(m, k).bfloat16()
                    if m:
                        x[0].zero_()
                    smoothing = torch.exp(torch.randn(k)) if smooth else None
                    bias = torch.randn(n).bfloat16() if smooth else None
                    nw = torch.randn(n).bfloat16()
                    nb = torch.randn(n).bfloat16()
                    residual = torch.randn(m, n).bfloat16()
                    wf = w.float() if smoothing is None else w.float() * smoothing
                    xf = (
                        x.float()
                        if smoothing is None
                        else x.float() * smoothing.reciprocal()
                    )
                    ws = (wf.abs().amax(1) / 127).clamp_min(1e-8)
                    peak = xf.abs().amax(1)
                    xs = torch.where(peak > 0, peak / 127, torch.ones_like(peak))
                    qw = (wf / ws[:, None]).round().clamp(-127, 127).int()
                    qx = (xf / xs[:, None]).round().clamp(-127, 127).int()
                    linear = (qx @ qw.T).float() * xs[:, None] * ws[None, :]
                    if bias is not None:
                        linear += bias.float()
                    expected = F.layer_norm(
                        linear.bfloat16() + residual, (n,), nw, nb, 1e-5
                    )
                    plan = PreparedW8A8I8MM(
                        w, 8, smoothing=smoothing, kernel="fused", order="auto"
                    )
                    fused = PreparedW8A8I8MMAddNorm(plan, nw, nb)
                    for threads in [1, 3, 4, 6, 8]:
                        with self.subTest(
                            m=m, n=n, k=k, smooth=smooth, threads=threads
                        ):
                            actual = fused(x, residual, bias, threads=threads)
                            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                            saved = actual.clone()
                            again = fused(x, residual, bias, threads=threads)
                            if m:
                                self.assertNotEqual(actual.data_ptr(), again.data_ptr())
                            self.assertTrue(torch.equal(saved, actual))

    @unittest.skipUnless(
        os.environ.get("FLAGGEMS_ARM_I8MM_TENSOR_RAW") == "1",
        "Tensor Raw frontend required",
    )
    def test_tensor_raw_shapes_fake_contract_errors_and_ownership(self):
        from torch._subclasses.fake_tensor import FakeTensorMode
        from flag_gems.runtime.backend._arm.fused.prepared_w8a8_i8mm import (
            PreparedW8A8I8MMAddNorm,
        )

        plan = PreparedW8A8I8MM(
            torch.randn(5, 64).bfloat16(), 8, kernel="fused", order="m"
        )
        self.assertIsNotNone(plan._tensor_raw_ops)
        bias = torch.randn(5).bfloat16()
        for x in [
            torch.randn(64).bfloat16(),
            torch.randn(64, 17).bfloat16().t(),
            torch.randn(2, 17, 64).bfloat16(),
            torch.empty(0, 64, dtype=torch.bfloat16),
        ]:
            with torch.inference_mode():
                operators = plan._tensor_raw_ops
                plan._tensor_raw_ops = None
                expected = plan(x, bias)
                plan._tensor_raw_ops = operators
                before = x.clone()
                actual = plan(x, bias)
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                saved = actual.clone()
                again = plan(x * 2, bias)
                self.assertTrue(torch.equal(saved, actual))
                self.assertTrue(torch.equal(before, x))
                if x.numel():
                    self.assertNotEqual(actual.data_ptr(), again.data_ptr())
        x = torch.randn(17, 64).bfloat16()
        residual = torch.randn(17, 5).bfloat16()
        fused = PreparedW8A8I8MMAddNorm(
            plan, torch.ones(5).bfloat16(), torch.zeros(5).bfloat16()
        )
        with FakeTensorMode(allow_non_fake_inputs=True) as mode:
            y = plan(mode.from_tensor(x), bias)
            self.assertEqual(y.shape, (17, 5))
            self.assertEqual(y.dtype, torch.bfloat16)
            y = fused(mode.from_tensor(x), mode.from_tensor(residual), bias)
            self.assertEqual(y.shape, (17, 5))
        with self.assertRaises(RuntimeError):
            plan(x.float(), bias)
        with self.assertRaises(RuntimeError):
            plan(torch.randn(17, 63).bfloat16(), bias)
        with self.assertRaises(RuntimeError):
            fused(x, residual[:, :4], bias)
        with torch.enable_grad(), self.assertRaisesRegex(
            RuntimeError, "inference only"
        ):
            plan(torch.randn(17, 64, dtype=torch.bfloat16, requires_grad=True), bias)

    @unittest.skipUnless(
        os.environ.get("FLAGGEMS_ARM_I8MM_TENSOR_RAW") == "1",
        "Tensor Raw frontend required",
    )
    def test_fused_ffn_reference_tails_threads_fake_and_ownership(self):
        from flag_gems.runtime.backend._arm.fused.prepared_w8a8_i8mm import (
            PreparedW8A8I8MMAddNorm,
            PreparedW8A8I8MMFFN,
        )
        from torch._subclasses.fake_tensor import FakeTensorMode

        torch.manual_seed(1103)
        with torch.inference_mode():
            values = (
                torch.arange(65536, dtype=torch.int32)
                .to(torch.int16)
                .view(torch.bfloat16)
            )
            lut = torch.nn.functional.gelu(values).contiguous()
            for k, h in [(64, 128), (1024, 4096)]:
                for smooth in [False, True]:
                    up = PreparedW8A8I8MM(
                        torch.randn(h, k).bfloat16(),
                        8,
                        smoothing=torch.exp(torch.randn(k) * 0.3) if smooth else None,
                        kernel="fused",
                        order="auto",
                    )
                    down = PreparedW8A8I8MM(
                        torch.randn(k, h).bfloat16(),
                        8,
                        smoothing=torch.exp(torch.randn(h) * 0.3) if smooth else None,
                        kernel="fused",
                        order="auto",
                    )
                    norm = PreparedW8A8I8MMAddNorm(
                        down,
                        torch.randn(k).bfloat16(),
                        torch.randn(k).bfloat16(),
                        norm_threads=8,
                    )
                    ub = torch.randn(h).bfloat16() if smooth else None
                    db = torch.randn(k).bfloat16() if smooth else None
                    shapes = (
                        [0, 1, 16, 17, 25, 31, 32, 33, 63, 64, 65, 128, 129]
                        if k == 64
                        else [16, 32, 64, 128]
                    )
                    for m in shapes:
                        x = torch.randn(m, k).bfloat16()
                        before = x.clone()
                        expanded = up(x, ub)
                        activated = lut[
                            expanded.view(torch.int16).to(torch.int32) & 65535
                        ]
                        for count in ([1, 3, 4, 6, 8] if k == 64 else [8]):
                            up.threads = down.threads = count
                            op = PreparedW8A8I8MMFFN(up, norm, lut)
                            with self.subTest(
                                m=m, k=k, h=h, smooth=smooth, threads=count
                            ):
                                expected = norm(activated, x, db)
                                actual = op(x, ub, db)
                                torch.testing.assert_close(
                                    actual.view(torch.int16),
                                    expected.view(torch.int16),
                                    atol=0,
                                    rtol=0,
                                )
                                saved = actual.clone()
                                again = op(x * 2, ub, db)
                                self.assertTrue(
                                    torch.equal(
                                        actual.view(torch.int16),
                                        saved.view(torch.int16),
                                    )
                                )
                                self.assertTrue(torch.equal(x, before))
                                if m:
                                    self.assertNotEqual(
                                        actual.data_ptr(), again.data_ptr()
                                    )
                        up.threads = down.threads = 8
                    if k == 64:
                        op = PreparedW8A8I8MMFFN(up, norm, lut)
                        for x in [
                            torch.randn(k, 32).bfloat16().t(),
                            torch.randn(2, 16, k).bfloat16(),
                        ]:
                            expected = norm(
                                lut[
                                    up(x, ub).view(torch.int16).to(torch.int32) & 65535
                                ],
                                x.contiguous(),
                                db,
                            )
                            torch.testing.assert_close(
                                op(x, ub, db).view(torch.int16),
                                expected.view(torch.int16),
                                atol=0,
                                rtol=0,
                            )
                        with FakeTensorMode(allow_non_fake_inputs=True):
                            out = op(torch.empty(17, k, dtype=torch.bfloat16), ub, db)
                            self.assertEqual(out.shape, (17, k))
                            self.assertEqual(out.dtype, torch.bfloat16)
                        with self.assertRaises(ValueError):
                            PreparedW8A8I8MMFFN(up, norm, lut[:16])
                        with self.assertRaises(RuntimeError):
                            op(torch.randn(17, k), ub, db)
                        with self.assertRaises(RuntimeError):
                            op(torch.randn(17, k + 1).bfloat16(), ub, db)
                        with self.assertRaises(RuntimeError):
                            op(torch.randn(k).bfloat16(), ub, db)
        with torch.enable_grad(), self.assertRaisesRegex(
            RuntimeError, "inference only"
        ):
            op(
                torch.randn(17, op.up.k, dtype=torch.bfloat16, requires_grad=True),
                ub,
                db,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
