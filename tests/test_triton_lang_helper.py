# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import SimpleNamespace

import pytest
import torch
import triton
import triton.language as tl

import flag_gems
from flag_gems.utils import triton_lang_helper


def test_patch_missing_nearbyint_uses_rint():
    rint = object()
    module = SimpleNamespace(rint=rint)

    patched = triton_lang_helper._patch_missing_symbols(module, ("nearbyint",))

    assert patched.nearbyint is rint


def test_patch_missing_hypot_preserves_backend_symbol():
    native = object()
    module = SimpleNamespace(hypot=native)
    triton_lang_helper._patch_missing_symbols(module, ("hypot",))
    assert module.hypot is native


def test_patch_missing_hypot_uses_portable_fallback():
    module = SimpleNamespace()
    triton_lang_helper._patch_missing_symbols(module, ("hypot",))
    assert module.hypot is triton_lang_helper._fallback_hypot


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_hypot_fallback_extreme_values(dtype):
    @triton.jit
    def kernel(x, y, out, COUNT: tl.constexpr, BLOCK: tl.constexpr):
        index = tl.arange(0, BLOCK)
        a = tl.load(x + index, index < COUNT, other=0)
        b = tl.load(y + index, index < COUNT, other=0)
        result = triton_lang_helper._fallback_hypot(a, b)
        tl.store(out + index, result, index < COUNT)

    info = torch.finfo(dtype)
    x = torch.tensor([0, -0.0, 3, -3, info.max / 4, info.tiny,
                      float("inf"), float("nan"), float("nan"), info.max],
                     dtype=dtype, device=flag_gems.device)
    y = torch.tensor([0, -0.0, 4, -4, info.max / 4, info.tiny,
                      float("nan"), float("inf"), 1, info.max],
                     dtype=dtype, device=flag_gems.device)
    result = torch.empty_like(x)
    kernel[(1,)](x, y, result, x.numel(), triton.next_power_of_2(x.numel()),
                 enable_fp_fusion=False)
    torch.testing.assert_close(result, torch.hypot(x, y),
                               rtol=1e-6 if dtype == torch.float32 else 1e-12,
                               atol=0, equal_nan=True)
    assert not torch.signbit(result[:2]).any()
