import pytest
import torch
from qwen_image_cpu import reference
from qwen_image_cpu.attention_layout import dispatch


@pytest.mark.parametrize(
    "m,n,h,d", [(37, 37, 4, 8), (129, 137, 4, 16), (1024, 1061, 32, 128)]
)
@pytest.mark.parametrize("mask_kind", ["padding", "causal"])
@torch.inference_mode()
def test_attention_head_layout_preserves_mask_and_bf16_output(m, n, h, d, mask_kind):
    torch.set_num_threads(4)
    torch.manual_seed(325)
    q = torch.randn(1, h, m, d).bfloat16().transpose(1, 2)
    k = torch.randn(1, h, n, d).bfloat16().transpose(1, 2)
    v = torch.randn(1, h, n, d).bfloat16().transpose(1, 2)
    mask = torch.ones(1, 1, 1, n, dtype=torch.bool)
    mask[..., 3:7] = False
    if mask_kind == "causal":
        mask = mask & torch.ones(m, n, dtype=torch.bool).tril()[None, None]
    args = {
        "attn_mask": mask,
        "dropout_p": 0.0,
        "backend": None,
        "parallel_config": None,
    }
    expected = reference._cpu_dispatch(q, k, v, **args)
    actual = dispatch(q, k, v, **args)
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
    if m >= 128:
        assert actual.is_contiguous()
