import pytest
import torch
from qwen_image_cpu.pointwise_fusion import enable_residual, ops


def check(actual, expected):
    assert torch.equal(actual.isnan(), expected.isnan())
    valid = ~expected.isnan()
    assert torch.equal(
        actual[valid].view(torch.int16), expected[valid].view(torch.int16)
    )


@pytest.mark.parametrize(
    "shape", [(0, 7, 1), (1, 0, 3), (1, 3, 1), (2, 7, 13), (1, 1024, 4096)]
)
@pytest.mark.parametrize("broadcast", [False, True])
@torch.inference_mode()
def test_residual_rounding_shapes_and_broadcast(shape, broadcast):
    torch.set_num_threads(4)
    torch.manual_seed(961)
    x = torch.randn(*shape, 2).bfloat16()[..., 0]
    y = torch.randn_like(x)
    b, s, d = shape
    g = torch.randn(b, 1 if broadcast else s, d).bfloat16()
    check(ops().residual(x, y, g), x + g * y)


@torch.inference_mode()
def test_residual_all_bf16_encodings():
    torch.set_num_threads(4)
    torch.manual_seed(962)
    all_values = (
        torch.arange(65536, dtype=torch.int32)
        .to(torch.int16)
        .view(torch.bfloat16)
        .reshape(1, 512, 128)
    )
    random = torch.randn_like(all_values)
    for x, y, g in [
        (all_values, random, random),
        (random, all_values, random),
        (random, random, all_values),
    ]:
        check(ops().residual(x, y, g), x + g * y)


@torch.inference_mode()
def test_residual_adapter_matches_shared_modulation(monkeypatch):
    from qwen_image_cpu import reference
    from qwen_image_cpu.shared_modulation import enable_shared_modulation

    torch.set_num_threads(4)
    torch.manual_seed(963)
    monkeypatch.setattr(
        reference.transformer_source, "dispatch_attention_fn", reference._cpu_dispatch
    )

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer_blocks = torch.nn.ModuleList(
                [reference.transformer_source.QwenImage21TransformerBlock(16, 2, 8)]
            )

        def forward(self, x, mod, mask):
            return self.transformer_blocks[0](x, mod, target_token_mask=mask)

    model = Model().bfloat16().eval()
    enable_shared_modulation(model)
    cases = []
    for masked in [False, True]:
        x = torch.randn(2, 7, 16).bfloat16()
        mod = torch.randn(3 if masked else 2, 64).bfloat16()
        mask = torch.arange(7) % 2 == 0 if masked else None
        cases.append((x, mod, mask, model(x, mod, mask)))
    assert enable_residual(model) == 1
    assert enable_residual(model) == 0
    for x, mod, mask, gold in cases:
        check(model(x, mod, mask), gold)
    model.transformer_blocks[0]._fused_residual_enabled = False
    for x, mod, mask, gold in cases:
        check(model(x, mod, mask), gold)
