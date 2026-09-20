import pytest
import torch
from qwen_image_cpu import reference
from qwen_image_cpu.shared_modulation import enable_shared_modulation


class TinyBlocks(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList(
            [
                reference.transformer_source.QwenImage21TransformerBlock(16, 2, 8)
                for _ in range(2)
            ]
        )

    def forward(self, x, modulation, mask):
        for block in self.transformer_blocks:
            x = block(x, modulation, target_token_mask=mask)
        return x


@pytest.mark.parametrize("masked", [False, True])
@torch.inference_mode()
def test_same_values_across_requests_and_changed_shapes(masked, monkeypatch):
    torch.set_num_threads(4)
    torch.manual_seed(80)
    monkeypatch.setattr(
        reference.transformer_source, "dispatch_attention_fn", reference._cpu_dispatch
    )
    model = TinyBlocks().bfloat16().eval()
    cases = []
    for length in [5, 7, 5]:
        x = torch.randn(2, length, 16).bfloat16()
        mod = torch.randn(3 if masked else 2, 64).bfloat16()
        mask = torch.arange(length) % 2 == 0 if masked else None
        cases.append((x, mod, mask))
    expected = [model(*case) for case in cases]
    assert enable_shared_modulation(model) == 2
    assert enable_shared_modulation(model) == 0
    for case, result in zip(cases, expected):
        assert torch.equal(model(*case), result)


def test_training_falls_back(monkeypatch):
    torch.set_num_threads(4)
    torch.manual_seed(81)
    monkeypatch.setattr(
        reference.transformer_source, "dispatch_attention_fn", reference._cpu_dispatch
    )
    model = TinyBlocks().train()
    x = torch.randn(1, 5, 16)
    mod = torch.randn(1, 64)
    expected = model(x, mod, None)
    enable_shared_modulation(model)
    actual = model(x, mod, None)
    assert torch.equal(actual, expected)
