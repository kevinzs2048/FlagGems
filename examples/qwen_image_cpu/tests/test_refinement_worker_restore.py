"""An interleaved off step must restore both arithmetic path and worker count."""

import pytest
import torch
from qwen_image_cpu.sme_refinements import enable, select


@pytest.mark.parametrize("original_workers", [9, 18])
def test_switch_restores_baseline_workers_and_output_path(original_workers):
    model = torch.nn.Module()
    model.transformer_blocks = torch.nn.ModuleList([torch.nn.Identity()])
    model.projection = torch.nn.Linear(8, 8, bias=False)
    layer = model.projection
    layer._sme_refinements_original_workers = original_workers
    layer._bf16_sme_workers = 12
    layer._bf16_sme_neon_pack = True
    layer._bf16_register_epilogue_enabled = True
    model._sme_refinements_mlp_workers = 12
    model._sme_refinements_register_output = True
    for enabled in [False, True, True, False]:
        select(model, enabled)
        assert layer._bf16_sme_workers == (12 if enabled else original_workers)
        assert layer._bf16_sme_neon_pack == enabled
        assert layer._bf16_register_epilogue_enabled == enabled


@pytest.mark.parametrize(
    "workers,w8", [(12, False), (0, True), (-1, True), (1.5, True)]
)
def test_invalid_workers_fail_before_model_mutation(workers, w8):
    model = torch.nn.Module()
    with pytest.raises(ValueError, match="worker"):
        enable(model, w8_swiglu=w8, mlp_workers=workers)
    assert not hasattr(model, "_sme_refinements_mlp_workers")
