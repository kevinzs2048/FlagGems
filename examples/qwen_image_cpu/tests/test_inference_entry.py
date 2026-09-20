"""Public inference arguments must preserve W8-only CPU semantics."""

import subprocess
import sys

import pytest
from qwen_image_cpu.__main__ import parse_args


def test_defaults_are_modelcard_configuration():
    a = parse_args(["--model", "weights", "--prompt", "watercolour landscape"])
    assert (a.width, a.height, a.steps, a.seed) == (1024, 1024, 40, 42)
    assert a.warmup_steps == 0 and a.runs == 1
    assert not hasattr(a, "transformer_precision")


@pytest.mark.parametrize(
    "extra",
    [
        ["--width", "513"],
        ["--height", "0"],
        ["--steps", "0"],
        ["--runs", "0"],
        ["--warmup-steps", "-1"],
        ["--output", "result.jpg"],
        ["--prompt", " "],
        ["--transformer-precision", "w4a8"],
    ],
)
def test_invalid_requests_fail_before_loading(extra):
    with pytest.raises(SystemExit) as exc:
        parse_args(["--model", "weights", "--prompt", "test", *extra])
    assert exc.value.code == 2


def test_help_does_not_load_torch_fl_or_model():
    code = (
        "from qwen_image_cpu.__main__ import parse_args; import sys; "
        "assert 'torch_fl' not in sys.modules; assert 'torch' not in sys.modules; "
        "parse_args(['--help'])"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0 and "--model" in result.stdout
