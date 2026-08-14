"""One-shot vLLM kernel-route coverage for the FlagGems Arm runtime."""

from __future__ import annotations

import functools
import json
import os
import platform
import tempfile
import time
from pathlib import Path

import torch

_INSTALLED = False
_ARMED = False
_PHASES: dict[str, object] = {}


def _attention_backend_names(runner) -> list[str]:
    names: list[str] = []
    for groups in getattr(runner, "attn_groups", ()):
        candidates = (groups,) if hasattr(groups, "backend") else groups
        for group in candidates:
            backend = getattr(group, "backend", None)
            if backend is not None:
                names.append(
                    getattr(backend, "__name__", type(backend).__name__)
                )
    return names


def _write_report(runner, path: Path) -> None:
    from flag_gems.integrations.vllm.qwen_gdn import route_stats
    from flag_gems.runtime.backend._arm.q4.linear import stats

    backends = _attention_backend_names(runner)
    attention_backend = (
        "CPUAttentionBackend"
        if any("CPUAttentionBackend" in name for name in backends)
        else ",".join(sorted(set(backends)))
    )
    payload = {
        "schema_version": 1,
        "captured_at_unix": time.time(),
        "pid": os.getpid(),
        "platform": {
            "machine": platform.machine(),
            "macos": platform.mac_ver()[0],
        },
        "strict": os.getenv("FLAGGEMS_ARM_Q4_STRICT", "1") != "0",
        "fallback_allowed": os.getenv("FLAGGEMS_FALLBACK_MODE", "0") == "1",
        "phases": _PHASES,
        "route_stats": stats(),
        "gdn_route_stats": route_stats(),
        "attention_backend": attention_backend,
        "attention_backends_observed": sorted(set(backends)),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def maybe_install_kernel_coverage() -> bool:
    """Install the probe only when the launcher supplies an output path."""
    global _INSTALLED
    if _INSTALLED:
        return False
    configured = os.getenv("FLAGGEMS_KERNEL_COVERAGE_FILE")
    if not configured:
        return False
    if not hasattr(torch.ops.triton_jit_cpu, "launch_profile_start"):
        raise RuntimeError(
            "FlagGems coverage requires Arm launch profiling operators"
        )

    from flag_gems.integrations.vllm.qwen_gdn import reset_route_stats
    from vllm.v1.worker.cpu_model_runner import CPUModelRunner

    output_path = Path(configured).expanduser()
    original_execute = CPUModelRunner.execute_model

    @functools.wraps(original_execute)
    def covered_execute(self, scheduler_output, intermediate_tensors=None):
        global _ARMED
        arm_file = os.getenv("FLAGGEMS_KERNEL_COVERAGE_ARM_FILE")
        if arm_file and not Path(arm_file).is_file():
            return original_execute(self, scheduler_output, intermediate_tensors)
        if not _ARMED:
            reset_route_stats()
            _PHASES.clear()
            _ARMED = True

        scheduled = int(scheduler_output.total_num_scheduled_tokens)
        phase = "prefill" if scheduled >= 4 else "decode"
        if scheduled <= 0 or phase in _PHASES:
            return original_execute(self, scheduler_output, intermediate_tensors)

        torch.ops.triton_jit_cpu.launch_profile_start()
        try:
            result = original_execute(self, scheduler_output, intermediate_tensors)
        finally:
            captured = json.loads(torch.ops.triton_jit_cpu.launch_profile_stop())
            captured["scheduled_tokens"] = scheduled
            _PHASES[phase] = captured
            _write_report(self, output_path)
        return result

    CPUModelRunner.execute_model = covered_execute
    CPUModelRunner._flag_gems_coverage_original_execute = original_execute
    _INSTALLED = True
    print(
        "[flag_gems] one-shot Arm kernel coverage armed",
        flush=True,
    )
    return True


__all__ = ["maybe_install_kernel_coverage"]
