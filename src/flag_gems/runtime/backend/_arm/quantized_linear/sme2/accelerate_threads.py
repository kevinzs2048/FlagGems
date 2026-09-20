"""Scoped, verified per-worker Accelerate BLAS policy for isolated benchmarks."""

from contextlib import contextmanager
from functools import lru_cache

import torch


@lru_cache(None)
def ops():
    from flag_gems.runtime.backend._arm.quantized_linear.image_sme import load_ops

    return load_ops("accelerate_threads")


@contextmanager
def policy(mode):
    """Do not change the OpenMP team size or use concurrent requests in scope."""
    native = ops()
    state = native.pool_mode(mode)
    try:
        if (
            state.shape[0] != torch.get_num_threads()
            or (state[:, 3] != 0).any()
            or (state[:, 4] != mode).any()
        ):
            raise RuntimeError(
                "Requested BLAS policy was not set on the complete OpenMP team"
            )
        yield state
    finally:
        restored = native.restore_pool(state)
        expected = {r[0]: r[2] for r in state.tolist()}
        if any(r[3] != 0 or r[4] != expected.get(r[0]) for r in restored.tolist()):
            raise RuntimeError(
                "Could not restore original BLAS policy on the same OpenMP workers"
            )
