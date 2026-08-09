"""Optional profiler ranges for Triton CPU launches.

Triton CPU kernels execute synchronously through generated shared libraries,
which torch.profiler cannot otherwise see.  Keep ranges disabled by default so
production decode has no context-manager overhead.
"""

from contextlib import nullcontext
import os


_ENABLED = os.getenv("FLAGGEMS_PROFILE_RANGES", "0").lower() in {
    "1",
    "true",
    "on",
}


def profile_range(name: str):
    if not _ENABLED:
        return nullcontext()
    from torch.autograd.profiler import record_function

    return record_function(name)
