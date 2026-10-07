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

from .fused_add_rms_norm import fused_add_rms_norm  # noqa: F401
from .patch_qwen3_mlp import patch_qwen3_mlp, unpatch_qwen3_mlp  # noqa: F401
from .silu_mul_a8_pack_native import silu_mul_a8_pack_native  # noqa: F401
from .native_ops import arm_native_ops  # noqa: F401

__all__ = [
    "fused_add_rms_norm",
    "patch_qwen3_mlp",
    "unpatch_qwen3_mlp",
    "silu_mul_a8_pack_native",
    "arm_native_ops",
]

# Keep existing backends importable before the optional CPU TLE leaf is built.
try:
    from .silu_a8_raw import silu_mul_a8_pack_raw
except ImportError:
    pass
else:
    __all__.append("silu_mul_a8_pack_raw")
