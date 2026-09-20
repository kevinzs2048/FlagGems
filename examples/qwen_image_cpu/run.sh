#!/usr/bin/env bash
set -euo pipefail
QWEN_EXAMPLE_ROOT="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
QWEN_GEMS_ROOT="$(CDPATH= cd -- "$QWEN_EXAMPLE_ROOT/../.." && pwd)"
QWEN_PYTHON="${QWEN_IMAGE_PYTHON:-python3}"
export PYTHONPATH="$QWEN_EXAMPLE_ROOT:$QWEN_GEMS_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export GEMS_VENDOR=arm
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MAX_JOBS="${MAX_JOBS:-2}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$HOME/Library/Caches/qwen-image-cpu/torch-extensions}"
exec "$QWEN_PYTHON" -m qwen_image_cpu "$@"
