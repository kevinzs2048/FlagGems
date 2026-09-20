#!/usr/bin/env bash
set -euo pipefail
# Compatibility launcher: the application is installed from flagos-macos-runtime.
exec "${QWEN_IMAGE_PYTHON:-python3}" -m qwen_image_cpu "$@"
