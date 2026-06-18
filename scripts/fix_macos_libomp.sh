#!/usr/bin/env bash
# macOS-only: LightGBM's compiled extension links against Homebrew's libomp.dylib, which
# is a different file than the one PyTorch bundles. Loading both crashes the process the
# first time LightGBM runs. Re-point LightGBM's copy at PyTorch's so only one is ever
# loaded. Scoped to this project's .venv; rerun after recreating the venv.
set -euo pipefail

VENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/.venv"
TORCH_LIBOMP=$(find "$VENV_DIR" -path "*/torch/lib/libomp.dylib" | head -1)
LIGHTGBM_DYLIB=$(find "$VENV_DIR" -name "lib_lightgbm.dylib" | head -1)

if [[ -z "$TORCH_LIBOMP" || -z "$LIGHTGBM_DYLIB" ]]; then
  echo "torch or lightgbm not found in $VENV_DIR — install dependencies first" >&2
  exit 1
fi

install_name_tool -change @rpath/libomp.dylib "$TORCH_LIBOMP" "$LIGHTGBM_DYLIB"
echo "Patched $LIGHTGBM_DYLIB to load $TORCH_LIBOMP"
