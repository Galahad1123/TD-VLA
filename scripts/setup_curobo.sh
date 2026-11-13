#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=12.0
export MAX_JOBS="${MAX_JOBS:-4}"
[[ -x "$CUDA_HOME/bin/nvcc" ]] || { echo 'Run setup_system.sh first' >&2; exit 1; }
# Runtime/build dependencies are explicitly pinned in envs/sim/uv.lock.
uv pip install --python "$ROBOTWIN_PYTHON" --no-build-isolation --no-deps \
    --editable "$TD_VLA_REPO_ROOT/external/curobo"
uv pip check --python "$ROBOTWIN_PYTHON"
