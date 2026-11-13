#!/usr/bin/env bash
# Source from bash. Override TD_VLA_STORAGE_ROOT before sourcing on other machines.
TD_VLA_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export TD_VLA_REPO_ROOT
export TD_VLA_STORAGE_ROOT="${TD_VLA_STORAGE_ROOT:-/root/autodl-tmp/td-vla}"
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR="$TD_VLA_STORAGE_ROOT/cache/uv"
export UV_LINK_MODE=copy
export UV_PYTHON_PREFERENCE=only-managed
export HF_HOME="$TD_VLA_STORAGE_ROOT/cache/huggingface"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export TORCH_HOME="$TD_VLA_STORAGE_ROOT/cache/torch"
export TORCH_EXTENSIONS_DIR="$TD_VLA_STORAGE_ROOT/cache/torch_extensions"
export TMPDIR="$TD_VLA_STORAGE_ROOT/cache/tmp"
export ROBOTWIN_PATH="$TD_VLA_REPO_ROOT/external/RoboTwin"
export STARVLA_PYTHON="$TD_VLA_REPO_ROOT/envs/policy/.venv/bin/python"
export ROBOTWIN_PYTHON="$TD_VLA_REPO_ROOT/envs/sim/.venv/bin/python"
