#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
case "${1:-}" in
    dev) project="$TD_VLA_REPO_ROOT" ;;
    policy|sim) project="$TD_VLA_REPO_ROOT/envs/$1" ;;
    *) echo 'Usage: bash scripts/setup_env.sh dev|policy|sim' >&2; exit 2 ;;
esac
[[ -d "$TMPDIR" ]] || { echo 'Run setup_storage.sh first' >&2; exit 1; }
uv sync --project "$project" --python "$(cat "$TD_VLA_REPO_ROOT/.python-version")" --locked
