#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
mkdir -p "$TD_VLA_STORAGE_ROOT"/{data,pretrained,outputs,logs/p1,assets,cache/{uv,huggingface,torch,torch_extensions,tmp}}
for name in data pretrained outputs logs assets; do
    target="$TD_VLA_REPO_ROOT/$name"
    if [[ -L "$target" && "$(readlink -f "$target")" == "$(readlink -f "$TD_VLA_STORAGE_ROOT/$name")" ]]; then
        continue
    elif [[ -e "$target" || -L "$target" ]]; then
        echo "Existing path does not point to expected storage: $target (preserved)" >&2
        exit 1
    else
        ln -s "$TD_VLA_STORAGE_ROOT/$name" "$target"
    fi
done
df -h "$TD_VLA_REPO_ROOT" "$TD_VLA_STORAGE_ROOT"
