#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
for project in "$root" "$root/envs/policy" "$root/envs/sim"; do
    uv lock --check --project "$project"
done
