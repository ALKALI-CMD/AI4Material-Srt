#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
out_dir="$root/dist"
stamp="$(date +%Y%m%d_%H%M%S)"
out="$out_dir/atomlens_phase1_${stamp}.tar.gz"

mkdir -p "$out_dir"
tar \
  --exclude='__pycache__' \
  --exclude='.pytest_cache' \
  --exclude='*.egg-info' \
  --exclude='dist' \
  -czf "$out" \
  -C "$root" \
  .gitignore README.md pyproject.toml configs docs scripts src tests results

printf '%s\n' "$out"
