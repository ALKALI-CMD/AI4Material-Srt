#!/usr/bin/env bash
set -euo pipefail

stamp="$(date +%Y%m%d_%H%M%S)"
out="atomlens_phase1_results_${stamp}.tar.gz"

items=()
[[ -d results ]] && items+=(results)
[[ -d logs ]] && items+=(logs)
[[ -f pyproject.toml ]] && items+=(pyproject.toml)
[[ -f README.md ]] && items+=(README.md)
[[ -d docs ]] && items+=(docs)
[[ -d configs ]] && items+=(configs)

if (( ${#items[@]} == 0 )); then
  echo "ERROR: no result artifacts found to archive." >&2
  exit 2
fi

tar -czf "$out" "${items[@]}"
printf '%s\n' "$out"
