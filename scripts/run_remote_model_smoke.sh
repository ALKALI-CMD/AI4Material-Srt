#!/usr/bin/env bash
set -euo pipefail

calculator="${1:-mace-mp}"
structure="${2:-diamond_si}"
device="${3:-cuda}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "ERROR: set CUDA_VISIBLE_DEVICES to one or two GPU ids before running." >&2
  exit 2
fi

gpu_count=$(python - <<'PY'
import os
visible = [item for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if item.strip()]
print(len(visible))
PY
)

if (( gpu_count > 2 )); then
  echo "ERROR: CUDA_VISIBLE_DEVICES exposes ${gpu_count} GPUs; the project limit is 2 per process." >&2
  exit 2
fi

if [[ "${ATOMLENS_SKIP_SETUP:-0}" != "1" ]]; then
  python -m pip install -e ".[dev]"
  python -m pytest -q
fi
atomlens manifest --out "results/manifest_${calculator}_${structure}.json"
atomlens model-smoke \
  --calculator "$calculator" \
  --structure "$structure" \
  --device "$device" \
  --out "results/${calculator}_${structure}.json"

if [[ "$structure" == "cubic_cspbi3" ]]; then
  atomlens attribute \
    --calculator "$calculator" \
    --structure-pair cspbi3_iodine_displacement \
    --steps 65 \
    --device "$device" \
    --out "results/${calculator}_cspbi3_path.json"
  atomlens convergence \
    --calculator "$calculator" \
    --structure-pair cspbi3_iodine_displacement \
    --steps 17 33 65 129 \
    --device "$device" \
    --out "results/${calculator}_cspbi3_convergence.json"
  atomlens validate-results \
    --attribution "results/${calculator}_cspbi3_path.json" \
    --convergence "results/${calculator}_cspbi3_convergence.json" \
    --out "results/${calculator}_cspbi3_validation.json"
  atomlens plot-attribution \
    --input "results/${calculator}_cspbi3_path.json" \
    --out "results/${calculator}_cspbi3_path.png"
  atomlens plot-convergence \
    --input "results/${calculator}_cspbi3_convergence.json" \
    --out "results/${calculator}_cspbi3_convergence.png"
  atomlens export-structure \
    --attribution "results/${calculator}_cspbi3_path.json" \
    --structure-pair cspbi3_iodine_displacement \
    --which end \
    --out "results/${calculator}_cspbi3_path.extxyz"
  atomlens report \
    --attribution "results/${calculator}_cspbi3_path.json" \
    --convergence "results/${calculator}_cspbi3_convergence.json" \
    --attribution-figure "results/${calculator}_cspbi3_path.png" \
    --convergence-figure "results/${calculator}_cspbi3_convergence.png" \
    --structure-file "results/${calculator}_cspbi3_path.extxyz" \
    --out "results/${calculator}_cspbi3_report.md"
fi
