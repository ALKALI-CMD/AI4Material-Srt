#!/usr/bin/env bash
set -euo pipefail

python_bin="${PYTHON:-python}"

mkdir -p logs results

{
  echo "== Python =="
  command -v "$python_bin"
  "$python_bin" --version

  echo "== Editable install =="
  "$python_bin" -m pip freeze |
    grep -E '^(torch|triton|cuda-|nvidia-.*cu13|nvidia-cublas|nvidia-cuda-|nvidia-cudnn|nvidia-cufft|nvidia-cufile|nvidia-curand|nvidia-cusolver|nvidia-cusparse|nvidia-nccl|nvidia-nvjitlink|nvidia-nvshmem|nvidia-nvtx)==' |
    xargs -r "$python_bin" -m pip uninstall -y
  "$python_bin" -m pip install -e ".[dev]"

  echo "== Unit tests =="
  "$python_bin" -m pytest -q

  echo "== Harmonic phase 1 smoke =="
  atomlens smoke --calculator harmonic --steps 65 --out results/phase1_smoke_harmonic.json
  atomlens attribute --calculator harmonic --structure-pair cspbi3_iodine_displacement --steps 65 --device cpu --out results/phase1_attribute_harmonic.json
  atomlens convergence --calculator harmonic --structure-pair cspbi3_iodine_displacement --steps 17 33 65 129 --device cpu --out results/phase1_convergence_harmonic.json
  atomlens validate-results --attribution results/phase1_attribute_harmonic.json --convergence results/phase1_convergence_harmonic.json --out results/phase1_validation_harmonic.json
  atomlens plot-attribution --input results/phase1_attribute_harmonic.json --out results/phase1_attribute_harmonic.png
  atomlens plot-convergence --input results/phase1_convergence_harmonic.json --out results/phase1_convergence_harmonic.png
  atomlens export-structure --attribution results/phase1_attribute_harmonic.json --structure-pair cspbi3_iodine_displacement --which end --out results/phase1_attribute_harmonic.extxyz
  atomlens report --attribution results/phase1_attribute_harmonic.json --convergence results/phase1_convergence_harmonic.json --attribution-figure results/phase1_attribute_harmonic.png --convergence-figure results/phase1_convergence_harmonic.png --structure-file results/phase1_attribute_harmonic.extxyz --out results/phase1_report.md
  atomlens compare-attributions --left results/phase1_attribute_harmonic.json --right results/phase1_attribute_harmonic.json --out results/phase1_compare_self.json
  atomlens plot-comparison --input results/phase1_compare_self.json --out results/phase1_compare_self.png
  atomlens manifest --out results/manifest.json

  echo "== Optional GPU preflight =="
  if command -v nvidia-smi >/dev/null 2>&1; then
    bash scripts/check_remote_gpu.sh
  else
    echo "nvidia-smi not found; skipping GPU preflight during bootstrap."
  fi
} 2>&1 | tee logs/bootstrap_remote.log
