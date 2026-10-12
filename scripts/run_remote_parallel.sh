#!/usr/bin/env bash
set -euo pipefail

gpu_a="${1:-0}"
gpu_b="${2:-1}"

mkdir -p results logs
rm -f logs/*.pid

python -m pip install -e ".[dev,models,viz]"
python -m pytest -q
atomlens manifest --out results/manifest_remote_parallel.json

run_job() {
  local gpu="$1"
  local calculator="$2"
  local structure="$3"
  local log_file="logs/${calculator}_${structure}.log"

  (
    set -euo pipefail
    export CUDA_VISIBLE_DEVICES="$gpu"
    export ATOMLENS_SKIP_SETUP=1
    bash scripts/run_remote_model_smoke.sh "$calculator" "$structure" cuda
  ) >"$log_file" 2>&1 &

  echo "$!" >"logs/${calculator}_${structure}.pid"
  printf 'started %s on GPU %s, log=%s\n' "$calculator" "$gpu" "$log_file"
}

run_job "$gpu_a" mace-mp cubic_cspbi3
run_job "$gpu_b" chgnet cubic_cspbi3

status=0
for pid_file in logs/*.pid; do
  pid="$(cat "$pid_file")"
  if ! wait "$pid"; then
    printf 'job failed: %s\n' "$pid_file" >&2
    log_file="${pid_file%.pid}.log"
    if [[ -f "$log_file" ]]; then
      tail -n 80 "$log_file" >&2 || true
    fi
    status=1
  fi
done

if [[ -f results/mace-mp_cspbi3_path.json && -f results/chgnet_cspbi3_path.json ]]; then
  atomlens compare-attributions \
    --left results/mace-mp_cspbi3_path.json \
    --right results/chgnet_cspbi3_path.json \
    --out results/mace-mp_vs_chgnet_cspbi3_comparison.json
  atomlens plot-comparison \
    --input results/mace-mp_vs_chgnet_cspbi3_comparison.json \
    --out results/mace-mp_vs_chgnet_cspbi3_comparison.png
fi

python - <<'PY'
from pathlib import Path

for path in sorted(Path("results").glob("*cspbi3*")):
    print(path)
PY

exit "$status"
