#!/usr/bin/env bash
set -euo pipefail

echo "python: $(command -v python)"
python --version

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found; this host may not expose NVIDIA GPUs." >&2
  exit 2
fi

nvidia-smi

python - <<'PY'
try:
    import torch
except ImportError:
    print("torch: not installed")
else:
    print(f"torch: {torch.__version__}")
    print(f"cuda_available: {torch.cuda.is_available()}")
    print(f"cuda_device_count: {torch.cuda.device_count()}")
    for index in range(torch.cuda.device_count()):
        print(f"cuda_device_{index}: {torch.cuda.get_device_name(index)}")
PY

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  count=$(python - <<'PY'
import os
visible = [item for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if item.strip()]
print(len(visible))
PY
)
  if (( count > 2 )); then
    echo "ERROR: CUDA_VISIBLE_DEVICES exposes ${count} GPUs; limit is 2 per process." >&2
    exit 2
  fi
fi
