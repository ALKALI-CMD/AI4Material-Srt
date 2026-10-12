# Volcengine Phase 1 Notes

Use this after CLI login succeeds and the target GPU server is identified.

## GPU Policy

- `CUDA_VISIBLE_DEVICES` must expose no more than two GPUs per process.
- Default inference jobs should use one GPU.
- Parallel runs are allowed only when each process respects the two-GPU cap.

## First Remote Commands

```bash
bash scripts/bootstrap_remote.sh
bash scripts/check_remote_gpu.sh
python -m pip install -e ".[dev]"
python -m pytest -q
atomlens smoke --calculator harmonic --steps 65
atomlens manifest --out results/manifest.json
```

After MACE/CHGNet installation:

```bash
python -m pip install ".[models,viz]"
```

Record package versions, visible GPUs, model checkpoint names, and output paths in `results/`.

```bash
CUDA_VISIBLE_DEVICES=0 atomlens model-smoke --calculator mace-mp --structure diamond_si --device cuda --out results/mace_mp_diamond_si.json
CUDA_VISIBLE_DEVICES=1 atomlens model-smoke --calculator chgnet --structure cubic_cspbi3 --device cuda --out results/chgnet_cspbi3.json
CUDA_VISIBLE_DEVICES=0 atomlens attribute --calculator mace-mp --structure-pair cspbi3_iodine_displacement --steps 65 --device cuda --out results/mace_mp_cspbi3_path.json
CUDA_VISIBLE_DEVICES=0 atomlens convergence --calculator mace-mp --structure-pair cspbi3_iodine_displacement --steps 17 33 65 129 --device cuda --out results/mace_mp_cspbi3_convergence.json
CUDA_VISIBLE_DEVICES=0 atomlens validate-results --attribution results/mace_mp_cspbi3_path.json --convergence results/mace_mp_cspbi3_convergence.json --out results/mace_mp_cspbi3_validation.json
```

Or use the guarded helper:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_remote_model_smoke.sh mace-mp diamond_si cuda
CUDA_VISIBLE_DEVICES=1 bash scripts/run_remote_model_smoke.sh chgnet cubic_cspbi3 cuda
CUDA_VISIBLE_DEVICES=0,1 bash scripts/run_remote_model_smoke.sh mace-mp cubic_cspbi3 cuda
```

Run the two main calculators in parallel:

```bash
bash scripts/run_remote_parallel.sh 0 1
```

Each child process sees one GPU. Use different GPU ids only after checking the server with `nvidia-smi`.
If both jobs finish, the script also writes:

```text
results/mace-mp_vs_chgnet_cspbi3_comparison.json
results/mace-mp_vs_chgnet_cspbi3_comparison.png
```
