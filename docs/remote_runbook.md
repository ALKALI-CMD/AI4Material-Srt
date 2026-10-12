# Remote Runbook

Use this after Volcengine CLI login succeeds and a GPU server has been selected.

## 1. Upload and Unpack

Upload the latest `dist/atomlens_phase1_*.zip` to a working directory on the GPU server, then unpack it.

From Windows, once the server IP and SSH user are known:

```powershell
.\scripts\upload_phase1_package.ps1 -HostName <server-ip> -User root
```

Manual upload:

```bash
unzip atomlens_phase1_*.zip -d atomlens_phase1
cd atomlens_phase1
```

## 2. Preflight

Run the bootstrap first. It installs the local package, runs tests, regenerates harmonic smoke outputs, and performs GPU preflight when `nvidia-smi` exists.

```bash
bash scripts/bootstrap_remote.sh
```

You can also run the GPU preflight by itself before installing model packages.

```bash
bash scripts/check_remote_gpu.sh
```

Expected signals:

- `nvidia-smi` is available.
- PyTorch imports successfully after environment setup.
- `CUDA_VISIBLE_DEVICES` exposes at most two GPUs per process.

## 3. Local Smoke on Server

The bootstrap already runs this block. Use these commands manually only when debugging.

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
atomlens smoke --calculator harmonic --steps 65
```

## 4. Model Dependencies

```bash
python -m pip install -e ".[dev,models,viz]"
```

This project currently pins `torch<2.14` to avoid CUDA 13 wheels on the known Volcengine server, whose driver reports CUDA 12.9.

If dependency resolution fails, capture the full log in `logs/install.log` and retry in a clean virtual environment.

## 5. Parallel Phase 1 Run

Use one GPU per child process by default.

```bash
bash scripts/run_remote_parallel.sh 0 1
```

Outputs are written to `results/` and logs to `logs/`.

## 6. Completion Check

Important outputs:

- `results/mace-mp_cspbi3_validation.json`
- `results/chgnet_cspbi3_validation.json`
- `results/mace-mp_vs_chgnet_cspbi3_comparison.json`
- `results/mace-mp_cspbi3_report.md`
- `results/chgnet_cspbi3_report.md`

Validation files should contain `"passed": true` before interpreting the attribution patterns.

Archive results for download:

```bash
bash scripts/archive_remote_results.sh
```
