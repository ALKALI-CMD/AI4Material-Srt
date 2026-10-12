# Atomlens Phase 1 Artifact Index

This index lists the current startup package for the SRT project.

## Code

- `src/atomlens/attribution.py`: coordinate-path force-integration attribution.
- `src/atomlens/models.py`: MACE-MP and CHGNet ASE calculator adapters.
- `src/atomlens/validation.py`: pass/fail checks for attribution and convergence outputs.
- `src/atomlens/compare.py`: model-to-model attribution comparison.
- `src/atomlens/export.py`: extxyz export with `atomlens_attribution` atom property.
- `src/atomlens/plotting.py`: PNG plots for attribution, convergence, and model comparison.
- `src/atomlens/reporting.py`: Markdown report generation.

## Local Evidence

- `results/phase1_attribute_harmonic.json`: local attribution smoke result.
- `results/phase1_convergence_harmonic.json`: quadrature convergence result.
- `results/phase1_validation_harmonic.json`: machine-readable validation result.
- `results/phase1_report.md`: readable report draft.
- `results/phase1_attribute_harmonic.extxyz`: structure file with attribution values.
- `results/phase1_compare_self.json`: comparison command smoke result.

## Remote Execution

- `scripts/run_remote_model_smoke.sh`: guarded single-model GPU run.
- `scripts/check_remote_gpu.sh`: remote CUDA and GPU visibility preflight.
- `scripts/bootstrap_remote.sh`: server-side setup and harmonic smoke bootstrap.
- `scripts/upload_phase1_package.ps1`: Windows-side SSH upload and bootstrap helper.
- `scripts/archive_remote_results.sh`: remote results/logs archive helper.
- `scripts/run_remote_parallel.sh`: parallel MACE-MP and CHGNet run, one visible GPU per child process.
- `configs/phase1_remote_matrix.yaml`: expected remote job matrix and output list.
- `docs/remote_runbook.md`: upload, setup, preflight, run, and validation checklist.

## Readiness

- `scripts/check_phase1_ready.ps1`: local startup-package self-check.
- `results/phase1_readiness.json`: latest self-check output.
