# atomlens

`atomlens` starts the SRT project from the prior proposal: interpretable analysis of pretrained atomistic machine-learning force fields.

## 中文阅读入口与项目状态

- [实验成果报告](docs/实验成果报告.md)：完成情况、主要数据、图表与尚未解决的问题。
- [中文使用说明](docs/中文指南.md)：方法解释和运行步骤。
- [第一期记录](results/case1/RESULTS.md)、[第二期记录](results/case2/RESULTS.md)、[第三期记录](results/case3/RESULTS.md)。

目前可用于阶段汇报，尚未完成全部研究验证。前两期已有主要实验输出；第三期通道门控仍在复核。CHGNet 忠实性的能量口径、MACE 通道切片、物理参照与模型权重记录需要补齐。下面列出已实现的功能，不表示每项研究结论均已验证。

## What is implemented (phase 1 + phase 2 + phase 3)

- **Structures**: cubic CsPbI3, n x n x n supercells (2x2x2 and 4x4x4 used),
  checkerboard octahedral tilting, one-iodine vacancy, reproducible thermal
  noise (`atomlens.structures`).
- **Models**: pretrained MACE-MP ("small") and CHGNet with package versions recorded;
  per-atom energy decomposition for both (`atomlens decompose`).
- **Method A — structure-path integrated attribution**: force integration along a
  physical path (octahedral tilt), for both models (`atomlens attribute`).
- **Method B — symmetry-preserving edge gating (core)**: per-edge gates on the
  interaction messages for **both** MACE (conv_tp output) and CHGNet
  (bond_weights), integrated-gradients attribution with a numerical completeness
  residual, leave-one-out ablation cross-check, top-k faithfulness curves
  against random/bottom controls (`atomlens gating`, `atomlens faithfulness`).
- **Differential gating**: vacancy minus perfect, aligned by canonical bond keys;
  the interventional check gates the same bonds in both structures and scores
  |dE_vac - dE_perf| (`atomlens diff-gating`).
- **Formation energy**: iodine vacancy formation energy with an explicit I2
  chemical potential, compared at two unrelaxed supercell sizes (2x2x2 vs 4x4x4)
  (`atomlens formation`, `scripts/analyze_vacancy.py`).
- **Method D — irreducible-channel diagnostics**: per-edge l=0..3 message channel
  norms and per-layer node-feature norms along the tilt series (`atomlens channels`).
- **Internal vs interventional correlation**: recorded MACE message-norm and
  channel statistics against edge attribution. Angular-channel slicing and
  interpretation remain under review; see the experiment report.
- **Evaluation**: translation/rotation/permutation/supercell invariance
  (`atomlens invariance`), rotation consistency of the edge attribution
  (machine-precision), stability under thermal noise via Spearman/Jaccard
  (`atomlens stability`).

Reproduce on one GPU:

```bash
bash scripts/run_case1.sh   # phase 1+2: baselines, gating, faithfulness, stability, channels
bash scripts/run_case2.sh   # phase 3: differential gating, formation energy, correlations
```

The first executable milestone is deliberately narrow:

- load reproducible structures through ASE;
- run pretrained MACE-MP or CHGNet calculators when available;
- compute coordinate-path integrated force attribution for fixed-composition comparisons;
- test completeness, translation, rotation, and atom-order sanity before interpreting model internals;
- keep every training or experiment process to at most two GPUs on the Volcengine server.

This repository does not claim that a neural-network message norm is a physical bond energy. The first target is model attribution: how much a node or local coordinate change affects a chosen model output under a stated path.

## Quick Start

```powershell
python -m pip install -e ".[dev]"
python -m pytest
atomlens smoke --calculator harmonic
atomlens manifest --out results/manifest.json
atomlens plot-attribution --input results/phase1_attribute_harmonic.json --out results/phase1_attribute_harmonic.png
atomlens plot-convergence --input results/phase1_convergence_harmonic.json --out results/phase1_convergence_harmonic.png
atomlens validate-results --attribution results/phase1_attribute_harmonic.json --convergence results/phase1_convergence_harmonic.json --out results/phase1_validation_harmonic.json
atomlens export-structure --attribution results/phase1_attribute_harmonic.json --structure-pair cspbi3_iodine_displacement --which end --out results/phase1_attribute_harmonic.extxyz
atomlens report --attribution results/phase1_attribute_harmonic.json --convergence results/phase1_convergence_harmonic.json --attribution-figure results/phase1_attribute_harmonic.png --convergence-figure results/phase1_convergence_harmonic.png --structure-file results/phase1_attribute_harmonic.extxyz --out results/phase1_report.md
atomlens compare-attributions --left results/phase1_attribute_harmonic.json --right results/phase1_attribute_harmonic.json --out results/phase1_compare_self.json
atomlens plot-comparison --input results/phase1_compare_self.json --out results/phase1_compare_self.png
```

Optional model installs:

```powershell
python -m pip install ".[models,viz]"
```

The project pins `torch<2.14` because the current Volcengine server reports a CUDA 12.9 driver; PyTorch wheels using CUDA 13 fail CUDA initialization there.

MACE/CHGNet runs should pin exact package versions and model checkpoints in `results/manifest.json` before comparing results.

GPU model smoke examples:

```bash
CUDA_VISIBLE_DEVICES=0 atomlens model-smoke --calculator mace-mp --structure diamond_si --device cuda --out results/mace_mp_diamond_si.json
CUDA_VISIBLE_DEVICES=1 atomlens model-smoke --calculator chgnet --structure cubic_cspbi3 --device cuda --out results/chgnet_cspbi3.json
CUDA_VISIBLE_DEVICES=0 atomlens attribute --calculator mace-mp --structure-pair cspbi3_iodine_displacement --steps 65 --device cuda --out results/mace_mp_cspbi3_path.json
CUDA_VISIBLE_DEVICES=0 atomlens convergence --calculator mace-mp --structure-pair cspbi3_iodine_displacement --steps 17 33 65 129 --device cuda --out results/mace_mp_cspbi3_convergence.json
CUDA_VISIBLE_DEVICES=0 atomlens validate-results --attribution results/mace_mp_cspbi3_path.json --convergence results/mace_mp_cspbi3_convergence.json --out results/mace_mp_cspbi3_validation.json
```

Parallel remote run:

```bash
bash scripts/bootstrap_remote.sh
bash scripts/check_remote_gpu.sh
bash scripts/run_remote_parallel.sh 0 1
```

This starts MACE-MP on GPU 0 and CHGNet on GPU 1. Each child process receives a single visible GPU through `CUDA_VISIBLE_DEVICES`.

Package the current phase for upload:

```powershell
.\scripts\check_phase1_ready.ps1
.\scripts\package_project.ps1
.\scripts\upload_phase1_package.ps1 -HostName <server-ip> -User root
```

## Project Layout

- `src/atomlens/`: reusable attribution and model adapter code.
- `tests/`: calculator-independent checks.
- `configs/`: experiment configs.
- `docs/`: project notes, literature support, and run protocol.
- `scripts/`: local and Volcengine helper scripts.
- `results/`: committed experiment data, figures, and phase records.
