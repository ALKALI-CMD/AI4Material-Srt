#!/bin/bash
# 第一期全部实验:CsPbI3 八面体倾斜 + 碘空位,MACE-MP 与 CHGNet 对照。
# 用法: bash scripts/run_case1.sh   (需要在仓库根目录,且有 GPU)
set -e
cd "$(dirname "$0")/.."
OUT=results/case1
mkdir -p $OUT
PY=${PY:-/root/miniconda3/bin/atomlens}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

echo "=== 1/8 不变性检验 ==="
$PY invariance --calculator mace-mp --structure cubic_cspbi3 --out $OUT/invariance_mace.json
$PY invariance --calculator chgnet --structure cubic_cspbi3 --out $OUT/invariance_chgnet.json

echo "=== 2/8 倾斜系列基线(能量/力) ==="
$PY tilt-series --calculators mace-mp chgnet --angles 0 2 4 6 8 10 12 --out $OUT/tilt_series.json

echo "=== 3/8 原子能分解 ==="
$PY decompose --calculator mace-mp --structure cspbi3_iodine_vacancy --out $OUT/decompose_mace_vacancy.json
$PY decompose --calculator chgnet --structure cspbi3_iodine_vacancy --out $OUT/decompose_chgnet_vacancy.json
$PY decompose --calculator mace-mp --structure cspbi3_supercell --out $OUT/decompose_mace_perfect.json

echo "=== 4/8 边门控归因(MACE,积分梯度) ==="
$PY gating --structure cspbi3_iodine_vacancy --steps 32 --with-ablation --out $OUT/gating_vacancy.json
$PY gating --structure cspbi3_supercell --steps 32 --out $OUT/gating_perfect.json

echo "=== 5/8 忠实性与稳定性 ==="
$PY faithfulness --structure cspbi3_iodine_vacancy --ks 5 10 20 40 80 --n-random 5 --out $OUT/faithfulness_vacancy.json
$PY stability --calculator mace-mp --structure cspbi3_iodine_vacancy --what edges --n-noises 5 --out $OUT/stability_edges_mace.json
$PY stability --calculator mace-mp --structure cspbi3_iodine_vacancy --what atoms --n-noises 5 --out $OUT/stability_atoms_mace.json
$PY stability --calculator chgnet --structure cspbi3_iodine_vacancy --what atoms --n-noises 5 --out $OUT/stability_atoms_chgnet.json

echo "=== 6/8 通道分析(倾斜系列) ==="
$PY channels --angles 0 4 8 12 --out $OUT/channels_tilt.json

echo "=== 7/8 结构路径积分(方法 A)与收敛性 ==="
$PY attribute --calculator mace-mp --structure-pair cspbi3_tilt --steps 65 --out $OUT/path_ig_mace_tilt.json
$PY attribute --calculator chgnet --structure-pair cspbi3_tilt --steps 65 --out $OUT/path_ig_chgnet_tilt.json
$PY convergence --calculator mace-mp --structure-pair cspbi3_tilt --steps 17 33 65 129 --out $OUT/convergence_mace_tilt.json
$PY convergence --calculator chgnet --structure-pair cspbi3_tilt --steps 17 33 65 129 --out $OUT/convergence_chgnet_tilt.json

echo "=== 8/8 模型对照与环境清单 ==="
$PY compare-attributions --left $OUT/decompose_mace_vacancy.json --right $OUT/decompose_chgnet_vacancy.json --out $OUT/compare_decompose.json
$PY manifest --out $OUT/manifest.json

echo "=== 画图 ==="
$PY plot-tilt-series --input $OUT/tilt_series.json --out $OUT/tilt_series.png
$PY plot-attribution --input $OUT/decompose_mace_vacancy.json --out $OUT/decompose_mace_vacancy.png
$PY plot-attribution --input $OUT/decompose_chgnet_vacancy.json --out $OUT/decompose_chgnet_vacancy.png
$PY plot-comparison --input $OUT/compare_decompose.json --out $OUT/compare_decompose.png
$PY plot-edge-attribution --input $OUT/gating_vacancy.json --out $OUT/gating_vacancy.png
$PY plot-edge-attribution --input $OUT/gating_perfect.json --out $OUT/gating_perfect.png
$PY plot-faithfulness --input $OUT/faithfulness_vacancy.json --out $OUT/faithfulness_vacancy.png
$PY plot-stability --input $OUT/stability_edges_mace.json --out $OUT/stability_edges_mace.png
$PY plot-stability --input $OUT/stability_atoms_chgnet.json --out $OUT/stability_atoms_chgnet.png
$PY plot-channels --input $OUT/channels_tilt.json --out $OUT/channels_tilt.png
$PY plot-attribution --input $OUT/path_ig_mace_tilt.json --out $OUT/path_ig_mace_tilt.png
$PY plot-attribution --input $OUT/path_ig_chgnet_tilt.json --out $OUT/path_ig_chgnet_tilt.png
$PY plot-convergence --input $OUT/convergence_mace_tilt.json --out $OUT/convergence_mace_tilt.png

echo "ALL DONE"
