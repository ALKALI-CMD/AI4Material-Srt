#!/bin/bash
# 第二期实验:差分边门控、形成能、内部指标-干预指标相关性,2x2x2 与 4x4x4。
# 用法: bash scripts/run_case2.sh   (仓库根目录,有 GPU)
set -e
cd "$(dirname "$0")/.."
OUT=results/case2
mkdir -p $OUT
PY=${PY:-/root/miniconda3/bin/atomlens}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

echo "=== 1/5 空位形成能(I2 化学势,两个模型 x 两种超胞) ==="
$PY formation --calculators mace-mp chgnet --n 2 4 --out $OUT/formation.json

echo "=== 2/5 差分边门控 2x2x2(MACE + CHGNet) ==="
$PY diff-gating --calculator mace-mp --n 2 --steps 32 --out $OUT/diff_gating_mace_n2.json
$PY diff-gating --calculator chgnet --n 2 --steps 24 --out $OUT/diff_gating_chgnet_n2.json

echo "=== 3/5 差分边门控 4x4x4(MACE) ==="
$PY diff-gating --calculator mace-mp --n 4 --steps 32 --out $OUT/diff_gating_mace_n4.json

echo "=== 4/5 内部指标 vs 干预指标(MACE) ==="
$PY correlation --n 2 --steps 32 --out $OUT/correlation_perfect_n2.json
$PY correlation --n 2 --vacancy --steps 32 --out $OUT/correlation_vacancy_n2.json
$PY correlation --n 4 --vacancy --steps 32 --out $OUT/correlation_vacancy_n4.json

echo "=== 5/5 CHGNet 门控基线(空位,带消融) ==="
$PY gating --calculator chgnet --structure cspbi3_iodine_vacancy --steps 24 --with-ablation --out $OUT/gating_chgnet_vacancy.json

echo "=== 画图 ==="
$PY plot-edge-attribution --input $OUT/diff_gating_mace_n2.json --out $OUT/diff_gating_mace_n2.png
$PY plot-edge-attribution --input $OUT/diff_gating_chgnet_n2.json --out $OUT/diff_gating_chgnet_n2.png
$PY plot-edge-attribution --input $OUT/diff_gating_mace_n4.json --out $OUT/diff_gating_mace_n4.png
$PY plot-faithfulness --input $OUT/diff_gating_mace_n2.json --out $OUT/diff_faithfulness_mace_n2.png
$PY plot-faithfulness --input $OUT/diff_gating_chgnet_n2.json --out $OUT/diff_faithfulness_chgnet_n2.png
$PY plot-faithfulness --input $OUT/diff_gating_mace_n4.json --out $OUT/diff_faithfulness_mace_n4.png
$PY plot-correlation --input $OUT/correlation_perfect_n2.json --out $OUT/correlation_perfect_n2.png
$PY plot-correlation --input $OUT/correlation_vacancy_n2.json --out $OUT/correlation_vacancy_n2.png
$PY plot-correlation --input $OUT/correlation_vacancy_n4.json --out $OUT/correlation_vacancy_n4.png
$PY plot-edge-attribution --input $OUT/gating_chgnet_vacancy.json --out $OUT/gating_chgnet_vacancy.png

echo "ALL DONE"
