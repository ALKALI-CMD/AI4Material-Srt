# 第一期实验记录

第一期完成了 CsPbI3 结构生成、两模型路径归因，以及 MACE 边门控的消融、忠实性、旋转和扰动检查。已有结果保留在本目录；数值解释及整个项目的完成情况见[实验成果报告](../../docs/实验成果报告.md)。

## 主要结果

| 项目 | 结果 | 数据文件 |
|---|---|---|
| MACE 倾斜路径能量差 | −0.588005 eV，积分残差 3.88×10⁻⁵ eV | `path_ig_mace_tilt.json` |
| CHGNet 倾斜路径能量差 | −0.793028 eV，积分残差 5.02×10⁻⁵ eV | `path_ig_chgnet_tilt.json` |
| MACE 边归因与留一消融排序 | Spearman 0.9867 | `gating_vacancy.json` |
| MACE 归因旋转检查 | 最大绝对差 8.88×10⁻¹⁶ | `gating_vacancy.json` |
| MACE 边归因扰动检查 | 平均 Spearman 0.9533，Top-10 Jaccard 0.6457 | `stability_edges_mace.json` |
| MACE 关闭前 10 条边 | 能量变化约 8.25 eV，随机均值约 0.63 eV | `faithfulness_vacancy.json` |

路径能量差均为整个 40 原子超胞的数值，不是每原胞数值。路径归因反映指定坐标路径，不能据此证明实际相稳定性。

原子能差分结果见 `vacancy_differential.json`，倾斜序列见 `tilt_series.json/png`，模型能量与力的对称性检查见 `invariance_*.json`。通道范数输出见 `channels_tilt.json/png`，通道切片尚待复核，暂不作为机制结论。

绝对边归因的高排名边没有更靠近空位，因此第二期改用完美与缺陷结构的差分。早期按原子数缩放所得的“形成能代理”不具有明确化学势口径，后续应读第二期固定 I2 参考的结果。

## 复现

环境见 `manifest.json`。在安装模型和绘图依赖后，于仓库根目录运行：

```bash
bash scripts/run_case1.sh
python scripts/analyze_vacancy.py
```

脚本会覆盖同名输出。CHGNet 门控是在第二期加入的，第一期未包含其归因旋转检查。
