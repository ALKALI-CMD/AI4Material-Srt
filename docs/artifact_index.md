# 实验成果索引

项目状态和数值解释见[实验成果报告](实验成果报告.md)，操作说明见[中文指南](中文指南.md)。

| 目录 | 成果 | 记录 |
|---|---|---|
| `results/case1/` | 路径归因、MACE 边门控、对称性与扰动检查 | [第一期](../results/case1/RESULTS.md) |
| `results/case2/` | CHGNet 门控、共同边差分、固定参考形成能、相关性 | [第二期](../results/case2/RESULTS.md) |
| `results/case3/` | 两组通道门控探索输出，待复核 | [第三期](../results/case3/RESULTS.md) |
| `results/phase1_*` | 早期 harmonic 计算器的功能验证与打包自检 | [早期报告](../results/phase1_report.md) |

`src/atomlens/` 保存实现，`tests/` 保存基础测试，`scripts/run_case1.sh` 和 `scripts/run_case2.sh` 保存前两期运行步骤。第三期尚无一键运行脚本。包版本记录在 `results/case1/manifest.json`，模型权重校验值尚未记录。

旧的 phase1 readiness 输出只反映当时的启动包检查，不是整个 SRT 项目的验收结果。
