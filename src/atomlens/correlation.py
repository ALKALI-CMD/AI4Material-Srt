from __future__ import annotations

import numpy as np
from ase import Atoms

from atomlens.channels import mace_message_channel_norms
from atomlens.gating import (
    _batch_and_pair_info,
    canonical_pair_keys,
    edge_gate_integrated_gradients,
    pair_distances,
)
from atomlens.metrics import spearman_correlation


# ---------------------------------------------------------------------------
# 内部指标 vs 干预指标(主论文核心问题):
#
# "网络内部哪条边的消息数值大" 和 "干预实验说哪条边重要" 是不是一回事?
# 对每条边(原子对)计算:
#   - 干预指标:边门控积分梯度归因 |A_ij|(因果干预,主指标)
#   - 内部指标:每层消息范数、各角动量 l 通道范数(表征诊断,辅助指标)
#   - 结构对照:键长(模型不学键长本身,但消息通常随距离衰减)
# 然后算逐边 Spearman 相关。相关高,说明内部指标可以代替昂贵的干预实验;
# 相关低,说明"画热力图"式的内部解释会误导。
# ---------------------------------------------------------------------------


def _edge_to_pair_mean(per_edge_values: np.ndarray, pair_ids: np.ndarray, n_pairs: int) -> np.ndarray:
    """每条有向边的量 -> 每个原子对的平均(正反两条边取平均)。"""

    per_pair = np.zeros(n_pairs)
    counts = np.zeros(n_pairs)
    np.add.at(per_pair, pair_ids, per_edge_values)
    np.add.at(counts, pair_ids, 1.0)
    return per_pair / np.clip(counts, 1, None)


def correlation_report(calc, atoms: Atoms, n_steps: int = 32) -> dict:
    """在同一个结构上跑一次边门控归因和一次消息通道提取,然后逐对算相关。"""

    gate_result = edge_gate_integrated_gradients(calc, atoms, n_steps=n_steps)
    channel_result = mace_message_channel_norms(calc, atoms)

    data, pair_ids = _batch_and_pair_info(calc, atoms)
    n_pairs = gate_result["n_pairs"]
    keys = canonical_pair_keys(pair_ids, data["edge_index"].cpu().numpy(), data["unit_shifts"].cpu().numpy())
    idx_i, idx_j, distances = pair_distances(calc, atoms, pair_ids)

    attr_abs = np.abs(gate_result["attribution"])
    rows = []

    # 结构对照:键长(负相关越强说明"归因≈只看距离")
    rows.append({"layer": "structure", "channel": "pair_distance", "spearman_vs_abs_attr": spearman_correlation(attr_abs, -distances)})

    for layer_index, layer in enumerate(channel_result["layers"]):
        overall_pair = _edge_to_pair_mean(layer["edge_overall_norms"], pair_ids, n_pairs)
        rows.append(
            {
                "layer": layer_index,
                "channel": "message_norm",
                "spearman_vs_abs_attr": spearman_correlation(attr_abs, overall_pair),
            }
        )
        for l, values in layer["edge_l_norms"].items():
            channel_pair = _edge_to_pair_mean(values, pair_ids, n_pairs)
            rows.append(
                {
                    "layer": layer_index,
                    "channel": f"l{l}",
                    "spearman_vs_abs_attr": spearman_correlation(attr_abs, channel_pair),
                }
            )

    return {
        "n_pairs": n_pairs,
        "n_edges": gate_result["n_edges"],
        "completeness_residual": gate_result["completeness_residual"],
        "l_max": channel_result["l_max"],
        "rows": rows,
        "pairs": [
            {
                "i": int(idx_i[p]),
                "j": int(idx_j[p]),
                "key": list(keys[p]),
                "distance": float(distances[p]),
                "abs_attribution": float(attr_abs[p]),
            }
            for p in range(n_pairs)
        ],
    }
