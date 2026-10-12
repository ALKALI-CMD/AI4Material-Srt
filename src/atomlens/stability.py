from __future__ import annotations

import numpy as np
from ase import Atoms

from atomlens.metrics import spearman_correlation, topk_jaccard
from atomlens.structures import add_thermal_noise


def stability_report(
    atoms: Atoms,
    attributor,
    n_noises: int = 5,
    sigma: float = 0.03,
    seed: int = 0,
    topk: int = 10,
) -> dict:
    """稳定性检验(研究计划第六节):给结构加一点热扰动,归因排序变不变。

    attributor 是一个函数:输入 atoms,返回 (keys, values)。
    - 原子级归因:keys = [0, 1, 2, ...] 原子编号;
    - 边级归因:keys = [(i, j), ...] 排好序的原子对。
    热扰动不改变原子顺序,近邻拓扑基本不变,所以 keys 可以对齐。

    对每次扰动计算:与无扰动归因的 Spearman 相关、前 topk 名的重合度
    (Jaccard)。数值越大越稳定。
    """

    base_keys, base_values = attributor(atoms)
    base_values = np.asarray(base_values, dtype=float)

    spearman_scores = []
    jaccard_scores = []
    overlap_ratios = []
    for noise_index in range(n_noises):
        noisy = add_thermal_noise(atoms, sigma=sigma, seed=seed + noise_index)
        keys, values = attributor(noisy)
        values = np.asarray(values, dtype=float)

        key_to_value = dict(zip(keys, values, strict=True))
        common = [k for k in base_keys if k in key_to_value]
        overlap_ratios.append(len(common) / len(base_keys))
        aligned_base = np.array([dict(zip(base_keys, base_values, strict=True))[k] for k in common])
        aligned_noisy = np.array([key_to_value[k] for k in common])

        spearman_scores.append(spearman_correlation(aligned_base, aligned_noisy))
        jaccard_scores.append(topk_jaccard(aligned_base, aligned_noisy, k=topk))

    return {
        "sigma": sigma,
        "n_noises": n_noises,
        "topk": topk,
        "spearman_per_noise": spearman_scores,
        "mean_spearman": float(np.mean(spearman_scores)),
        "jaccard_per_noise": jaccard_scores,
        "mean_jaccard": float(np.mean(jaccard_scores)),
        "key_overlap_per_noise": overlap_ratios,
    }
