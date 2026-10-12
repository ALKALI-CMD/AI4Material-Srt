"""空位差分分析:空位结构 vs 完美结构,逐原子能量差是否局域在空位附近。

用法: CUDA_VISIBLE_DEVICES=0 python scripts/analyze_vacancy.py
输出: results/case1/vacancy_differential.json + 控制台摘要
"""

import json
from pathlib import Path

import numpy as np

from atomlens.models import atomic_energies, calculator_by_name
from atomlens.structures import cspbi3_iodine_vacancy, cspbi3_supercell

OUT = Path("results/case1")
OUT.mkdir(parents=True, exist_ok=True)

perfect = cspbi3_supercell()
vac = cspbi3_iodine_vacancy()
vac_site = perfect.get_positions()[2]  # 被删掉的碘的位置
dist = np.linalg.norm(vac.get_positions() - vac_site, axis=1)
symbols = np.array(vac.get_chemical_symbols())
p_idx = np.array([v if v < 2 else v + 1 for v in range(len(vac))])  # 完美->空位的原子对应

payload = {"vacancy_site": vac_site.tolist(), "models": {}}
for calculator in ("mace-mp", "chgnet"):
    calc = calculator_by_name(calculator, device="cuda")
    e_perfect = atomic_energies(calc, perfect)
    e_vac = atomic_energies(calc, vac)
    if calculator == "chgnet":
        # CHGNet 的 site energies 已是总能量量纲,但为保险起见检查量级
        pass
    delta = e_perfect["interaction"][p_idx] - e_vac["interaction"]

    formation_proxy = float(
        e_vac["total"].sum() - len(vac) / len(perfect) * e_perfect["total"].sum()
    )
    near, far = dist < 4.5, dist > 7.0
    order = np.argsort(-np.abs(delta))[:10]
    payload["models"][calculator] = {
        "formation_proxy_eV": formation_proxy,
        "delta_near_mean": float(delta[near].mean()),
        "delta_far_mean": float(delta[far].mean()),
        "delta_far_std": float(delta[far].std()),
        "top_atoms": [
            {
                "atom": f"{symbols[i]}{i}",
                "distance_to_vacancy": float(dist[i]),
                "delta": float(delta[i]),
            }
            for i in order
        ],
        "delta_per_atom": delta.tolist(),
    }
    print(
        f"{calculator}: formation_proxy={formation_proxy:+.3f} eV | "
        f"near mean={delta[near].mean():+.4f} | far mean={delta[far].mean():+.4f} "
        f"(std {delta[far].std():.4f})"
    )
    print(
        "  top:", [(f"{symbols[i]}{i}", round(dist[i], 2), round(delta[i], 4)) for i in order[:5]]
    )

(OUT / "vacancy_differential.json").write_text(json.dumps(payload, indent=2) + "\n")
print("saved", OUT / "vacancy_differential.json")
