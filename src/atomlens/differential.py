from __future__ import annotations

import numpy as np
from ase import Atoms

from atomlens.gating import (
    EdgeGater,
    _batch_and_pair_info,
    canonical_pair_keys,
    edge_gate_integrated_gradients,
    mace_gated_energy,
    pair_distances,
)
from atomlens.structures import cspbi3_iodine_vacancy, cspbi3_supercell


# ---------------------------------------------------------------------------
# 差分边门控:回答"空位改变了哪些键的重要性"。
#
# 做法:对完美超胞和空位超胞各跑一次边门控积分梯度,然后按"规范键"
# (原子对 + 单胞镜像)把两次归因对齐相减:
#     ΔA(键) = A(键 | 空位结构) − A(键 | 完美结构)
# 数学上这就是"对能量差 ΔE 做门控积分"的结果(积分是线性的,
# 两条路径相同,梯度相减等于差分的梯度)。
#
# 注意原子编号:空位删掉的是完美超胞的第 2 号原子(第一个 I),
# 所以空位里的原子 v 对应完美里的原子 v(v<2)或 v+1(v≥2)。
# ---------------------------------------------------------------------------

REMOVED_PERFECT_INDEX = 2  # cspbi3_iodine_vacancy 删掉的那个原子在完美超胞里的编号


def vacancy_to_perfect_index(n_vacancy_atoms: int) -> np.ndarray:
    """空位原子编号 -> 完美超胞原子编号 的对应表。"""

    return np.array(
        [v if v < REMOVED_PERFECT_INDEX else v + 1 for v in range(n_vacancy_atoms)],
        dtype=int,
    )


def _mace_pair_keys(calc, atoms: Atoms, pair_ids: np.ndarray) -> list[tuple]:
    data, _ = _batch_and_pair_info(calc, atoms)
    edge_index, unit_shifts = data["edge_index"].cpu().numpy(), data["unit_shifts"].cpu().numpy()
    return canonical_pair_keys(pair_ids, edge_index, unit_shifts)


def gates_from_keys(calc, atoms: Atoms, keyed_gate: dict[tuple, float]) -> np.ndarray:
    """把"按规范键给的门控值"展开成当前结构的逐原子对门控数组。

    结构里没有出现在 keyed_gate 里的键保持 1(完全打开)。
    """

    data, pair_ids = _batch_and_pair_info(calc, atoms)
    keys = _mace_pair_keys(calc, atoms, pair_ids)
    gate = np.ones(len(keys))
    for index, key in enumerate(keys):
        if key in keyed_gate:
            gate[index] = keyed_gate[key]
    return gate


def differential_edge_attribution(calc, n_supercell: int = 2, n_steps: int = 32) -> dict:
    """空位结构 vs 完美结构的差分边归因(MACE)。"""

    perfect = cspbi3_supercell(n=n_supercell)
    vac = cspbi3_iodine_vacancy(n=n_supercell)
    index_map = vacancy_to_perfect_index(len(vac))

    result_perf = edge_gate_integrated_gradients(calc, perfect, n_steps=n_steps)
    result_vac = edge_gate_integrated_gradients(calc, vac, n_steps=n_steps)

    keys_perf = _mace_pair_keys(calc, perfect, result_perf["pair_ids"])
    keys_vac = _mace_pair_keys(calc, vac, result_vac["pair_ids"])
    attr_perf = dict(zip(keys_perf, result_perf["attribution"], strict=True))

    idx_i, idx_j, distances = pair_distances(calc, vac, result_vac["pair_ids"])
    rows = []
    for p, key in enumerate(keys_vac):
        # 把空位里的键翻译回完美结构的编号
        key_in_perfect = (int(index_map[key[0]]), int(index_map[key[1]]), key[2])
        delta = float(result_vac["attribution"][p] - attr_perf.get(key_in_perfect, np.nan))
        rows.append(
            {
                "i": int(idx_i[p]),
                "j": int(idx_j[p]),
                "key": key,
                "distance": float(distances[p]),
                "value": delta,
            }
        )
    rows.sort(key=lambda row: abs(row["value"]), reverse=True)
    return {
        "n_pairs_vac": result_vac["n_pairs"],
        "n_pairs_perf": result_perf["n_pairs"],
        "energy_full_vac": result_vac["energy_full"],
        "energy_full_perf": result_perf["energy_full"],
        "completeness_residual_vac": result_vac["completeness_residual"],
        "completeness_residual_perf": result_perf["completeness_residual"],
        "pairs": rows,
    }


def differential_faithfulness(
    calc,
    n_supercell: int = 2,
    ranking_rows: list[dict] | None = None,
    ks: list[int] | None = None,
    n_random: int = 5,
    seed: int = 0,
) -> dict:
    """差分归因的忠实性检验。

    按差分归因排名关边(两个结构里同时关同一批键),看
    |ΔΔE| = |[E空位(关边)-E空位] - [E完美(关边)-E完美]|
    是不是 top-k 最大。这检验的是"归因差值"本身的因果意义。
    """

    perfect = cspbi3_supercell(n=n_supercell)
    vac = cspbi3_iodine_vacancy(n=n_supercell)
    if ks is None:
        ks = [5, 10, 20, 40, 80]
    if ranking_rows is None:
        ranking_rows = differential_edge_attribution(calc, n_supercell=n_supercell)["pairs"]

    gater = EdgeGater(calc)
    gater.install()
    rng = np.random.default_rng(seed)
    try:
        e_vac = mace_gated_energy(calc, vac, gater, None)
        e_perf = mace_gated_energy(calc, perfect, gater, None)

        ranked_keys = [tuple(row["key"]) for row in ranking_rows]
        n_pairs_vac = len(ranked_keys)
        rows = []
        for k in ks:
            k = min(k, n_pairs_vac)
            chosen = ranked_keys[:k]
            gate_top = {key: 0.0 for key in chosen}

            delta_top = _differential_energy_change(calc, perfect, vac, gater, gate_top, e_perf, e_vac)

            bottom_keys = ranked_keys[n_pairs_vac - k :]
            delta_bottom = _differential_energy_change(
                calc, perfect, vac, gater, {key: 0.0 for key in bottom_keys}, e_perf, e_vac
            )

            random_deltas = []
            for _ in range(n_random):
                picked = rng.choice(n_pairs_vac, size=k, replace=False)
                random_gate = {ranked_keys[index]: 0.0 for index in picked}
                random_deltas.append(
                    _differential_energy_change(calc, perfect, vac, gater, random_gate, e_perf, e_vac)
                )

            rows.append(
                {
                    "k": k,
                    "delta_top": delta_top,
                    "delta_random": float(np.mean(random_deltas)),
                    "delta_random_std": float(np.std(random_deltas)),
                    "delta_bottom": delta_bottom,
                }
            )
    finally:
        gater.remove()

    return {"energy_vac": e_vac, "energy_perf": e_perf, "rows": rows}


def _differential_energy_change(calc, perfect, vac, gater, keyed_gate, e_perf, e_vac) -> float:
    index_map = vacancy_to_perfect_index(len(vac))
    gate_vac = gates_from_keys(calc, vac, keyed_gate)
    # 完美结构的原子编号整体错一位,键也要翻译回去
    gate_perf_keys = {
        (int(index_map[key[0]]), int(index_map[key[1]]), key[2]): value
        for key, value in keyed_gate.items()
    }
    gate_perf = gates_from_keys(calc, perfect, gate_perf_keys)
    e_vac_gated = mace_gated_energy(calc, vac, gater, gate_vac)
    e_perf_gated = mace_gated_energy(calc, perfect, gater, gate_perf)
    return float(abs((e_vac_gated - e_vac) - (e_perf_gated - e_perf)))


# ---------------------------------------------------------------------------
# 空位形成能:让"能量对比"有一个物理上说得通的口径。
#
# 标准定义(中性碘空位):
#     E_form = E(空位超胞) - E(完美超胞) + mu_I
# 其中 mu_I 是碘的化学势。碘富集极限取 mu_I = E(I2 分子)/2
# (空位翻出来的碘原子回到 I2 气体 reservoir)。
# 同时记录"按原子数缩放"的粗略代理,方便和第一期结果对照。
# ---------------------------------------------------------------------------


def iodine_molecule(a: float = 2.67, box: float = 12.0) -> Atoms:
    """放在大盒子里的 I2 分子(实验键长 2.67 埃)。"""

    return Atoms(
        "I2",
        positions=[[0.0, 0.0, 0.0], [a, 0.0, 0.0]],
        cell=[box, box, box],
        pbc=True,
    )


def formation_numbers(e_perf: float, e_vac: float, e_i2: float, n_perf: int, n_vac: int) -> dict:
    """形成能的纯算术部分,方便单独测试。"""

    mu_i = e_i2 / 2.0
    return {
        "mu_I_eV": mu_i,
        "formation_energy_I_rich_eV": e_vac - e_perf + mu_i,
        "formation_energy_crude_scaling_eV": e_vac - (n_vac / n_perf) * e_perf,
    }


def formation_energy(calc, n_supercell: int = 2) -> dict:
    """计算一个模型的碘空位形成能(MACE 或 CHGNet 都能用)。"""

    perfect = cspbi3_supercell(n=n_supercell)
    vac = cspbi3_iodine_vacancy(n=n_supercell)
    molecule = iodine_molecule()

    perfect.calc = calc
    vac.calc = calc
    molecule.calc = calc
    e_perf = float(perfect.get_potential_energy())
    e_vac = float(vac.get_potential_energy())
    e_i2 = float(molecule.get_potential_energy())

    return {
        "n_supercell": n_supercell,
        "n_atoms_perfect": len(perfect),
        "n_atoms_vacancy": len(vac),
        "energy_perfect_eV": e_perf,
        "energy_vacancy_eV": e_vac,
        "energy_I2_eV": e_i2,
        **formation_numbers(e_perf, e_vac, e_i2, len(perfect), len(vac)),
    }
