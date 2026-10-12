from __future__ import annotations

import numpy as np
import torch
from ase import Atoms


# ---------------------------------------------------------------------------
# CHGNet 的边门控:和 gating.py 里 MACE 的做法一一对应。
#
# CHGNet 每个原子卷积层里的消息是
#     message = MLP(concat[中心原子, 键特征, 邻居原子]) * bond_weight
# 其中 bond_weight = bond_weights_ag(键的高斯基),按"无向键"排列,
# 每条键一个 64 维向量,逐通道乘进消息。所以把 bond_weights_ag/bg 的输出
# 按无向键乘上标量门 g,就精确实现了 message -> g * message。
#
# 无向键就是"原子对"(i, j, 镜像),和 MACE 的原子对定义一致,
# 天然满足"无向原子对共用门控"。
#
# 注意:这套门控管的是"两体消息通道"。三体(角度)通道如果在图里存在,
# 其键更新同样会被对应键的门缩放;我们测试的 CsPbI3 结构里三体图是空的,
# 所以实际只有两体路径。
# ---------------------------------------------------------------------------


class _ChgnetGatedModule(torch.nn.Module):
    """包一层:先算原来的 bond_weights 模块,再把输出按无向键乘 g。"""

    def __init__(self, inner: torch.nn.Module, holder: "ChgnetEdgeGater"):
        super().__init__()
        self.inner = inner
        self.holder = holder

    def forward(self, *args, **kwargs):
        out = self.inner(*args, **kwargs)
        gate = self.holder.current_bond_gate
        if gate is None:
            return out
        return out * gate.unsqueeze(-1)


class ChgnetEdgeGater:
    """把 CHGNet 的 bond_weights_ag/bg 模块包上门控,用完记得 remove() 还原。"""

    def __init__(self, calc):
        self.calc = calc
        self.model = calc.model
        self.current_bond_gate: torch.Tensor | None = None
        self._wrapped: list[tuple[torch.nn.Module, str, torch.nn.Module]] = []

    def install(self) -> None:
        for name in ("bond_weights_ag", "bond_weights_bg"):
            original = getattr(self.model, name)
            setattr(self.model, name, _ChgnetGatedModule(original, self))
            self._wrapped.append((self.model, name, original))

    def remove(self) -> None:
        for module, name, original in self._wrapped:
            setattr(module, name, original)
        self._wrapped = []

    def set_bond_gate(self, gate_per_bond: torch.Tensor | None) -> None:
        self.current_bond_gate = gate_per_bond


def _build_graph(calc, atoms: Atoms):
    """把 ase 结构转成 CHGNet 的 CrystalGraph 并搬上计算设备。"""

    from pymatgen.io.ase import AseAtomsAdaptor

    structure = AseAtomsAdaptor.get_structure(atoms)
    return calc.model.graph_converter(structure).to(calc.device)


def chgnet_bond_info(calc, atoms: Atoms) -> dict:
    """每个无向键的身份:原子对 (i, j)、镜像位移、距离。

    返回:
    - keys: 每个无向键的规范键 (min(i,j), max(i,j), 镜像),用于跨结构对齐
    - idx_i, idx_j, distances: 每个无向键的原子编号和距离
    """

    graph = _build_graph(calc, atoms)
    # 注意:做过一次反向传播后图里的张量会带 grad,统一 detach 再转 numpy
    atom_graph = graph.atom_graph.detach().cpu().numpy()
    undirected2directed = graph.undirected2directed.detach().cpu().numpy()
    images = graph.neighbor_image.detach().cpu().numpy().astype(int)
    frac = graph.atom_frac_coord.detach().cpu().numpy()
    lattice = graph.lattice.detach().cpu().numpy()
    cart = frac @ lattice

    n_bonds = len(undirected2directed)
    idx_i = np.zeros(n_bonds, dtype=int)
    idx_j = np.zeros(n_bonds, dtype=int)
    distances = np.zeros(n_bonds)
    keys: list[tuple] = []
    for u in range(n_bonds):
        d = int(undirected2directed[u])
        i, j = int(atom_graph[d, 0]), int(atom_graph[d, 1])
        image = tuple(int(round(v)) for v in images[d])
        vec = cart[j] + np.array(image) @ lattice - cart[i]
        distances[u] = float(np.linalg.norm(vec))
        # 规范键:让 (i, j, 镜像) 有唯一写法,正反两个方向映射到同一个键
        if i > j or (i == j and image < (0, 0, 0)):
            i, j, image = j, i, tuple(-v for v in image)
        idx_i[u], idx_j[u] = i, j
        keys.append((i, j, image))
    return {"keys": keys, "idx_i": idx_i, "idx_j": idx_j, "distances": distances}


def chgnet_gated_energy(calc, atoms: Atoms, gater: ChgnetEdgeGater, gate_per_bond=None) -> float:
    """带门控地算一次总能量。gate_per_bond 为 None 表示全部打开(正常模型)。"""

    graph = _build_graph(calc, atoms)
    if gate_per_bond is None:
        gater.set_bond_gate(None)
    else:
        g = torch.as_tensor(np.asarray(gate_per_bond, dtype=np.float32), device=calc.device)
        gater.set_bond_gate(g)
    energy = calc.model([graph], task="e")["e"]
    return float(energy.detach().cpu().item())


def chgnet_edge_gate_integrated_gradients(calc, atoms: Atoms, n_steps: int = 24) -> dict:
    """CHGNet 的边门控积分梯度归因,逻辑和 MACE 版完全一样。

    把每个无向键的门从 0 慢慢开到 1,沿路记录 dE/dg 并积分。
    完备性:所有键的归因之和 ≈ E(门全开) - E(门全关)。
    """

    graph = _build_graph(calc, atoms)
    n_bonds = graph.undirected2directed.shape[0]
    n_atoms = int(graph.atomic_number.shape[0])
    info = chgnet_bond_info(calc, atoms)

    gater = ChgnetEdgeGater(calc)
    gater.install()
    try:
        ones = torch.ones(n_bonds, device=calc.device)
        zeros = torch.zeros(n_bonds, device=calc.device)
        gater.set_bond_gate(ones)
        energy_full = float(calc.model([graph], task="e")["e"].detach().cpu().item())
        gater.set_bond_gate(zeros)
        energy_closed = float(calc.model([graph], task="e")["e"].detach().cpu().item())

        alphas = np.linspace(0.0, 1.0, n_steps)
        grads = np.zeros((n_steps, n_bonds))
        for step, alpha in enumerate(alphas):
            g = torch.full((n_bonds,), float(alpha), device=calc.device, requires_grad=True)
            gater.set_bond_gate(g)
            energy = calc.model([graph], task="e")["e"].sum()
            energy.backward()
            grads[step] = g.grad.detach().cpu().numpy()

        trapezoid = getattr(np, "trapezoid", None) or np.trapz
        attribution = trapezoid(grads, x=alphas, axis=0)
    finally:
        gater.remove()

    # 注意:CHGNet 的 forward 直接返回"每原子"能量(is_intensive),
    # 这里统一乘回原子数,让输出和 MACE 一样是总能量口径(eV)。
    return {
        "attribution": attribution * n_atoms,
        "n_bonds": int(n_bonds),
        "energy_full": energy_full * n_atoms,
        "energy_closed": energy_closed * n_atoms,
        "completeness_residual": float(attribution.sum() - (energy_full - energy_closed)) * n_atoms,
        "keys": info["keys"],
        "idx_i": info["idx_i"],
        "idx_j": info["idx_j"],
        "distances": info["distances"],
        "bond_graph_empty": bool(len(graph.bond_graph) == 0),
    }


def chgnet_edge_ablation_importance(calc, atoms: Atoms) -> dict:
    """留一法对照:每次只关一个无向键,看能量变化多少。"""

    graph = _build_graph(calc, atoms)
    n_bonds = graph.undirected2directed.shape[0]
    info = chgnet_bond_info(calc, atoms)

    gater = ChgnetEdgeGater(calc)
    gater.install()
    try:
        energy_full = chgnet_gated_energy(calc, atoms, gater, None)
        importance = np.zeros(n_bonds)
        ones = np.ones(n_bonds)
        for bond in range(n_bonds):
            gate = ones.copy()
            gate[bond] = 0.0
            importance[bond] = energy_full - chgnet_gated_energy(calc, atoms, gater, gate)
    finally:
        gater.remove()
    return {"importance": importance, "energy_full": energy_full, "keys": info["keys"]}


# ---------------------------------------------------------------------------
# CHGNet 的差分归因:和 differential.py 里 MACE 的做法一致,
# 只是底层换成 CHGNet 的键列表。键编号跨结构对齐用同一张
# "空位 -> 完美" 的原子映射表。
# ---------------------------------------------------------------------------


def chgnet_gates_from_keys(calc, atoms: Atoms, keyed_gate: dict[tuple, float]) -> np.ndarray:
    """把"按规范键给的门控值"展开成当前结构按无向键排列的门控数组。"""

    info = chgnet_bond_info(calc, atoms)
    gate = np.ones(len(info["keys"]))
    for index, key in enumerate(info["keys"]):
        if key in keyed_gate:
            gate[index] = keyed_gate[key]
    return gate


def chgnet_differential_edge_attribution(calc, n_supercell: int = 2, n_steps: int = 24) -> dict:
    """空位 vs 完美的差分边归因(CHGNet 版)。"""

    from atomlens.differential import vacancy_to_perfect_index
    from atomlens.structures import cspbi3_iodine_vacancy, cspbi3_supercell

    perfect = cspbi3_supercell(n=n_supercell)
    vac = cspbi3_iodine_vacancy(n=n_supercell)
    index_map = vacancy_to_perfect_index(len(vac))

    result_perf = chgnet_edge_gate_integrated_gradients(calc, perfect, n_steps=n_steps)
    result_vac = chgnet_edge_gate_integrated_gradients(calc, vac, n_steps=n_steps)

    attr_perf = dict(zip(result_perf["keys"], result_perf["attribution"], strict=True))
    rows = []
    for p, key in enumerate(result_vac["keys"]):
        key_in_perfect = (int(index_map[key[0]]), int(index_map[key[1]]), key[2])
        delta = float(result_vac["attribution"][p] - attr_perf.get(key_in_perfect, np.nan))
        rows.append(
            {
                "i": int(result_vac["idx_i"][p]),
                "j": int(result_vac["idx_j"][p]),
                "key": key,
                "distance": float(result_vac["distances"][p]),
                "value": delta,
            }
        )
    rows.sort(key=lambda row: abs(row["value"]), reverse=True)
    return {
        "n_bonds_vac": result_vac["n_bonds"],
        "n_bonds_perf": result_perf["n_bonds"],
        "energy_full_vac": result_vac["energy_full"],
        "energy_full_perf": result_perf["energy_full"],
        "completeness_residual_vac": result_vac["completeness_residual"],
        "completeness_residual_perf": result_perf["completeness_residual"],
        "pairs": rows,
    }


def chgnet_differential_faithfulness(
    calc,
    n_supercell: int = 2,
    ranking_rows: list[dict] | None = None,
    ks: list[int] | None = None,
    n_random: int = 5,
    seed: int = 0,
) -> dict:
    """差分归因忠实性(CHGNet 版):关掉同一批键,看 |ΔΔE| 是否 top-k 最大。"""

    from atomlens.differential import vacancy_to_perfect_index
    from atomlens.structures import cspbi3_iodine_vacancy, cspbi3_supercell

    perfect = cspbi3_supercell(n=n_supercell)
    vac = cspbi3_iodine_vacancy(n=n_supercell)
    index_map = vacancy_to_perfect_index(len(vac))
    if ks is None:
        ks = [5, 10, 20, 40, 80]
    if ranking_rows is None:
        ranking_rows = chgnet_differential_edge_attribution(calc, n_supercell=n_supercell)["pairs"]

    gater = ChgnetEdgeGater(calc)
    gater.install()
    rng = np.random.default_rng(seed)
    try:
        e_vac = chgnet_gated_energy(calc, vac, gater, None)
        e_perf = chgnet_gated_energy(calc, perfect, gater, None)

        def differential_change(keyed_gate: dict[tuple, float]) -> float:
            gate_vac = chgnet_gates_from_keys(calc, vac, keyed_gate)
            gate_perf_keys = {
                (int(index_map[k[0]]), int(index_map[k[1]]), k[2]): v for k, v in keyed_gate.items()
            }
            gate_perf = chgnet_gates_from_keys(calc, perfect, gate_perf_keys)
            e_vac_g = chgnet_gated_energy(calc, vac, gater, gate_vac)
            e_perf_g = chgnet_gated_energy(calc, perfect, gater, gate_perf)
            return float(abs((e_vac_g - e_vac) - (e_perf_g - e_perf)))

        ranked_keys = [tuple(row["key"]) for row in ranking_rows]
        n_bonds = len(ranked_keys)
        rows = []
        for k in ks:
            k = min(k, n_bonds)
            delta_top = differential_change({key: 0.0 for key in ranked_keys[:k]})
            delta_bottom = differential_change({key: 0.0 for key in ranked_keys[n_bonds - k :]})
            random_deltas = []
            for _ in range(n_random):
                picked = rng.choice(n_bonds, size=k, replace=False)
                random_deltas.append(differential_change({ranked_keys[i]: 0.0 for i in picked}))
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
