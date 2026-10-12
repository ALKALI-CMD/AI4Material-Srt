from __future__ import annotations

import numpy as np
import torch
from ase import Atoms


# ---------------------------------------------------------------------------
# 边门控(研究计划"方法 B"):给 MACE 里每条"边"(原子对)的消息乘一个 0~1 的开关 g。
#
# MACE 每层交互里的消息 m_ij 是在 conv_tp 模块里按边算出来的
# (节点特征 ⊗ 方向球谐特征,再用边径向特征算出的权重组合)。
# 把 conv_tp 的输出按边乘上 g_ij,就精确实现了 m_ij -> g_ij * m_ij,
# 后续的线性层、聚合、读出全部保持不变。g 是标量,不改变任何张量的
# 旋转性质,所以等变性保持不变;无向原子对 (i,j) 和 (j,i) 共用一个 g。
#
# 门的含义:g=1 是正常模型;g=0 表示这条边的消息被完全关掉。
# ---------------------------------------------------------------------------


class _GatedModule(torch.nn.Module):
    """包一层:先算原来的模块,再把输出按每条边乘 g。"""

    def __init__(self, inner: torch.nn.Module, holder: "EdgeGater"):
        super().__init__()
        self.inner = inner
        self.holder = holder

    def forward(self, *args, **kwargs):
        out = self.inner(*args, **kwargs)
        gate = self.holder.current_edge_gate
        if gate is None:
            return out
        return out * gate.unsqueeze(-1)


class EdgeGater:
    """把 MACE 每个交互层里的消息模块(conv_tp)包上门控,用完记得 remove() 还原。"""

    def __init__(self, calc):
        self.calc = calc
        self.model = calc.models[0]
        self.current_edge_gate: torch.Tensor | None = None
        self._wrapped: list[tuple[torch.nn.Module, str, torch.nn.Module]] = []

    def install(self) -> None:
        for interaction in self.model.interactions:
            original = interaction.conv_tp
            interaction.conv_tp = _GatedModule(original, self)
            self._wrapped.append((interaction, "conv_tp", original))

    def remove(self) -> None:
        for interaction, name, original in self._wrapped:
            setattr(interaction, name, original)
        self._wrapped = []

    def set_edge_gate(self, gate_per_edge: torch.Tensor | None) -> None:
        self.current_edge_gate = gate_per_edge


def edge_pair_ids(edge_index: np.ndarray, unit_shifts: np.ndarray) -> np.ndarray:
    """把有向边配对成"无向原子对",返回每条边所属原子对的编号。

    (i->j, 位移 s) 和 (j->i, 位移 -s) 是同一个原子对,分到同一个编号。
    编号按第一次出现的顺序分配,所以同一批原子顺序下结果是稳定的。
    """

    n_edges = edge_index.shape[1]
    ids = np.full(n_edges, -1, dtype=int)
    lookup: dict[tuple, int] = {}
    for e in range(n_edges):
        i = int(edge_index[0, e])
        j = int(edge_index[1, e])
        s = tuple(int(round(float(v))) for v in unit_shifts[e])
        reverse_key = (j, i, tuple(-v for v in s))
        if reverse_key in lookup:
            ids[e] = lookup[reverse_key]
        else:
            ids[e] = len(lookup)
            lookup[(i, j, s)] = int(ids[e])
    return ids


def canonical_pair_keys(pair_ids: np.ndarray, edge_index: np.ndarray, unit_shifts: np.ndarray) -> list[tuple]:
    """把每个原子对写成一个规范键 (i, j, 单胞镜像位移),i < j。

    正反两条有向边会映射到同一个键;晶胞不变时,同一个物理原子对在
    任何结构里键都相同,所以可以用它跨结构对齐(比如完美 vs 空位)。
    """

    keys: dict[int, tuple] = {}
    for e, pair in enumerate(pair_ids):
        if int(pair) in keys:
            continue
        i, j = int(edge_index[0, e]), int(edge_index[1, e])
        shift = tuple(int(round(float(v))) for v in unit_shifts[e])
        if i > j or (i == j and shift < (0, 0, 0)):
            i, j, shift = j, i, tuple(-v for v in shift)
        keys[int(pair)] = (i, j, shift)
    return [keys[int(p)] for p in range(len(keys))]


def edge_index_shifts(data: dict) -> tuple[np.ndarray, np.ndarray]:
    """从 MACE 的图数据里拿有向边表和单胞位移。"""

    edge_index = data["edge_index"].cpu().numpy()
    unit_shifts = data["unit_shifts"].cpu().numpy()
    return edge_index, unit_shifts


def _batch_and_pair_info(calc, atoms: Atoms):
    """拿到 MACE 的图数据,并算出"每条边属于哪个原子对"。"""

    batch = calc._atoms_to_batch(atoms)
    data = batch.to_dict()
    edge_index, unit_shifts = edge_index_shifts(data)
    pair_ids = edge_pair_ids(edge_index, unit_shifts)
    return data, pair_ids


def pair_distances(calc, atoms: Atoms, pair_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """每个原子对的代表性信息:原子 i、原子 j、最小镜像距离。

    同一个原子对有两条有向边,随便取第一条的距离(两者相等)。
    """

    batch = calc._atoms_to_batch(atoms)
    data = batch.to_dict()
    edge_index = data["edge_index"].cpu().numpy()
    positions = data["positions"].cpu().numpy()
    shifts = data["shifts"].cpu().numpy()

    first_edge_of_pair: dict[int, int] = {}
    for e, pair in enumerate(pair_ids):
        if pair not in first_edge_of_pair:
            first_edge_of_pair[pair] = e

    n_pairs = len(first_edge_of_pair)
    idx_i = np.zeros(n_pairs, dtype=int)
    idx_j = np.zeros(n_pairs, dtype=int)
    dist = np.zeros(n_pairs)
    for pair, e in first_edge_of_pair.items():
        i, j = int(edge_index[0, e]), int(edge_index[1, e])
        # shifts 已经是笛卡尔坐标下的绝对位移(单位位移乘过晶胞了),不要再乘
        vec = positions[j] + shifts[e] - positions[i]
        idx_i[pair], idx_j[pair] = i, j
        dist[pair] = float(np.linalg.norm(vec))
    return idx_i, idx_j, dist


def mace_gated_energy(calc, atoms: Atoms, gater: EdgeGater, gate_per_pair=None) -> float:
    """带门控地算一次总能量。gate_per_pair 为 None 表示全部打开(正常模型)。"""

    data, pair_ids = _batch_and_pair_info(calc, atoms)
    if gate_per_pair is None:
        gater.set_edge_gate(None)
    else:
        g = torch.as_tensor(gate_per_pair, dtype=data["positions"].dtype, device=data["positions"].device)
        gater.set_edge_gate(g[torch.as_tensor(pair_ids, device=g.device)])
    out = calc.models[0](data, training=False, compute_force=False)
    return float(out["energy"].sum().detach().cpu().item())


def edge_gate_integrated_gradients(
    calc,
    atoms: Atoms,
    n_steps: int = 32,
) -> dict:
    """边门控的积分梯度归因(研究计划的核心方法)。

    通俗地说:把每条边的开关 g 从 0 慢慢打开到 1,一路上记录
    "总能量对每个开关的敏感程度" (∂E/∂g),把敏感度沿着路径加起来,
    就得到这条边对总能量的归因值。归因值越大,说明这条边携带的消息
    对模型能量越重要。

    基线取 g=0(所有消息关闭)。数学上这满足"完备性":
    所有边的归因之和 ≈ E(g=1) - E(g=0),代码里会把这个残差也记下来。
    """

    data, pair_ids = _batch_and_pair_info(calc, atoms)
    n_pairs = int(pair_ids.max()) + 1
    model = calc.models[0]
    dtype = data["positions"].dtype
    device = data["positions"].device
    pair_ids_t = torch.as_tensor(pair_ids, device=device)

    # 包上门控
    gater = EdgeGater(calc)
    gater.install()
    try:
        # E(g=1):正常模型
        gater.set_edge_gate(torch.ones(data["edge_index"].shape[1], dtype=dtype, device=device))
        energy_full = float(model(data, training=False, compute_force=False)["energy"].sum().item())
        # E(g=0):所有消息关闭
        gater.set_edge_gate(torch.zeros(data["edge_index"].shape[1], dtype=dtype, device=device))
        energy_closed = float(model(data, training=False, compute_force=False)["energy"].sum().item())

        alphas = np.linspace(0.0, 1.0, n_steps)
        grads = np.zeros((n_steps, n_pairs))
        for step, alpha in enumerate(alphas):
            # 注意:必须直接构造叶子张量(不能 ones*alpha),否则拿不到梯度
            g = torch.full((n_pairs,), float(alpha), dtype=dtype, device=device, requires_grad=True)
            gater.set_edge_gate(g[pair_ids_t])
            energy = model(data, training=False, compute_force=False)["energy"].sum()
            energy.backward()
            grads[step] = g.grad.detach().cpu().numpy()

        # 沿 alpha 积分(梯形公式)。路径从 0 到 1,乘的系数就是 1。
        attribution = np.trapz(grads, x=alphas, axis=0)
    finally:
        gater.remove()

    return {
        "attribution": attribution,
        "n_pairs": n_pairs,
        "n_edges": data["edge_index"].shape[1],
        "energy_full": energy_full,
        "energy_closed": energy_closed,
        "completeness_residual": float(attribution.sum() - (energy_full - energy_closed)),
        "pair_ids": pair_ids,
        "has_pair_repulsion": bool(hasattr(model, "pair_repulsion") and getattr(model, "pair_repulsion", False)),
    }


def edge_ablation_importance(calc, atoms: Atoms) -> dict:
    """留一法对照:每次只关掉一个原子对,看能量变化多少。

    这是最"笨"但最直接的因果实验,用来检验积分梯度的归因是否靠谱。
    """

    data, pair_ids = _batch_and_pair_info(calc, atoms)
    n_pairs = int(pair_ids.max()) + 1
    gater = EdgeGater(calc)
    gater.install()
    try:
        energy_full = mace_gated_energy(calc, atoms, gater, None)
        importance = np.zeros(n_pairs)
        ones = np.ones(n_pairs)
        for pair in range(n_pairs):
            gate = ones.copy()
            gate[pair] = 0.0
            importance[pair] = energy_full - mace_gated_energy(calc, atoms, gater, gate)
    finally:
        gater.remove()
    return {"importance": importance, "energy_full": energy_full, "pair_ids": pair_ids}


def faithfulness_curve(
    calc,
    atoms: Atoms,
    ranking: np.ndarray,
    ks: list[int],
    n_random: int = 5,
    seed: int = 0,
) -> dict:
    """忠实性检验(研究计划第六节):按归因排名从高到低关边,能量应该掉得最快。

    - top-k:关掉归因最大的 k 条边;
    - random:随机关 k 条边(重复 n_random 次取平均);
    - bottom-k:关掉归因最小的 k 条边。
    如果 top-k 曲线明显高于另外两条,说明归因确实抓住了"模型在用哪些边"。
    """

    data, pair_ids = _batch_and_pair_info(calc, atoms)
    n_pairs = int(pair_ids.max()) + 1
    gater = EdgeGater(calc)
    gater.install()
    rng = np.random.default_rng(seed)
    order = np.argsort(-np.abs(ranking))
    try:
        energy_full = mace_gated_energy(calc, atoms, gater, None)
        rows = []
        for k in ks:
            k = min(k, n_pairs)
            top_gate = np.ones(n_pairs)
            top_gate[order[:k]] = 0.0
            delta_top = abs(energy_full - mace_gated_energy(calc, atoms, gater, top_gate))

            bottom_gate = np.ones(n_pairs)
            bottom_gate[order[n_pairs - k :]] = 0.0
            delta_bottom = abs(energy_full - mace_gated_energy(calc, atoms, gater, bottom_gate))

            random_deltas = []
            for _ in range(n_random):
                random_gate = np.ones(n_pairs)
                random_gate[rng.choice(n_pairs, size=k, replace=False)] = 0.0
                random_deltas.append(abs(energy_full - mace_gated_energy(calc, atoms, gater, random_gate)))

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

    return {"energy_full": energy_full, "n_pairs": n_pairs, "rows": rows}


def rotation_consistency(calc, atoms: Atoms, angle_degrees: float = 37.0) -> dict:
    """对称性检验:整体旋转结构后,每个原子对的门控归因应该不变。

    旋转不改变原子编号和原子对的集合,但会改变邻居列表里边的"顺序",
    所以两次归因必须先按 (i, j) 原子对身份对齐,再逐对比较。
    """

    from atomlens.symmetry import rotated_z

    result_a = edge_gate_integrated_gradients(calc, atoms)
    result_b = edge_gate_integrated_gradients(calc, rotated_z(atoms, angle_degrees))

    idx_i_a, idx_j_a, _ = pair_distances(calc, atoms, result_a["pair_ids"])
    idx_i_b, idx_j_b, _ = pair_distances(calc, rotated_z(atoms, angle_degrees), result_b["pair_ids"])
    values_a = {(int(idx_i_a[p]), int(idx_j_a[p])): float(result_a["attribution"][p]) for p in range(result_a["n_pairs"])}
    values_b = {(int(idx_i_b[p]), int(idx_j_b[p])): float(result_b["attribution"][p]) for p in range(result_b["n_pairs"])}
    if len(values_a) != result_a["n_pairs"] or len(values_b) != result_b["n_pairs"]:
        raise RuntimeError("duplicate atom pairs in edge list, cannot align attributions")
    common = sorted(set(values_a) & set(values_b))

    from atomlens.metrics import pairwise_max_error, spearman_correlation

    aligned_a = np.array([values_a[key] for key in common])
    aligned_b = np.array([values_b[key] for key in common])
    return {
        "n_common_pairs": len(common),
        "spearman": spearman_correlation(aligned_a, aligned_b),
        "max_abs_error": pairwise_max_error(aligned_a, aligned_b),
        "attribution_scale": float(np.max(np.abs(aligned_a))),
    }
