from __future__ import annotations

import numpy as np
import torch
from ase import Atoms


# ---------------------------------------------------------------------------
# 通道门控:把门控从"整条消息"细化到"消息里的某个角动量 l 通道"。
#
# MACE 的消息张量按 (通道数, l) 分块排列(比如 128x0e + 128x1o + ...)。
# 给每个 l 块单独乘一个 0~1 的门 g_{ij,l},就能问:
#   "关掉 l2 通道(只关它)会怎么改变能量?"
#
# 数学事实:整条消息的门 = 所有 l 块同时乘同一个 g。梯度是线性的,所以
# "所有块同时沿 0→1 路径走"时,各通道的积分梯度之和**精确等于**
# 整条消息的积分梯度(不用近似)。这样我们能把上一期的"哪条边重要"
# 细化成"哪条边的哪个通道重要",直接检验:
#   l2/l3 通道的范数之所以和归因相关,是不是因为能量敏感性真的从
#   这些通道流过?
# ---------------------------------------------------------------------------


def _irreps_l_blocks(irreps) -> list[tuple[int, int, int]]:
    """把 e3nn 的 irreps 解析成 (l, 起始列, 通道数) 列表。"""

    blocks = []
    start = 0
    for mul, ir in irreps:
        blocks.append((int(ir.l), start, int(mul)))
        start += mul
    return blocks


class ChannelGateWrapper(torch.nn.Module):
    """包住 conv_tp:对指定的 l 块,把输出按每条边的门缩放。"""

    def __init__(self, inner: torch.nn.Module, blocks: list[tuple[int, int, int]], holder: "ChannelGater"):
        super().__init__()
        self.inner = inner
        self.blocks = blocks
        self.holder = holder

    def forward(self, *args, **kwargs):
        out = self.inner(*args, **kwargs)
        gate = self.holder.current_channel_gate  # [n_edges, n_channels]
        if gate is None:
            return out
        # 构造按列展开的门矩阵:被门控通道的列乘上 g,其余列保持 1。
        # 用整张乘法而不是切片赋值,避免 autograd 的原地修改问题。
        n_edges, width = out.shape
        scale = torch.ones(n_edges, width, dtype=out.dtype, device=out.device)
        for index, (l, start, mul) in enumerate(self.blocks):
            scale[:, start : start + mul] = gate[:, index].unsqueeze(-1)
        return out * scale


class ChannelGater:
    """把每个交互层的 conv_tp 包上通道门控,用完记得 remove() 还原。"""

    def __init__(self, calc):
        self.calc = calc
        self.model = calc.models[0]
        self.current_channel_gate: torch.Tensor | None = None
        self._wrapped: list[tuple[torch.nn.Module, torch.nn.Module]] = []
        self.channel_names: list[str] = []

    def install(self) -> None:
        for interaction in self.model.interactions:
            blocks = _irreps_l_blocks(interaction.conv_tp.irreps_out)
            if not self.channel_names:
                self.channel_names = [f"l{l}" for l, _, _ in blocks]
            original = interaction.conv_tp
            interaction.conv_tp = ChannelGateWrapper(original, blocks, self)
            self._wrapped.append((interaction, original))

    def remove(self) -> None:
        for interaction, original in self._wrapped:
            interaction.conv_tp = original
        self._wrapped = []

    def set_channel_gate(self, gate_per_edge_per_channel: torch.Tensor | None) -> None:
        self.current_channel_gate = gate_per_edge_per_channel


def channel_integrated_gradients(calc, atoms: Atoms, n_steps: int = 24) -> dict:
    """所有 l 通道的门同时从 0 开到 1,分别积分每个通道的梯度。

    返回:
    - channels: 通道名列表(如 ["l0", "l1", "l2", "l3"])
    - attribution_per_channel: {通道: 每个原子对的归因}
    - attribution_total: 各通道之和(应等于整条消息门控的归因)
    - completeness: 之和与 (E全开 - E全关) 的残差
    """

    from atomlens.gating import _batch_and_pair_info, edge_pair_ids, pair_distances

    data, pair_ids = _batch_and_pair_info(calc, atoms)
    n_pairs = int(pair_ids.max()) + 1
    n_edges = data["edge_index"].shape[1]
    model = calc.models[0]
    dtype = data["positions"].dtype
    device = data["positions"].device
    pair_ids_t = torch.as_tensor(pair_ids, device=device)

    gater = ChannelGater(calc)
    gater.install()
    n_channels = len(gater.channel_names)
    try:
        # E(门全开)与 E(门全关):所有通道的门同时为 1 / 0
        gater.set_channel_gate(torch.ones(n_edges, n_channels, dtype=dtype, device=device))
        energy_full = float(model(data, training=False, compute_force=False)["energy"].sum().item())
        gater.set_channel_gate(torch.zeros(n_edges, n_channels, dtype=dtype, device=device))
        energy_closed = float(model(data, training=False, compute_force=False)["energy"].sum().item())

        alphas = np.linspace(0.0, 1.0, n_steps)
        grads = np.zeros((n_steps, n_pairs, n_channels))
        for step, alpha in enumerate(alphas):
            # 叶子张量:每个原子对每个通道一个门;正反两条有向边共享同一原子对的门
            g_pair_level = torch.zeros(n_pairs, n_channels, dtype=dtype, device=device, requires_grad=True)
            g_pair_level.data.fill_(float(alpha))
            gater.set_channel_gate(g_pair_level[pair_ids_t])
            energy = model(data, training=False, compute_force=False)["energy"].sum()
            energy.backward()
            grads[step] = g_pair_level.grad.detach().cpu().numpy()

        trapezoid = getattr(np, "trapezoid", None) or np.trapz
        per_channel = trapezoid(grads, x=alphas, axis=0)  # [n_pairs, n_channels]
    finally:
        gater.remove()

    idx_i, idx_j, distances = pair_distances(calc, atoms, pair_ids)
    attribution_total = per_channel.sum(axis=1)
    return {
        "channels": list(gater.channel_names),
        "attribution_per_channel": {name: per_channel[:, c] for c, name in enumerate(gater.channel_names)},
        "attribution_total": attribution_total,
        "energy_full": energy_full,
        "energy_closed": energy_closed,
        "completeness": float(attribution_total.sum() - (energy_full - energy_closed)),
        "n_pairs": n_pairs,
        "idx_i": idx_i,
        "idx_j": idx_j,
        "distances": distances,
        "pair_ids": pair_ids,
    }
