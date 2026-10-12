from __future__ import annotations

import numpy as np
from ase import Atoms


# ---------------------------------------------------------------------------
# MACE 的"角动量通道"分析(研究计划"方法 D",只做内部表征诊断):
#
# 等变网络的边特征按角动量 l 分成不同通道:l=0 是标量(只看距离),
# l=1 是向量(方向信息),l=2 及以上对应更精细的角向信息。
#
# 注意:直接看球谐函数本身的范数是没有意义的——单位向量的球谐函数
# 每个 l 块的范数永远是常数,不携带任何结构信息。所以这里记录的是
# 每层交互里"消息张量"(conv_tp 的输出)按 l 分块的均方根范数:
# 它反映模型实际使用了多少各个角动量通道的信息。
#
# 这只说明网络内部哪部分数值活跃,不能单独当成物理重要性,
# 必须和门控、消融结果对照。
# ---------------------------------------------------------------------------


def _irreps_l_blocks(irreps) -> list[tuple[int, int, int]]:
    """把 e3nn 的 irreps 字符串解析成 (l, 起始通道, 通道数) 列表。"""

    blocks = []
    start = 0
    for mul, ir in irreps:
        blocks.append((int(ir.l), start, int(mul)))
        start += mul
    return blocks


def mace_message_channel_norms(calc, atoms: Atoms) -> dict:
    """抓取 MACE 每层交互中,每条边消息张量按 l 分块的范数。

    返回:
    - l_max: 消息里最高的角动量通道
    - layers: 每层一个字典:
        edge_l_norms: {l: 每条边的均方根范数}
        atom_l_norms: {l: 每个原子的平均(原子作为消息的接收方)}
    """

    import torch

    batch = calc._atoms_to_batch(atoms)
    data = batch.to_dict()
    model = calc.models[0]
    n_atoms = data["positions"].shape[0]

    layers = []
    handles = []

    for interaction in model.interactions:
        l_blocks = _irreps_l_blocks(interaction.conv_tp.irreps_out)
        captured: dict = {}

        def make_hook(container, blocks):
            def hook(module, inputs, output):
                messages = output.detach()
                per_l = {}
                for l, start, mul in blocks:
                    block = messages[:, start : start + mul]
                    per_l[l] = block.pow(2).mean(dim=1).sqrt().cpu().numpy()
                container["per_l"] = per_l
                # 整条消息的均方根范数(所有通道一起算),给相关性分析用
                container["overall"] = messages.pow(2).mean(dim=1).sqrt().cpu().numpy()

            return hook

        handle = interaction.conv_tp.register_forward_hook(make_hook(captured, l_blocks))
        handles.append(handle)
        layers.append((captured, l_blocks))

    try:
        model(data, training=False, compute_force=False)
    finally:
        for handle in handles:
            handle.remove()

    edge_index = data["edge_index"].cpu().numpy()
    receiver = edge_index[1]
    counts = np.bincount(receiver, minlength=n_atoms).clip(min=1)

    result: dict = {"l_max": max(l for _, l_blocks in layers for l, _, _ in l_blocks), "layers": []}
    for captured, l_blocks in layers:
        per_l = captured["per_l"]
        layer_result = {"edge_l_norms": {}, "atom_l_norms": {}, "edge_overall_norms": None}
        for l in sorted(per_l):
            norms = np.asarray(per_l[l], dtype=float)
            atom_means = np.zeros(n_atoms)
            np.add.at(atom_means, receiver, norms)
            layer_result["edge_l_norms"][l] = norms
            layer_result["atom_l_norms"][l] = atom_means / counts
        overall = np.asarray(captured["overall"], dtype=float)
        atom_overall = np.zeros(n_atoms)
        np.add.at(atom_overall, receiver, overall)
        layer_result["edge_overall_norms"] = overall
        layer_result["atom_overall_norms"] = atom_overall / counts
        result["layers"].append(layer_result)
    return result


def channel_summary(calc, atoms: Atoms) -> dict:
    """把通道范数压缩成好画图的摘要:按元素求平均,外加全体平均。"""

    result = mace_message_channel_norms(calc, atoms)
    symbols = np.array(atoms.get_chemical_symbols())
    summary: dict = {
        "l_max": result["l_max"],
        "n_atoms": len(atoms),
        "layers": [],
    }
    for layer_index, layer in enumerate(result["layers"]):
        layer_entry: dict = {"layer": layer_index, "overall": {}, "by_element": {}}
        for l, values in layer["atom_l_norms"].items():
            layer_entry["overall"][f"l{l}"] = float(np.mean(values))
            for element in sorted(set(symbols)):
                layer_entry["by_element"].setdefault(element, {})[f"l{l}"] = float(
                    np.mean(values[symbols == element])
                )
        summary["layers"].append(layer_entry)
    return summary
