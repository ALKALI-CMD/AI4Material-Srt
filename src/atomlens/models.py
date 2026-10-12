from __future__ import annotations

from typing import Any

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator


def mace_mp_calculator(
    model: str = "small",
    device: str = "cuda",
    default_dtype: str = "float32",
    dispersion: bool = False,
    **kwargs: Any,
) -> Calculator:
    """Create a MACE-MP ASE calculator if mace-torch is installed.

    第一次调用会自动下载预训练权重(约 45 MB),之后走本地缓存。
    做归因实验建议 default_dtype="float64",梯度数值更稳定。
    """

    try:
        from mace.calculators import mace_mp
    except ImportError as exc:
        raise RuntimeError("Install mace-torch to use the MACE-MP adapter") from exc

    return mace_mp(
        model=model,
        device=device,
        default_dtype=default_dtype,
        dispersion=dispersion,
        **kwargs,
    )


def chgnet_calculator(device: str = "cuda", **kwargs: Any) -> Calculator:
    """Create a CHGNet ASE calculator if chgnet is installed.

    第一次调用会自动下载预训练权重,之后走本地缓存。
    """

    try:
        from chgnet.model.dynamics import CHGNetCalculator
    except ImportError as exc:
        raise RuntimeError("Install chgnet to use the CHGNet adapter") from exc

    return CHGNetCalculator(use_device=device, **kwargs)


def calculator_by_name(name: str, device: str = "cuda", float64: bool = False) -> Calculator:
    """按名字创建计算器,可选 "mace-mp" 和 "chgnet"(CLI 里用)。"""

    if name == "mace-mp":
        dtype = "float64" if float64 else "float32"
        return mace_mp_calculator(device=device, default_dtype=dtype)
    if name == "chgnet":
        return chgnet_calculator(device=device)
    raise ValueError(f"unknown calculator: {name}")


# ---------------------------------------------------------------------------
# 原子能分解:很多力场把总能量写成 E = sum_i E_i,所以模型内部天然有一个
# "每个原子分到多少能量"的数。注意:训练只约束总能量,单个 E_i 不是唯一的
# 可观测量,不能直接当成真实单原子能量;这里输出的是模型内部的能量分配。
# ---------------------------------------------------------------------------


def mace_forward(calc: Calculator, atoms: Atoms, compute_force: bool = True) -> dict:
    """直接调用 MACE 模型 forward,拿到底层输出张量。

    MACECalculator 内部就是这么算的,单独拿出来方便做能量分解、门控、
    特征提取,不用反复走 ASE 的接口。
    """

    batch = calc._atoms_to_batch(atoms)
    model = calc.models[0]
    return model(batch.to_dict(), training=False, compute_force=compute_force)


def _mace_e0(calc: Calculator, atoms: Atoms) -> np.ndarray:
    """MACE 的孤立原子基准能量 E0(每个元素一个常数,单位 eV)。"""

    batch = calc._atoms_to_batch(atoms)
    e0 = calc.models[0].atomic_energies_fn(batch["node_attrs"])
    return e0.detach().cpu().numpy().astype(float).flatten()


def mace_atomic_energies(calc: Calculator, atoms: Atoms) -> dict:
    """MACE 的逐原子能量,返回两个数组(单位 eV):

    - total:每个原子的总能量贡献(含 E0 基准),求和 = 总能量;
    - interaction:相互作用部分(total 减去 E0),求和 = 总能量 - sum(E0)。
    """

    out = mace_forward(calc, atoms, compute_force=False)
    total = out["node_energy"].detach().cpu().numpy().astype(float)
    return {"total": total, "interaction": total - _mace_e0(calc, atoms)}


def chgnet_atomic_energies(calc: Calculator, atoms: Atoms) -> dict:
    """CHGNet 的逐原子能量(单位 eV)。

    用 CHGNet 官方接口的 return_site_energies 选项,求和 = 总能量。
    """

    from pymatgen.io.ase import AseAtomsAdaptor

    structure = AseAtomsAdaptor.get_structure(atoms)
    graph = calc.model.graph_converter(structure)
    pred = calc.model.predict_graph(graph.to(calc.device), task="e", return_site_energies=True)
    site = pred["site_energies"]
    if hasattr(site, "detach"):  # 可能是 torch 张量,也可能是 numpy 数组
        site = site.detach().cpu().numpy()
    site = np.asarray(site, dtype=float)
    # CHGNet 没有单独输出 E0 基准,interaction 这里就等于 total
    return {"total": site, "interaction": site}


def atomic_energies(calc: Calculator, atoms: Atoms) -> dict:
    """按计算器类型自动选择原子能分解函数。"""

    if calc.__class__.__name__ == "MACECalculator":
        return mace_atomic_energies(calc, atoms)
    if calc.__class__.__name__ == "CHGNetCalculator":
        return chgnet_atomic_energies(calc, atoms)
    raise ValueError(f"atomic energy decomposition not supported for {type(calc).__name__}")
