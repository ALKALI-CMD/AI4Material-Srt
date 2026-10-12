from __future__ import annotations

import numpy as np
from ase import Atoms

from atomlens.symmetry import rotated_z, translated


def _energy_forces(atoms: Atoms, calculator_factory) -> tuple[float, np.ndarray]:
    """给结构挂上计算器,算一次能量和力。每次都新建一个计算器,避免缓存干扰。"""

    test = atoms.copy()
    test.calc = calculator_factory()
    energy = float(test.get_potential_energy())
    forces = np.asarray(test.get_forces(), dtype=float)
    return energy, forces


def invariance_report(
    atoms: Atoms,
    calculator_factory,
    supercell_n: int = 2,
    rotation_degrees: float = 37.0,
) -> dict:
    """检验一个模型对"不该改变的变换"是否真的不改变,对应研究计划第六节"对称性"。

    检查四件事(都是等变模型理论上必须满足的):
    1. 平移:整体挪一格,能量、每个原子的力都不该变;
    2. 旋转:整体转一个角度,能量不变,力要跟着一起转;
    3. 原子重新编号:能量不变,力跟着原子走;
    4. 超胞:2x2x2 超胞的能量应该约等于 8 倍原胞能量。

    返回一个字典,里面是每种变换的最大误差,越接近 0 越好。
    """

    energy0, forces0 = _energy_forces(atoms, calculator_factory)

    # ---- 1. 平移 ----
    shift = (0.31, -1.7, 0.62)
    energy_t, forces_t = _energy_forces(translated(atoms, shift), calculator_factory)
    trans_energy_error = abs(energy_t - energy0)
    trans_force_error = float(np.max(np.abs(forces_t - forces0)))

    # ---- 2. 旋转 ----
    rotated = rotated_z(atoms, rotation_degrees)
    energy_r, forces_r = _energy_forces(rotated, calculator_factory)
    rot_energy_error = abs(energy_r - energy0)
    angle = np.deg2rad(rotation_degrees)
    rot_matrix = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    # 旋转后的力应该等于原力被同一个矩阵旋转
    rot_force_error = float(np.max(np.abs(forces_r - forces0 @ rot_matrix.T)))

    # ---- 3. 原子重新编号 ----
    rng = np.random.default_rng(0)
    order = rng.permutation(len(atoms))
    shuffled = atoms[order]
    energy_p, forces_p = _energy_forces(shuffled, calculator_factory)
    perm_energy_error = abs(energy_p - energy0)
    # 力要跟着原子换位置
    perm_force_error = float(np.max(np.abs(forces_p - forces0[order])))

    # ---- 4. 超胞(能量是广延量)----
    repeated = atoms.repeat((supercell_n, supercell_n, supercell_n))
    energy_s, _ = _energy_forces(repeated, calculator_factory)
    expected = energy0 * supercell_n**3
    scale = max(abs(expected), 1.0)
    supercell_error = abs(energy_s - expected) / scale  # 用相对误差,因为能量本身可能很大

    report = {
        "translation_energy_error": trans_energy_error,
        "translation_force_error": trans_force_error,
        "rotation_energy_error": rot_energy_error,
        "rotation_force_error": rot_force_error,
        "permutation_energy_error": perm_energy_error,
        "permutation_force_error": perm_force_error,
        "supercell_relative_energy_error": supercell_error,
        "energy_reference": energy0,
        "n_atoms": len(atoms),
    }
    # 容差:单精度浮点 + 邻居列表的数值细节,给一个宽松但有用的门槛
    report["passed"] = bool(
        trans_energy_error < 1e-3
        and trans_force_error < 1e-3
        and rot_energy_error < 1e-3
        and rot_force_error < 1e-3
        and perm_energy_error < 1e-3
        and perm_force_error < 1e-3
        and supercell_error < 1e-3
    )
    return report
