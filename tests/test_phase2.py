import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.neighborlist import neighbor_list

from atomlens.differential import formation_numbers, vacancy_to_perfect_index
from atomlens.gating import canonical_pair_keys, edge_pair_ids
from atomlens.invariance import invariance_report
from atomlens.metrics import spearman_correlation, topk_jaccard
from atomlens.structures import (
    add_thermal_noise,
    cspbi3_iodine_vacancy,
    cspbi3_supercell,
    cubic_cspbi3,
    tilted_cspbi3,
)


class PairSumCalculator(Calculator):
    """一个极简的两体计算器:E = sum_ij exp(-(r_ij/sigma)^2)。

    只依赖原子间距离,所以平移、旋转、重新编号都不变;
    最近邻环境不因扩胞改变,所以超胞能量是 8 倍。专门用来测 invariance_report。
    """

    implemented_properties = ["energy", "forces"]

    def __init__(self, sigma: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.sigma = float(sigma)

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        i, j, D = neighbor_list("ijD", self.atoms, cutoff=3 * self.sigma)
        r = np.linalg.norm(D, axis=1)
        pair_energy = np.exp(-(r / self.sigma) ** 2)
        energy = float(pair_energy.sum()) / 2.0  # 每对原子被数了两次
        # 能量对坐标的梯度: dE/dr = exp(...) * (-2 r / sigma^2),力取负号
        pair_force_scalar = pair_energy * (-2.0 * r / self.sigma**2) / 2.0
        forces = np.zeros((len(self.atoms), 3))
        np.add.at(forces, i, -pair_force_scalar[:, None] * D / r[:, None])
        np.add.at(forces, j, pair_force_scalar[:, None] * D / r[:, None])
        self.results["energy"] = energy
        self.results["forces"] = forces


def test_tilted_cspbi3_moves_only_iodine():
    base = cspbi3_supercell()
    tilted = tilted_cspbi3(tilt_deg=10.0)

    assert list(tilted.get_chemical_symbols()) == list(base.get_chemical_symbols())
    base_pos = base.get_positions()
    tilt_pos = tilted.get_positions()
    moved = np.linalg.norm(tilt_pos - base_pos, axis=1)
    for index, symbol in enumerate(base.get_chemical_symbols()):
        if symbol in ("Cs", "Pb"):
            assert moved[index] == pytest.approx(0.0, abs=1e-12)
    assert moved.max() > 0.1  # 碘确实被推动了


def test_tilted_cspbi3_scales_with_angle():
    small = tilted_cspbi3(tilt_deg=4.0).get_positions()
    base = cspbi3_supercell().get_positions()
    large = tilted_cspbi3(tilt_deg=12.0).get_positions()
    small_shift = np.linalg.norm(small - base, axis=1).max()
    large_shift = np.linalg.norm(large - base, axis=1).max()
    assert large_shift > small_shift


def test_iodine_vacancy_removes_one_iodine():
    vacancy = cspbi3_iodine_vacancy()
    symbols = vacancy.get_chemical_symbols()
    assert len(symbols) == 39  # 40 - 1
    assert symbols.count("I") == 23  # 24 - 1
    assert symbols.count("Pb") == 8


def test_thermal_noise_is_reproducible():
    atoms = cubic_cspbi3()
    first = add_thermal_noise(atoms, sigma=0.05, seed=3).get_positions()
    second = add_thermal_noise(atoms, sigma=0.05, seed=3).get_positions()
    third = add_thermal_noise(atoms, sigma=0.05, seed=4).get_positions()
    assert np.array_equal(first, second)
    assert not np.array_equal(first, third)


def test_spearman_and_jaccard():
    values = np.array([3.0, -1.0, 2.0, 0.5])
    assert spearman_correlation(values, values) == pytest.approx(1.0)
    assert spearman_correlation(values, -values) == pytest.approx(-1.0)
    assert topk_jaccard(values, values, k=2) == pytest.approx(1.0)
    # top-1 完全不同的两组
    assert topk_jaccard(np.array([3.0, 0.1, 0.2, 0.4]), np.array([0.1, 2.9, 0.2, 0.4]), k=1) == pytest.approx(0.0)


def test_edge_pair_ids_matches_reverse_edges():
    # 边 0-1(正反), 0-2 带 (1,0,0) 位移(正反), 1-2(只有一条,自成一对)
    edge_index = np.array([[0, 1, 0, 2, 1], [1, 0, 2, 0, 2]])
    unit_shifts = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    ids = edge_pair_ids(edge_index, unit_shifts)
    assert ids.tolist() == [0, 0, 1, 1, 2]


def test_canonical_pair_keys_normalizes_direction():
    # 正反两条边 -> 同一个规范键;单向边 -> i<j 且镜像取反
    edge_index = np.array([[1, 0, 2], [0, 1, 1]])
    unit_shifts = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    pair_ids = np.array([0, 0, 1])
    keys = canonical_pair_keys(pair_ids, edge_index, unit_shifts)
    assert keys[0] == (0, 1, (0, 0, 0))
    assert keys[1] == (1, 2, (-1, 0, 0))


def test_vacancy_to_perfect_index_map():
    mapping = vacancy_to_perfect_index(39)
    assert len(mapping) == 39
    assert mapping[0] == 0 and mapping[1] == 1  # 删掉的原子(编号2)之前的原子不变
    assert mapping[2] == 3  # 之后的整体 +1
    assert mapping[-1] == 39  # 完美超胞最后一个原子
    assert set(mapping.tolist()) == set(range(40)) - {2}  # 恰好跳过被删的 2 号


def test_formation_numbers_formula():
    numbers = formation_numbers(e_perf=-113.4, e_vac=-111.05, e_i2=-3.0, n_perf=40, n_vac=39)
    assert numbers["mu_I_eV"] == pytest.approx(-1.5)
    # E_form = E_vac - E_perf + mu_I = -111.05 + 113.4 - 1.5
    assert numbers["formation_energy_I_rich_eV"] == pytest.approx(0.85)
    # 粗略缩放: E_vac - 39/40 * E_perf
    assert numbers["formation_energy_crude_scaling_eV"] == pytest.approx(-111.05 + 110.565)


def test_invariance_report_passes_for_distance_based_calculator():
    atoms = cspbi3_supercell(n=1)
    atoms.positions[2] += np.array([0.05, -0.03, 0.02])  # 轻微扰动,让能量非零
    report = invariance_report(atoms, lambda: PairSumCalculator(), supercell_n=2)
    assert report["passed"] is True
    assert report["translation_force_error"] < 1e-8
    assert report["rotation_force_error"] < 1e-6
    assert report["supercell_relative_energy_error"] < 1e-6
