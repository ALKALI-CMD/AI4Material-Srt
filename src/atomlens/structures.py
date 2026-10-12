from __future__ import annotations

import numpy as np
from ase import Atoms


# ---------------------------------------------------------------------------
# 基础结构。所有函数都返回 ase.Atoms,单位:埃、电子伏(和 ASE 约定一致)。
# ---------------------------------------------------------------------------


def cubic_cspbi3(a: float = 6.29) -> Atoms:
    """理想的立方相 CsPbI3,一个原胞 5 个原子(Cs + Pb + 3 个 I)。"""

    symbols = ["Cs", "Pb", "I", "I", "I"]
    scaled = [
        (0.0, 0.0, 0.0),
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.0),
        (0.5, 0.0, 0.5),
        (0.0, 0.5, 0.5),
    ]
    return Atoms(symbols=symbols, scaled_positions=scaled, cell=[a, a, a], pbc=True)


def cspbi3_iodine_displacement(a: float = 6.29, dz: float = 0.08) -> tuple[Atoms, Atoms]:
    """返回(扰动前, 扰动后)两个 CsPbI3 原胞,用于固定的坐标路径测试。"""

    start = cubic_cspbi3(a=a)
    end = start.copy()
    positions = end.get_positions()
    positions[2, 2] += dz
    positions[3, 0] -= dz
    positions[4, 1] += dz
    end.set_positions(positions)
    return start, end


def diamond_si(a: float = 5.43) -> Atoms:
    """两原子的金刚石结构硅,用来做周期边界的最简单测试。"""

    return Atoms(
        symbols=["Si", "Si"],
        scaled_positions=[(0.0, 0.0, 0.0), (0.25, 0.25, 0.25)],
        cell=[
            (0.0, a / 2.0, a / 2.0),
            (a / 2.0, 0.0, a / 2.0),
            (a / 2.0, a / 2.0, 0.0),
        ],
        pbc=True,
    )


# ---------------------------------------------------------------------------
# CsPbI3 的主案例结构:超胞、八面体倾斜、碘空位、热扰动。
# 这些是研究计划里"钙钛矿主案例"的最小集合。
# ---------------------------------------------------------------------------


def cspbi3_supercell(n: int = 2, a: float = 6.29) -> Atoms:
    """n x n x n 的 CsPbI3 超胞(默认 2x2x2,共 40 个原子)。"""

    return cubic_cspbi3(a=a).repeat((n, n, n))


def tilted_cspbi3(n: int = 2, tilt_deg: float = 10.0, a: float = 6.29) -> Atoms:
    """让每个 PbI6 八面体绕 z 轴转一个角度,做出"八面体倾斜"的结构。

    做法(通俗版):
    1. 先建 n x n x n 超胞;
    2. 对每个 Pb 原子,找到它周围的 6 个最近的 I;
    3. 让这个八面体绕 z 轴转 tilt_deg,方向按棋盘格正负交替(相邻八面体转得
       相反,这样才是真实的"倾斜",而不是整体旋转);
    4. 每个 I 被两个八面体共享,所以它的位移取两个八面体给出位移的平均。
    Cs 和 Pb 不动。角度越大,八面体歪得越厉害。
    """

    atoms = cspbi3_supercell(n=n, a=a)
    positions = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()
    cell = atoms.cell.array

    # 找出所有 Pb 的下标
    pb_indices = [i for i, s in enumerate(symbols) if s == "Pb"]

    # 每个原子收到的位移累加,以及被几个八面体"推过"的次数
    displacement = np.zeros_like(positions)
    count = np.zeros(len(atoms))

    angle = np.deg2rad(tilt_deg)
    cos_t, sin_t = np.cos(angle), np.sin(angle)

    for pb in pb_indices:
        # Pb 原胞在超胞里的格子编号(0..n-1),用来决定倾斜的正负号(棋盘交替)。
        # 注意 Pb 总在格子中心(frac*-n 是 0.5, 1.5, ...),所以先减 0.5 再取整。
        frac = np.linalg.solve(cell.T, positions[pb])
        mx = int(round(frac[0] * n - 0.5)) % n
        my = int(round(frac[1] * n - 0.5)) % n
        sign = 1.0 if (mx + my) % 2 == 0 else -1.0

        # 找这个 Pb 周围 6 个最近的 I(最短镜像距离)
        neighbors = []
        for i, s in enumerate(symbols):
            if s != "I":
                continue
            dist = atoms.get_distance(pb, i, mic=True)
            neighbors.append((dist, i))
        neighbors.sort()
        nearest = [i for _, i in neighbors[:6]]

        for iod in nearest:
            # I 相对 Pb 的位移矢量(最小镜像约定下)
            vec = atoms.get_distance(pb, iod, mic=True, vector=True)
            # 绕 z 轴旋转 sign*tilt_deg 之后的新矢量,减去原来的就是切向位移
            rotated = np.array(
                [
                    cos_t * vec[0] - sign * sin_t * vec[1],
                    sign * sin_t * vec[0] + cos_t * vec[1],
                    vec[2],
                ]
            )
            displacement[iod] += (rotated - vec) / 2.0
            count[iod] += 1

    # 检查:每个 I 都应该被 2 个八面体共享
    for i, s in enumerate(symbols):
        if s == "I" and count[i] != 2:
            raise RuntimeError(f"iodine atom {i} was shared by {count[i]:.0f} octahedra, expected 2")

    tilted = atoms.copy()
    tilted.set_positions(positions + displacement)
    return tilted


def cspbi3_tilt_path(n: int = 2, tilt_deg: float = 10.0, a: float = 6.29) -> tuple[Atoms, Atoms]:
    """返回(不倾斜, 倾斜 tilt_deg)两个结构,构成一条物理上有意义的路径:
    从立方相出发、八面体逐渐倾斜。用于结构路径积分梯度。
    """

    return cspbi3_supercell(n=n, a=a), tilted_cspbi3(n=n, tilt_deg=tilt_deg, a=a)


def cspbi3_iodine_vacancy(n: int = 2, vacancy: int = 0, a: float = 6.29) -> Atoms:
    """带一个碘空位的 CsPbI3 超胞。

    vacancy=0 表示删掉超胞里第 0 个 I 原子(我们的原子顺序是固定的,
    所以这个选择是可以复现的)。
    """

    atoms = cspbi3_supercell(n=n, a=a)
    iodine_indices = [i for i, s in enumerate(atoms.get_chemical_symbols()) if s == "I"]
    del atoms[iodine_indices[vacancy]]
    return atoms


def add_thermal_noise(atoms: Atoms, sigma: float = 0.03, seed: int = 0) -> Atoms:
    """给每个原子的坐标加上一小段高斯随机位移,模拟热振动。

    sigma 是位移的标准差(埃),0.03 埃大约对应很轻微的热扰动。
    同一个 seed 生成的扰动是一样的,保证实验可以复现。
    """

    rng = np.random.default_rng(seed)
    noisy = atoms.copy()
    noise = rng.normal(0.0, sigma, size=atoms.get_positions().shape)
    noisy.set_positions(atoms.get_positions() + noise)
    return noisy


def structure_by_name(name: str) -> Atoms:
    """按名字构造结构(CLI 里用)。"""

    if name == "diamond_si":
        return diamond_si()
    if name == "cubic_cspbi3":
        return cubic_cspbi3()
    if name == "cspbi3_supercell":
        return cspbi3_supercell()
    if name == "cspbi3_tilted":
        return tilted_cspbi3()
    if name == "cspbi3_iodine_vacancy":
        return cspbi3_iodine_vacancy()
    raise ValueError(f"unknown structure: {name}")


def structure_pair_by_name(name: str) -> tuple[Atoms, Atoms]:
    """按名字构造(起点, 终点)结构对(CLI 里用)。"""

    if name == "cspbi3_iodine_displacement":
        return cspbi3_iodine_displacement()
    if name == "cspbi3_tilt":
        return cspbi3_tilt_path()
    raise ValueError(f"unknown structure pair: {name}")
