from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def spearman_correlation(values_a: np.ndarray, values_b: np.ndarray) -> float:
    """两个重要性列表的 Spearman 秩相关(越大越接近,1 表示排序完全一样)。"""

    values_a = np.asarray(values_a, dtype=float)
    values_b = np.asarray(values_b, dtype=float)
    rho, _ = spearmanr(values_a, values_b)
    if np.isnan(rho):
        # 常数序列会导致相关没有定义,这种情况按 0 处理
        return 0.0
    return float(rho)


def topk_jaccard(values_a: np.ndarray, values_b: np.ndarray, k: int) -> float:
    """按绝对值取前 k 个,看两个集合有多少重合(Jaccard = 交集/并集)。"""

    values_a = np.asarray(values_a, dtype=float)
    values_b = np.asarray(values_b, dtype=float)
    k = max(1, min(k, len(values_a)))
    top_a = set(np.argsort(-np.abs(values_a))[:k].tolist())
    top_b = set(np.argsort(-np.abs(values_b))[:k].tolist())
    union = top_a | top_b
    if not union:
        return 1.0
    return len(top_a & top_b) / len(union)


def pairwise_max_error(values_a: np.ndarray, values_b: np.ndarray) -> float:
    """逐元素比较的最大绝对误差。"""

    values_a = np.asarray(values_a, dtype=float)
    values_b = np.asarray(values_b, dtype=float)
    return float(np.max(np.abs(values_a - values_b)))


def partial_spearman(values_a: np.ndarray, values_b: np.ndarray, control: np.ndarray) -> float:
    """控制第三个变量后的偏 Spearman 相关。

    做法(秩回归残差法,通俗版):先把三个量都换成排名,再用"排名 b"
    对"排名 control"做一次线性回归,取残差;"排名 a"同样处理;
    最后算两组残差的 Spearman。直觉:把 control 能解释的部分都扣掉,
    看 a 和 b 还剩多少相关。
    """

    values_a = np.asarray(values_a, dtype=float)
    values_b = np.asarray(values_b, dtype=float)
    control = np.asarray(control, dtype=float)
    rank_a = _rank(values_a)
    rank_b = _rank(values_b)
    rank_c = _rank(control)
    resid_a = rank_a - _linear_fit(rank_c, rank_a)
    resid_b = rank_b - _linear_fit(rank_c, rank_b)
    return spearman_correlation(resid_a, resid_b)


def _rank(values: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata

    return np.asarray(rankdata(values), dtype=float)


def _linear_fit(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    slope, intercept = np.polyfit(x, y, 1)
    return slope * x + intercept
