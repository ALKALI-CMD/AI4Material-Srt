from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def plot_attribution(input_path: Path, output_path: Path) -> None:
    payload = _read_json(input_path)
    ranking = payload.get("ranking")
    if not isinstance(ranking, list) or not ranking:
        raise RuntimeError(f"{input_path} does not contain a non-empty ranking")

    labels = [f'{row["symbol"]}{row["index"]}' for row in ranking]
    values = [float(row["value"]) for row in ranking]

    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(10, 4.5))
    colors = ["#2f6f9f" if value >= 0 else "#b24a3a" for value in values]
    ax.bar(labels, values, color=colors)
    ax.axhline(0.0, color="#222222", linewidth=0.8)
    ax.set_ylabel("Attribution")
    ax.set_title(_title(payload, "Atomic attribution"))
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    fig.tight_layout()
    _save(fig, output_path)


def plot_convergence(input_path: Path, output_path: Path) -> None:
    payload = _read_json(input_path)
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError(f"{input_path} does not contain a non-empty rows list")

    steps = [int(row["steps"]) for row in rows]
    residuals = [abs(float(row["residual"])) for row in rows]
    totals = [float(row["attribution_total"]) for row in rows]
    deltas = [float(row["energy_delta"]) for row in rows]

    plt = _pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    axes[0].plot(steps, residuals, marker="o", color="#2f6f9f")
    axes[0].set_xlabel("Quadrature steps")
    axes[0].set_ylabel("|Residual|")
    axes[0].set_yscale("symlog", linthresh=1e-18)
    axes[0].set_title("Completeness residual")

    axes[1].plot(steps, deltas, marker="o", label="Energy delta", color="#2f6f9f")
    axes[1].plot(steps, totals, marker="s", label="Attribution total", color="#b36b2c")
    axes[1].set_xlabel("Quadrature steps")
    axes[1].set_ylabel("Energy")
    axes[1].set_title("Energy check")
    axes[1].legend()

    fig.suptitle(_title(payload, "Convergence"))
    fig.tight_layout()
    _save(fig, output_path)


def plot_comparison(input_path: Path, output_path: Path) -> None:
    payload = _read_json(input_path)
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError(f"{input_path} does not contain a non-empty rows list")

    labels = [str(row["index"]) for row in rows]
    left = [float(row["left"]) for row in rows]
    right = [float(row["right"]) for row in rows]

    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = list(range(len(labels)))
    width = 0.38
    ax.bar([value - width / 2 for value in x], left, width=width, label=payload.get("left_calculator", "left"))
    ax.bar([value + width / 2 for value in x], right, width=width, label=payload.get("right_calculator", "right"))
    ax.axhline(0.0, color="#222222", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_xlabel("Atom index, sorted by absolute difference")
    ax.set_ylabel("Attribution")
    ax.set_title(_comparison_title(payload))
    ax.legend()
    fig.tight_layout()
    _save(fig, output_path)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Install matplotlib to use plotting commands") from exc
    return plt


def _save(fig, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)


def _title(payload: dict[str, Any], fallback: str) -> str:
    calculator = payload.get("calculator")
    structure = payload.get("structure_pair") or payload.get("structure")
    if calculator and structure:
        return f"{calculator}: {structure}"
    return fallback


def _comparison_title(payload: dict[str, Any]) -> str:
    left = payload.get("left_calculator", "left")
    right = payload.get("right_calculator", "right")
    pearson = payload.get("pearson")
    if pearson is None:
        return f"{left} vs {right}"
    return f"{left} vs {right}, r={float(pearson):.3f}"


# ---------------------------------------------------------------------------
# 下面是第二阶段(归因与评价)新增的图。
# ---------------------------------------------------------------------------


def plot_edge_attribution(input_path: Path, output_path: Path) -> None:
    """边归因图:横轴是原子对距离,纵轴是归因值,归因最大的几条边标出来。"""

    payload = _read_json(input_path)
    pairs = payload.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise RuntimeError(f"{input_path} does not contain a non-empty pairs list")

    distances = [float(row["distance"]) for row in pairs]
    values = [float(row["value"]) for row in pairs]

    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sizes = [12 + 900 * abs(v) / max(abs(v) for v in values) for v in values]
    ax.scatter(distances, values, s=sizes, alpha=0.6, color="#2f6f9f", edgecolors="none")
    ax.axhline(0.0, color="#222222", linewidth=0.8)
    # 只标注前 3 条边;按排名阶梯错位,免得数值相同的点标签叠在一起
    for rank, row in enumerate(pairs[:3]):
        dx = 0.06 if rank % 2 == 0 else -0.95
        dy = 0.05 * (rank + 1) * (1 if rank % 2 == 0 else -1)
        ax.annotate(
            f'{row["i"]}-{row["j"]} ({float(row["distance"]):.2f} Å)',
            (float(row["distance"]) + dx, float(row["value"]) + dy),
            fontsize=8,
        )
    ax.set_xlabel("Pair distance (Å)")
    ax.set_ylabel("Edge attribution (∂E/∂g, integrated)")
    ax.set_title(_title(payload, "Edge gating attribution"))
    fig.tight_layout()
    _save(fig, output_path)


def plot_faithfulness(input_path: Path, output_path: Path) -> None:
    """忠实性曲线:关掉前 k 条边后能量变化,和随机/最不重要边对照。"""

    payload = _read_json(input_path)
    if isinstance(payload.get("faithfulness"), dict):  # 差分门控把曲线包在 faithfulness 里
        payload = payload["faithfulness"]
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError(f"{input_path} does not contain a non-empty rows list")

    ks = [int(row["k"]) for row in rows]
    top = [float(row["delta_top"]) for row in rows]
    random = [float(row["delta_random"]) for row in rows]
    random_std = [float(row["delta_random_std"]) for row in rows]
    bottom = [float(row["delta_bottom"]) for row in rows]

    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.errorbar(ks, random, yerr=random_std, marker="^", color="#7a7a7a", capsize=3, label="random edges")
    ax.plot(ks, bottom, marker="v", color="#b24a3a", label="least important edges")
    ax.plot(ks, top, marker="o", color="#2f6f9f", label="top attributed edges")
    ax.set_xlabel("number of gated (closed) edge pairs")
    ax.set_ylabel("|energy change| (eV)")
    ax.set_yscale("log")
    ax.set_title(_title(payload, "Faithfulness of edge attribution"))
    ax.legend()
    fig.tight_layout()
    _save(fig, output_path)


def plot_stability(input_path: Path, output_path: Path) -> None:
    """稳定性图:每次热扰动后,归因排序与无扰动时的 Spearman 相关和 Top-k 重合。"""

    payload = _read_json(input_path)
    if isinstance(payload.get("report"), dict):  # CLI 输出把数据包在 report 里
        payload = payload["report"]
    spearman = payload.get("spearman_per_noise")
    jaccard = payload.get("jaccard_per_noise")
    if not spearman:
        raise RuntimeError(f"{input_path} does not contain spearman_per_noise")

    noise_index = list(range(1, len(spearman) + 1))
    plt = _pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].bar(noise_index, spearman, color="#2f6f9f")
    axes[0].set_ylim(0, 1)
    axes[0].set_xlabel(f"noise sample (sigma={payload.get('sigma')})")
    axes[0].set_ylabel("Spearman vs clean")
    axes[0].set_title("Rank stability")
    axes[1].bar(noise_index, jaccard, color="#b36b2c")
    axes[1].set_ylim(0, 1)
    axes[1].set_xlabel(f"noise sample (top-{payload.get('topk')})")
    axes[1].set_ylabel("Jaccard overlap")
    axes[1].set_title("Top-k stability")
    fig.suptitle(_title(payload, "Stability under thermal noise"))
    fig.tight_layout()
    _save(fig, output_path)


def plot_channels(input_path: Path, output_path: Path) -> None:
    """通道分析图:八面体倾斜角增大时,各层消息里 l=0/1/2/3 通道的活跃度变化。"""

    payload = _read_json(input_path)
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError(f"{input_path} does not contain a non-empty rows list")

    angles = [float(row["tilt_deg"]) for row in rows]
    n_layers = len(rows[0].get("layers", []))
    if n_layers == 0:
        raise RuntimeError(f"{input_path} does not contain per-layer channel data")

    plt = _pyplot()
    fig, axes = plt.subplots(1, n_layers + 1, figsize=(4.2 * (n_layers + 1), 4.2))

    for layer_index in range(n_layers):
        ax = axes[layer_index]
        l_keys = sorted(rows[0]["layers"][layer_index]["overall"], key=lambda name: int(name[1:]))
        for name in l_keys:
            values = [float(row["layers"][layer_index]["overall"][name]) for row in rows]
            ax.plot(angles, values, marker="o", label=name)
        ax.set_xlabel("octahedral tilt (deg)")
        ax.set_ylabel("message channel RMS norm")
        ax.set_title(f"layer {layer_index}, all elements")
        ax.legend()

    # 最后一幅:第 0 层按元素拆开(以 Pb 为例最容易看出趋势)
    ax = axes[n_layers]
    elements = sorted(rows[0]["layers"][0]["by_element"])
    for element in elements:
        l_keys = sorted(rows[0]["layers"][0]["by_element"][element], key=lambda name: int(name[1:]))
        for name in l_keys:
            values = [float(row["layers"][0]["by_element"][element][name]) for row in rows]
            ax.plot(angles, values, marker=".", label=f"{element} {name}")
    ax.set_xlabel("octahedral tilt (deg)")
    ax.set_ylabel("message channel RMS norm")
    ax.set_title("layer 0, per element")
    ax.legend(fontsize=7)

    fig.suptitle(_title(payload, "MACE message angular channels"))
    fig.tight_layout()
    _save(fig, output_path)


def plot_correlation(input_path: Path, output_path: Path) -> None:
    """相关性图:每个"层 x 通道"的内部指标与干预归因的 Spearman 相关。

    相关越高,说明"看模型内部数值"越能代替"真的去干预"。
    最左边的 pair_distance 是对照:归因和键长的相关。
    """

    payload = _read_json(input_path)
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError(f"{input_path} does not contain a non-empty rows list")

    labels = [f'L{row["layer"]}:{row["channel"]}' for row in rows]
    values = [float(row["spearman_vs_abs_attr"]) for row in rows]
    colors = ["#7a7a7a" if row["channel"] == "pair_distance" else ("#2f6f9f" if row["channel"] == "message_norm" else "#b36b2c") for row in rows]

    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(labels, values, color=colors)
    ax.axhline(0.0, color="#222222", linewidth=0.8)
    ax.set_ylabel("Spearman vs |edge attribution|")
    ax.set_ylim(-1.05, 1.05)
    ax.set_title(_title(payload, "Internal indicators vs intervention"))
    ax.tick_params(axis="x", labelrotation=45, labelsize=8)
    from matplotlib.patches import Patch

    ax.legend(
        handles=[
            Patch(color="#7a7a7a", label="pair distance (control)"),
            Patch(color="#2f6f9f", label="message norm"),
            Patch(color="#b36b2c", label="l-channels"),
        ],
        fontsize=8,
    )
    fig.tight_layout()
    _save(fig, output_path)


def plot_channel_mechanism(input_path: Path, output_path: Path) -> None:
    """通道机理图:三个面板看"能量敏感性从哪些通道流过"。

    左:各通道敏感性份额;中:通道归因与总归因的相关;
    右:通道范数预测通道归因的相关(实心=原始,空心=控制键长后)。
    """

    payload = _read_json(input_path)
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError(f"{input_path} does not contain a non-empty rows list")

    labels = [f'L{row["layer"]}:{row["channel"]}' for row in rows]
    share = [float(row["sensitivity_share"]) for row in rows]
    rho_total = [float(row["spearman_attr_vs_total"]) for row in rows]
    rho_norm = [float(row["spearman_norm_vs_attr"]) for row in rows]
    rho_partial = [float(row["partial_spearman_norm_vs_attr"]) for row in rows]
    colors = ["#2f6f9f" if row["channel"] == "message_norm" else "#b36b2c" for row in rows]

    plt = _pyplot()
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))

    axes[0].bar(labels, share, color=colors)
    axes[0].set_ylabel("share of |attribution| sum")
    axes[0].set_title("Energy sensitivity per channel")
    axes[0].tick_params(axis="x", labelrotation=45, labelsize=8)

    axes[1].bar(labels, rho_total, color=colors)
    axes[1].axhline(0.0, color="#222222", linewidth=0.8)
    axes[1].set_ylabel("Spearman vs |total attribution|")
    axes[1].set_title("Channel attribution vs total")
    axes[1].tick_params(axis="x", labelrotation=45, labelsize=8)

    axes[2].bar(labels, rho_norm, color=colors, alpha=0.35, label="raw")
    axes[2].scatter(labels, rho_partial, color="#222222", marker="D", s=18, zorder=3, label="partial (control distance)")
    axes[2].axhline(0.0, color="#222222", linewidth=0.8)
    axes[2].set_ylabel("Spearman(norm, channel attribution)")
    axes[2].set_title("Norm predicts its own causal role?")
    axes[2].tick_params(axis="x", labelrotation=45, labelsize=8)
    axes[2].legend(fontsize=8)

    fig.suptitle(_title(payload, "Which angular channel carries the attribution?"))
    fig.tight_layout()
    _save(fig, output_path)


def plot_tilt_series(input_path: Path, output_path: Path) -> None:
    """倾斜系列基线图:能量和最大原子力随倾斜角的变化(两个模型各一条线)。"""

    payload = _read_json(input_path)
    series = payload.get("series")
    if not isinstance(series, dict) or not series:
        raise RuntimeError(f"{input_path} does not contain a non-empty series dict")

    plt = _pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for calculator, rows in series.items():
        angles = [float(row["tilt_deg"]) for row in rows]
        axes[0].plot(angles, [float(row["energy_per_atom"]) for row in rows], marker="o", label=calculator)
        axes[1].plot(angles, [float(row["max_force"]) for row in rows], marker="s", label=calculator)
    axes[0].set_xlabel("octahedral tilt (deg)")
    axes[0].set_ylabel("energy per atom (eV)")
    axes[0].set_title("Energy vs tilt")
    axes[0].legend()
    axes[1].set_xlabel("octahedral tilt (deg)")
    axes[1].set_ylabel("max atomic force (eV/Å)")
    axes[1].set_title("Force vs tilt")
    axes[1].legend()
    fig.suptitle("CsPbI3 octahedral tilt series (model baseline)")
    fig.tight_layout()
    _save(fig, output_path)
