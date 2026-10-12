from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from atomlens.attribution import integrated_force_attribution, summarize_atomic_values
from atomlens.channels import channel_summary
from atomlens.compare import compare_attribution_files, write_comparison
from atomlens.environment import collect_environment, write_json
from atomlens.export import write_attribution_structure
from atomlens.gating import (
    edge_ablation_importance,
    edge_gate_integrated_gradients,
    faithfulness_curve,
    pair_distances,
    rotation_consistency,
)
from atomlens.invariance import invariance_report
from atomlens.metrics import spearman_correlation
from atomlens.models import atomic_energies, calculator_by_name
from atomlens.plotting import (
    plot_attribution,
    plot_channels,
    plot_comparison,
    plot_convergence,
    plot_correlation,
    plot_channel_mechanism,
    plot_edge_attribution,
    plot_faithfulness,
    plot_stability,
    plot_tilt_series,
)
from atomlens.reporting import write_report
from atomlens.stability import stability_report
from atomlens.structures import (
    cspbi3_iodine_displacement,
    structure_by_name,
    structure_pair_by_name,
    tilted_cspbi3,
)
from atomlens.validation import validate_results, write_validation


def main() -> None:
    parser = argparse.ArgumentParser(prog="atomlens")
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke = subparsers.add_parser("smoke", help="Run a calculator-independent attribution smoke test")
    smoke.add_argument("--calculator", choices=["harmonic"], default="harmonic")
    smoke.add_argument("--steps", type=int, default=65)
    smoke.add_argument("--out", type=Path, help="Optional JSON output path")

    manifest = subparsers.add_parser("manifest", help="Write environment and package metadata")
    manifest.add_argument("--out", type=Path, default=Path("results/manifest.json"))

    model_smoke = subparsers.add_parser("model-smoke", help="Run a real pretrained calculator smoke test")
    model_smoke.add_argument("--calculator", choices=["mace-mp", "chgnet"], required=True)
    model_smoke.add_argument(
        "--structure",
        choices=["diamond_si", "cubic_cspbi3", "cspbi3_supercell", "cspbi3_tilted", "cspbi3_iodine_vacancy"],
        default="diamond_si",
    )
    model_smoke.add_argument("--device", default="cuda")
    model_smoke.add_argument("--out", type=Path, help="Optional JSON output path")

    attribute = subparsers.add_parser("attribute", help="Run path attribution on a supported structure pair")
    attribute.add_argument("--calculator", choices=["harmonic", "mace-mp", "chgnet"], required=True)
    attribute.add_argument(
        "--structure-pair",
        choices=["cspbi3_iodine_displacement", "cspbi3_tilt"],
        default="cspbi3_iodine_displacement",
    )
    attribute.add_argument("--steps", type=int, default=65)
    attribute.add_argument("--device", default="cuda")
    attribute.add_argument("--out", type=Path, help="Optional JSON output path")

    convergence = subparsers.add_parser("convergence", help="Run path attribution over multiple quadrature step counts")
    convergence.add_argument("--calculator", choices=["harmonic", "mace-mp", "chgnet"], required=True)
    convergence.add_argument(
        "--structure-pair",
        choices=["cspbi3_iodine_displacement", "cspbi3_tilt"],
        default="cspbi3_iodine_displacement",
    )
    convergence.add_argument("--steps", nargs="+", type=int, default=[17, 33, 65, 129])
    convergence.add_argument("--device", default="cuda")
    convergence.add_argument("--out", type=Path, help="Optional JSON output path")

    plot_attr = subparsers.add_parser("plot-attribution", help="Plot an attribution JSON file")
    plot_attr.add_argument("--input", type=Path, required=True)
    plot_attr.add_argument("--out", type=Path, required=True)

    plot_conv = subparsers.add_parser("plot-convergence", help="Plot a convergence JSON file")
    plot_conv.add_argument("--input", type=Path, required=True)
    plot_conv.add_argument("--out", type=Path, required=True)

    report = subparsers.add_parser("report", help="Write a Markdown report from attribution and convergence JSON files")
    report.add_argument("--attribution", type=Path, required=True)
    report.add_argument("--convergence", type=Path, required=True)
    report.add_argument("--out", type=Path, required=True)
    report.add_argument("--attribution-figure", type=Path)
    report.add_argument("--convergence-figure", type=Path)
    report.add_argument("--structure-file", type=Path)

    export_structure = subparsers.add_parser("export-structure", help="Write an extxyz structure with attribution as an atom array")
    export_structure.add_argument("--attribution", type=Path, required=True)
    export_structure.add_argument("--structure-pair", choices=["cspbi3_iodine_displacement"], default="cspbi3_iodine_displacement")
    export_structure.add_argument("--which", choices=["start", "end"], default="end")
    export_structure.add_argument("--out", type=Path, required=True)

    compare = subparsers.add_parser("compare-attributions", help="Compare two attribution JSON files")
    compare.add_argument("--left", type=Path, required=True)
    compare.add_argument("--right", type=Path, required=True)
    compare.add_argument("--out", type=Path, required=True)

    plot_compare = subparsers.add_parser("plot-comparison", help="Plot a comparison JSON file")
    plot_compare.add_argument("--input", type=Path, required=True)
    plot_compare.add_argument("--out", type=Path, required=True)

    validate = subparsers.add_parser("validate-results", help="Validate attribution and convergence result JSON files")
    validate.add_argument("--attribution", type=Path, required=True)
    validate.add_argument("--convergence", type=Path)
    validate.add_argument("--max-residual", type=float, default=1e-6)
    validate.add_argument("--out", type=Path, required=True)

    # ------------------------------------------------------------------
    # 第二阶段新增:不变性检验、原子能分解、边门控归因、忠实性、
    # 稳定性、通道分析、倾斜系列基线。
    # ------------------------------------------------------------------

    invariance = subparsers.add_parser("invariance", help="Check translation/rotation/permutation/supercell invariance")
    invariance.add_argument("--calculator", choices=["mace-mp", "chgnet"], required=True)
    invariance.add_argument(
        "--structure",
        choices=["diamond_si", "cubic_cspbi3", "cspbi3_supercell", "cspbi3_tilted", "cspbi3_iodine_vacancy"],
        default="cubic_cspbi3",
    )
    invariance.add_argument("--device", default="cuda")
    invariance.add_argument("--out", type=Path, help="Optional JSON output path")

    decompose = subparsers.add_parser("decompose", help="Decompose model energy onto atoms")
    decompose.add_argument("--calculator", choices=["mace-mp", "chgnet"], required=True)
    decompose.add_argument(
        "--structure",
        choices=["diamond_si", "cubic_cspbi3", "cspbi3_supercell", "cspbi3_tilted", "cspbi3_iodine_vacancy"],
        default="cspbi3_iodine_vacancy",
    )
    decompose.add_argument("--device", default="cuda")
    decompose.add_argument("--out", type=Path, help="Optional JSON output path")

    gating = subparsers.add_parser("gating", help="Edge gating attribution with integrated gradients")
    gating.add_argument("--calculator", choices=["mace-mp", "chgnet"], default="mace-mp")
    gating.add_argument(
        "--structure",
        choices=["cspbi3_supercell", "cspbi3_tilted", "cspbi3_iodine_vacancy"],
        default="cspbi3_iodine_vacancy",
    )
    gating.add_argument("--steps", type=int, default=32, help="IG integration steps")
    gating.add_argument("--with-ablation", action="store_true", help="Also run leave-one-out ablation and correlate")
    gating.add_argument("--device", default="cuda")
    gating.add_argument("--out", type=Path, help="Optional JSON output path")

    faithful = subparsers.add_parser("faithfulness", help="Top-k gating energy-change curve vs random/bottom controls")
    faithful.add_argument("--calculator", choices=["mace-mp"], default="mace-mp")
    faithful.add_argument(
        "--structure",
        choices=["cspbi3_supercell", "cspbi3_tilted", "cspbi3_iodine_vacancy"],
        default="cspbi3_iodine_vacancy",
    )
    faithful.add_argument("--ks", nargs="+", type=int, default=[5, 10, 20, 40, 80])
    faithful.add_argument("--n-random", type=int, default=5)
    faithful.add_argument("--device", default="cuda")
    faithful.add_argument("--out", type=Path, help="Optional JSON output path")

    stability = subparsers.add_parser("stability", help="Attribution stability under small thermal noise")
    stability.add_argument("--calculator", choices=["mace-mp", "chgnet"], required=True)
    stability.add_argument(
        "--structure",
        choices=["cspbi3_supercell", "cspbi3_tilted", "cspbi3_iodine_vacancy"],
        default="cspbi3_iodine_vacancy",
    )
    stability.add_argument("--what", choices=["edges", "atoms"], default="edges")
    stability.add_argument("--sigma", type=float, default=0.03)
    stability.add_argument("--n-noises", type=int, default=5)
    stability.add_argument("--topk", type=int, default=10)
    stability.add_argument("--steps", type=int, default=24, help="IG steps when --what edges")
    stability.add_argument("--device", default="cuda")
    stability.add_argument("--out", type=Path, help="Optional JSON output path")

    channels = subparsers.add_parser("channels", help="MACE l-channel norms along the octahedral tilt series")
    channels.add_argument("--calculator", choices=["mace-mp"], default="mace-mp")
    channels.add_argument("--angles", nargs="+", type=float, default=[0.0, 4.0, 8.0, 12.0])
    channels.add_argument("--device", default="cuda")
    channels.add_argument("--out", type=Path, help="Optional JSON output path")

    tilt = subparsers.add_parser("tilt-series", help="Energy/force vs octahedral tilt for one or more models")
    tilt.add_argument("--calculators", nargs="+", choices=["mace-mp", "chgnet"], default=["mace-mp", "chgnet"])
    tilt.add_argument("--angles", nargs="+", type=float, default=[0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0])
    tilt.add_argument("--device", default="cuda")
    tilt.add_argument("--out", type=Path, help="Optional JSON output path")

    plot_edges = subparsers.add_parser("plot-edge-attribution", help="Plot an edge gating attribution JSON file")
    plot_edges.add_argument("--input", type=Path, required=True)
    plot_edges.add_argument("--out", type=Path, required=True)

    plot_faith = subparsers.add_parser("plot-faithfulness", help="Plot a faithfulness curve JSON file")
    plot_faith.add_argument("--input", type=Path, required=True)
    plot_faith.add_argument("--out", type=Path, required=True)

    plot_stab = subparsers.add_parser("plot-stability", help="Plot a stability JSON file")
    plot_stab.add_argument("--input", type=Path, required=True)
    plot_stab.add_argument("--out", type=Path, required=True)

    plot_chan = subparsers.add_parser("plot-channels", help="Plot a channel analysis JSON file")
    plot_chan.add_argument("--input", type=Path, required=True)
    plot_chan.add_argument("--out", type=Path, required=True)

    plot_tilt = subparsers.add_parser("plot-tilt-series", help="Plot a tilt series JSON file")
    plot_tilt.add_argument("--input", type=Path, required=True)
    plot_tilt.add_argument("--out", type=Path, required=True)

    # ------------------------------------------------------------------
    # 第三期新增:差分边门控、形成能、内部指标-干预指标相关性。
    # ------------------------------------------------------------------

    diff_gating = subparsers.add_parser("diff-gating", help="Differential edge gating: vacancy vs perfect supercell")
    diff_gating.add_argument("--calculator", choices=["mace-mp", "chgnet"], required=True)
    diff_gating.add_argument("--n", type=int, default=2, help="supercell size n x n x n")
    diff_gating.add_argument("--steps", type=int, default=32, help="IG steps")
    diff_gating.add_argument("--ks", nargs="+", type=int, default=[5, 10, 20, 40, 80])
    diff_gating.add_argument("--n-random", type=int, default=5)
    diff_gating.add_argument("--device", default="cuda")
    diff_gating.add_argument("--out", type=Path, help="Optional JSON output path")

    formation = subparsers.add_parser("formation", help="Iodine vacancy formation energy with I2 chemical potential")
    formation.add_argument("--calculators", nargs="+", choices=["mace-mp", "chgnet"], default=["mace-mp", "chgnet"])
    formation.add_argument("--n", nargs="+", type=int, default=[2, 4], help="supercell sizes")
    formation.add_argument("--device", default="cuda")
    formation.add_argument("--out", type=Path, help="Optional JSON output path")

    correlation = subparsers.add_parser("correlation", help="Internal message indicators vs interventional attribution (MACE)")
    correlation.add_argument("--n", type=int, default=2, help="supercell size n x n x n")
    correlation.add_argument("--vacancy", action="store_true", help="use the vacancy structure instead of the perfect one")
    correlation.add_argument("--steps", type=int, default=32)
    correlation.add_argument("--device", default="cuda")
    correlation.add_argument("--out", type=Path, help="Optional JSON output path")

    plot_corr = subparsers.add_parser("plot-correlation", help="Plot a correlation JSON file")
    plot_corr.add_argument("--input", type=Path, required=True)
    plot_corr.add_argument("--out", type=Path, required=True)

    channel_gating = subparsers.add_parser(
        "channel-gating", help="Channel-resolved edge gating: which angular channel carries the energy sensitivity (MACE)"
    )
    channel_gating.add_argument("--n", type=int, default=2, help="supercell size n x n x n")
    channel_gating.add_argument("--vacancy", action="store_true", help="use the vacancy structure")
    channel_gating.add_argument("--steps", type=int, default=24)
    channel_gating.add_argument("--device", default="cuda")
    channel_gating.add_argument("--out", type=Path, help="Optional JSON output path")

    plot_channel = subparsers.add_parser("plot-channel-mechanism", help="Plot a channel-gating JSON file")
    plot_channel.add_argument("--input", type=Path, required=True)
    plot_channel.add_argument("--out", type=Path, required=True)

    args = parser.parse_args()
    try:
        if args.command == "smoke":
            run_smoke(steps=args.steps, out=args.out)
        elif args.command == "manifest":
            print(write_json(collect_environment(), args.out))
        elif args.command == "model-smoke":
            run_model_smoke(
                calculator=args.calculator,
                structure=args.structure,
                device=args.device,
                out=args.out,
            )
        elif args.command == "attribute":
            run_attribute(
                calculator=args.calculator,
                structure_pair=args.structure_pair,
                steps=args.steps,
                device=args.device,
                out=args.out,
            )
        elif args.command == "convergence":
            run_convergence(
                calculator=args.calculator,
                structure_pair=args.structure_pair,
                steps=args.steps,
                device=args.device,
                out=args.out,
            )
        elif args.command == "plot-attribution":
            plot_attribution(args.input, args.out)
        elif args.command == "plot-convergence":
            plot_convergence(args.input, args.out)
        elif args.command == "report":
            write_report(
                attribution_path=args.attribution,
                convergence_path=args.convergence,
                output_path=args.out,
                attribution_figure=args.attribution_figure,
                convergence_figure=args.convergence_figure,
                structure_file=args.structure_file,
            )
        elif args.command == "export-structure":
            start, end = _structure_pair_by_name(args.structure_pair)
            atoms = start if args.which == "start" else end
            write_attribution_structure(atoms, args.attribution, args.out)
        elif args.command == "compare-attributions":
            write_comparison(compare_attribution_files(args.left, args.right), args.out)
        elif args.command == "plot-comparison":
            plot_comparison(args.input, args.out)
        elif args.command == "validate-results":
            write_validation(
                validate_results(
                    attribution_path=args.attribution,
                    convergence_path=args.convergence,
                    max_residual=args.max_residual,
                ),
                args.out,
            )
        elif args.command == "invariance":
            run_invariance(
                calculator=args.calculator,
                structure=args.structure,
                device=args.device,
                out=args.out,
            )
        elif args.command == "decompose":
            run_decompose(
                calculator=args.calculator,
                structure=args.structure,
                device=args.device,
                out=args.out,
            )
        elif args.command == "gating":
            run_gating(
                calculator=args.calculator,
                structure=args.structure,
                steps=args.steps,
                with_ablation=args.with_ablation,
                device=args.device,
                out=args.out,
            )
        elif args.command == "faithfulness":
            run_faithfulness(
                structure=args.structure,
                ks=args.ks,
                n_random=args.n_random,
                device=args.device,
                out=args.out,
            )
        elif args.command == "stability":
            run_stability(
                calculator=args.calculator,
                structure=args.structure,
                what=args.what,
                sigma=args.sigma,
                n_noises=args.n_noises,
                topk=args.topk,
                steps=args.steps,
                device=args.device,
                out=args.out,
            )
        elif args.command == "channels":
            run_channels(angles=args.angles, device=args.device, out=args.out)
        elif args.command == "tilt-series":
            run_tilt_series(
                calculators=args.calculators,
                angles=args.angles,
                device=args.device,
                out=args.out,
            )
        elif args.command == "plot-edge-attribution":
            plot_edge_attribution(args.input, args.out)
        elif args.command == "diff-gating":
            run_diff_gating(
                calculator=args.calculator,
                n_supercell=args.n,
                steps=args.steps,
                ks=args.ks,
                n_random=args.n_random,
                device=args.device,
                out=args.out,
            )
        elif args.command == "formation":
            run_formation(
                calculators=args.calculators,
                sizes=args.n,
                device=args.device,
                out=args.out,
            )
        elif args.command == "correlation":
            run_correlation(
                n_supercell=args.n,
                vacancy=args.vacancy,
                steps=args.steps,
                device=args.device,
                out=args.out,
            )
        elif args.command == "plot-correlation":
            plot_correlation(args.input, args.out)
        elif args.command == "channel-gating":
            run_channel_gating(
                n_supercell=args.n,
                vacancy=args.vacancy,
                steps=args.steps,
                device=args.device,
                out=args.out,
            )
        elif args.command == "plot-channel-mechanism":
            plot_channel_mechanism(args.input, args.out)
        elif args.command == "plot-faithfulness":
            plot_faithfulness(args.input, args.out)
        elif args.command == "plot-stability":
            plot_stability(args.input, args.out)
        elif args.command == "plot-channels":
            plot_channels(args.input, args.out)
        elif args.command == "plot-tilt-series":
            plot_tilt_series(args.input, args.out)
    except RuntimeError as exc:
        parser.exit(2, f"ERROR: {exc}\n")


def run_smoke(steps: int, out: Path | None = None) -> None:
    start, end = cspbi3_iodine_displacement()
    reference = start.get_positions()

    result = integrated_force_attribution(
        start,
        end,
        calculator_factory=lambda: HarmonicCalculator(reference_positions=reference),
        n_steps=steps,
    )
    payload = {
        "calculator": "harmonic",
        "structure_pair": "cspbi3_iodine_displacement",
        "steps": steps,
        "energy_start": result.energy_start,
        "energy_end": result.energy_end,
        "energy_delta": result.energy_delta,
        "attribution_total": result.total,
        "residual": result.residual,
        "ranking": summarize_atomic_values(start.get_chemical_symbols(), result.atomic),
    }
    print(write_json(payload, out))


def run_model_smoke(calculator: str, structure: str, device: str, out: Path | None = None) -> None:
    atoms = structure_by_name(structure)
    atoms.calc = calculator_by_name(calculator, device=device)

    forces = atoms.get_forces()
    payload = {
        "calculator": calculator,
        "structure": structure,
        "device": device,
        "energy": float(atoms.get_potential_energy()),
        "max_force_norm": float((forces * forces).sum(axis=1).max() ** 0.5),
        "environment": collect_environment(),
    }
    print(write_json(payload, out))


def run_attribute(
    calculator: str,
    structure_pair: str,
    steps: int,
    device: str,
    out: Path | None = None,
) -> None:
    start, end = _structure_pair_by_name(structure_pair)
    calculator_factory = _calculator_factory_for_path(calculator, device, start)

    result = integrated_force_attribution(
        start,
        end,
        calculator_factory=calculator_factory,
        n_steps=steps,
    )
    payload = {
        "calculator": calculator,
        "structure_pair": structure_pair,
        "steps": steps,
        "device": device,
        "energy_start": result.energy_start,
        "energy_end": result.energy_end,
        "energy_delta": result.energy_delta,
        "attribution_total": result.total,
        "residual": result.residual,
        "ranking": summarize_atomic_values(start.get_chemical_symbols(), result.atomic),
        "environment": collect_environment(),
    }
    print(write_json(payload, out))


def run_convergence(
    calculator: str,
    structure_pair: str,
    steps: list[int],
    device: str,
    out: Path | None = None,
) -> None:
    start, end = _structure_pair_by_name(structure_pair)
    calculator_factory = _calculator_factory_for_path(calculator, device, start)

    rows = []
    for step_count in steps:
        result = integrated_force_attribution(
            start,
            end,
            calculator_factory=calculator_factory,
            n_steps=step_count,
        )
        rows.append(
            {
                "steps": step_count,
                "energy_delta": result.energy_delta,
                "attribution_total": result.total,
                "residual": result.residual,
                "max_abs_atomic": float(max(abs(value) for value in result.atomic)),
            }
        )

    payload = {
        "calculator": calculator,
        "structure_pair": structure_pair,
        "device": device,
        "steps": steps,
        "rows": rows,
        "environment": collect_environment(),
    }
    print(write_json(payload, out))


def _structure_by_name(name: str):
    return structure_by_name(name)


def _structure_pair_by_name(name: str):
    return structure_pair_by_name(name)


def _cached_calculator_factory(calculator: str, device: str):
    cached = None

    def factory():
        nonlocal cached
        if cached is None:
            cached = calculator_by_name(calculator, device=device)
        return cached

    return factory


def _calculator_factory_for_path(calculator: str, device: str, start):
    if calculator == "harmonic":
        reference = start.get_positions()
        return lambda: HarmonicCalculator(reference_positions=reference)
    return _cached_calculator_factory(calculator, device)


# ---------------------------------------------------------------------------
# 第二阶段新增命令的实现。
# ---------------------------------------------------------------------------


def run_invariance(calculator: str, structure: str, device: str, out: Path | None = None) -> None:
    """检验模型对平移/旋转/重编号/超胞这四种"不该变"的变换是否真的不变。"""

    atoms = structure_by_name(structure)
    report = invariance_report(atoms, lambda: calculator_by_name(calculator, device=device))
    payload = {
        "calculator": calculator,
        "structure": structure,
        "device": device,
        "report": report,
        "environment": collect_environment(),
    }
    print(write_json(payload, out))


def run_decompose(calculator: str, structure: str, device: str, out: Path | None = None) -> None:
    """原子能分解:看模型把总能量分配给了哪些原子的局域环境。"""

    atoms = structure_by_name(structure)
    calc = calculator_by_name(calculator, device=device)
    energies = atomic_energies(calc, atoms)

    # 一致性检查:原子能求和应该等于模型总能量
    atoms.calc = calc
    total_energy = float(atoms.get_potential_energy())
    sum_total = float(np.sum(energies["total"]))

    payload = {
        "calculator": calculator,
        "structure": structure,
        "device": device,
        "energy_total": total_energy,
        "atomic_sum": sum_total,
        "decomposition_residual": sum_total - total_energy,
        "ranking": summarize_atomic_values(atoms.get_chemical_symbols(), energies["interaction"]),
        "environment": collect_environment(),
    }
    print(write_json(payload, out))


def run_gating(
    calculator: str,
    structure: str,
    steps: int,
    with_ablation: bool,
    device: str,
    out: Path | None = None,
) -> None:
    """边门控积分梯度归因。MACE 用消息门控,CHGNet 用键权重门控。"""

    from atomlens.chgnet_gating import (
        chgnet_bond_info,
        chgnet_edge_ablation_importance,
        chgnet_edge_gate_integrated_gradients,
    )

    atoms = structure_by_name(structure)
    if calculator == "mace-mp":
        calc = calculator_by_name("mace-mp", device=device, float64=True)
        result = edge_gate_integrated_gradients(calc, atoms, n_steps=steps)
        idx_i, idx_j, distances = pair_distances(calc, atoms, result["pair_ids"])
        attribution = result["attribution"]
        rotation = rotation_consistency(calc, atoms)
    else:
        calc = calculator_by_name("chgnet", device=device)
        result = chgnet_edge_gate_integrated_gradients(calc, atoms, n_steps=steps)
        info = chgnet_bond_info(calc, atoms)
        idx_i, idx_j, distances = info["idx_i"], info["idx_j"], info["distances"]
        attribution = result["attribution"]
        rotation = None  # CHGNet 的旋转一致性检验暂未实现(键集合随旋转重排更复杂)

    order = np.argsort(-np.abs(attribution))
    pairs = [
        {
            "pair": int(pair),
            "i": int(idx_i[pair]),
            "j": int(idx_j[pair]),
            "distance": float(distances[pair]),
            "value": float(attribution[pair]),
        }
        for pair in order
    ]

    payload = {
        "calculator": calculator,
        "structure": structure,
        "method": "edge-gating-integrated-gradients",
        "steps": steps,
        "device": device,
        "n_pairs": result.get("n_pairs", result.get("n_bonds")),
        "n_edges": result.get("n_edges", result.get("n_pairs", result.get("n_bonds"))),
        "energy_full": result["energy_full"],
        "energy_all_closed": result["energy_closed"],
        "completeness_residual": result["completeness_residual"],
        "has_pair_repulsion": result.get("has_pair_repulsion", False),
        "pairs": pairs,
        "environment": collect_environment(),
    }
    if rotation is not None:
        payload["rotation_check"] = rotation

    if with_ablation:
        if calculator == "mace-mp":
            ablation = edge_ablation_importance(calc, atoms)
        else:
            ablation = chgnet_edge_ablation_importance(calc, atoms)
        payload["ablation_spearman"] = spearman_correlation(attribution, ablation["importance"])
        payload["ablation_total"] = float(np.sum(ablation["importance"]))
    print(write_json(payload, out))


def run_faithfulness(
    structure: str,
    ks: list[int],
    n_random: int,
    device: str,
    out: Path | None = None,
) -> None:
    """忠实性检验:先算边归因,再按排名关边,看能量变化是否显著更大。"""

    atoms = structure_by_name(structure)
    calc = calculator_by_name("mace-mp", device=device, float64=True)
    result = edge_gate_integrated_gradients(calc, atoms, n_steps=24)
    curve = faithfulness_curve(calc, atoms, result["attribution"], ks=ks, n_random=n_random)

    payload = {
        "calculator": "mace-mp",
        "structure": structure,
        "device": device,
        "n_pairs": curve["n_pairs"],
        "energy_full": curve["energy_full"],
        "rows": curve["rows"],
        "environment": collect_environment(),
    }
    print(write_json(payload, out))


def run_stability(
    calculator: str,
    structure: str,
    what: str,
    sigma: float,
    n_noises: int,
    topk: int,
    steps: int,
    device: str,
    out: Path | None = None,
) -> None:
    """稳定性检验:小幅热扰动下,归因排序保持吗?"""

    atoms = structure_by_name(structure)
    calc = calculator_by_name(calculator, device=device, float64=(calculator == "mace-mp"))

    if what == "edges":
        if calculator != "mace-mp":
            raise RuntimeError("edge gating attribution is only implemented for mace-mp")

        def attributor(current_atoms):
            result = edge_gate_integrated_gradients(calc, current_atoms, n_steps=steps)
            idx_i, idx_j, _ = pair_distances(calc, current_atoms, result["pair_ids"])
            keys = [(int(idx_i[p]), int(idx_j[p])) for p in range(result["n_pairs"])]
            return keys, result["attribution"]

    else:

        def attributor(current_atoms):
            energies = atomic_energies(calc, current_atoms)
            values = energies["interaction"]
            if calculator == "chgnet":
                values = values * len(current_atoms)
            return list(range(len(current_atoms))), values

    report = stability_report(atoms, attributor, n_noises=n_noises, sigma=sigma, topk=topk)
    payload = {
        "calculator": calculator,
        "structure": structure,
        "what": what,
        "device": device,
        "report": report,
        "environment": collect_environment(),
    }
    print(write_json(payload, out))


def run_channels(angles: list[float], device: str, out: Path | None = None) -> None:
    """通道分析:八面体倾斜角从 0 加大时,l=0/1/2 通道的活跃度怎么变。"""

    calc = calculator_by_name("mace-mp", device=device, float64=True)
    rows = []
    for angle in angles:
        atoms = tilted_cspbi3(tilt_deg=angle)
        summary = channel_summary(calc, atoms)
        summary["tilt_deg"] = float(angle)
        rows.append(summary)
        print(f"tilt {angle:.1f} deg done", flush=True)

    payload = {
        "calculator": "mace-mp",
        "structure": "cspbi3_tilt_series",
        "device": device,
        "rows": rows,
        "environment": collect_environment(),
    }
    print(write_json(payload, out))


def run_tilt_series(calculators: list[str], angles: list[float], device: str, out: Path | None = None) -> None:
    """倾斜系列基线:能量/最大力随倾斜角的变化,模型间可以对照。"""

    series: dict[str, list] = {}
    for calculator in calculators:
        calc = calculator_by_name(calculator, device=device)
        rows = []
        for angle in angles:
            atoms = tilted_cspbi3(tilt_deg=angle)
            atoms.calc = calc
            energy = float(atoms.get_potential_energy())
            forces = np.asarray(atoms.get_forces(), dtype=float)
            rows.append(
                {
                    "tilt_deg": float(angle),
                    "energy_per_atom": energy / len(atoms),
                    "max_force": float(np.max(np.linalg.norm(forces, axis=1))),
                }
            )
        series[calculator] = rows
        print(f"{calculator} done", flush=True)

    payload = {
        "structure": "cspbi3_tilt_series",
        "angles": angles,
        "device": device,
        "series": series,
        "environment": collect_environment(),
    }
    print(write_json(payload, out))


# ---------------------------------------------------------------------------
# 第三期新增命令的实现:差分门控、形成能、相关性。
# ---------------------------------------------------------------------------


def run_diff_gating(
    calculator: str,
    n_supercell: int,
    steps: int,
    ks: list[int],
    n_random: int,
    device: str,
    out: Path | None = None,
) -> None:
    """差分边门控:空位 vs 完美,归因差值 + 忠实性检验。"""

    if calculator == "mace-mp":
        from atomlens.differential import differential_edge_attribution, differential_faithfulness

        calc = calculator_by_name("mace-mp", device=device, float64=True)
        diff = differential_edge_attribution(calc, n_supercell=n_supercell, n_steps=steps)
        faith = differential_faithfulness(
            calc, n_supercell=n_supercell, ranking_rows=diff["pairs"], ks=ks, n_random=n_random
        )
        method = "edge-gating-IG, vacancy minus perfect"
    else:
        from atomlens.chgnet_gating import chgnet_differential_edge_attribution, chgnet_differential_faithfulness

        calc = calculator_by_name("chgnet", device=device)
        diff = chgnet_differential_edge_attribution(calc, n_supercell=n_supercell, n_steps=steps)
        faith = chgnet_differential_faithfulness(
            calc, n_supercell=n_supercell, ranking_rows=diff["pairs"], ks=ks, n_random=n_random
        )
        method = "chgnet edge-gating-IG, vacancy minus perfect"

    payload = {
        "calculator": calculator,
        "structure": f"cspbi3_iodine_vacancy_{n_supercell}x{n_supercell}x{n_supercell}_vs_perfect",
        "method": method,
        "steps": steps,
        "device": device,
        **diff,
        "faithfulness": faith,
        "environment": collect_environment(),
    }
    print(write_json(payload, out))


def run_formation(calculators: list[str], sizes: list[int], device: str, out: Path | None = None) -> None:
    """碘空位形成能(带 I2 化学势),多个模型、多种超胞。"""

    from atomlens.differential import formation_energy

    results = {}
    for calculator in calculators:
        calc = calculator_by_name(calculator, device=device)
        results[calculator] = [formation_energy(calc, n_supercell=n) for n in sizes]
        print(f"{calculator} done", flush=True)

    payload = {
        "structure": "cspbi3_iodine_vacancy",
        "chemical_potential": "mu_I = E(I2)/2 (iodine-rich limit)",
        "device": device,
        "results": results,
        "environment": collect_environment(),
    }
    print(write_json(payload, out))


def run_correlation(n_supercell: int, vacancy: bool, steps: int, device: str, out: Path | None = None) -> None:
    """内部指标(消息范数/l通道)vs 干预指标(门控归因)的逐边相关。"""

    from atomlens.correlation import correlation_report
    from atomlens.structures import cspbi3_iodine_vacancy, cspbi3_supercell

    if vacancy:
        atoms = cspbi3_iodine_vacancy(n=n_supercell)
        structure_name = f"cspbi3_iodine_vacancy_{n_supercell}x{n_supercell}x{n_supercell}"
    else:
        atoms = cspbi3_supercell(n=n_supercell)
        structure_name = f"cspbi3_supercell_{n_supercell}x{n_supercell}x{n_supercell}"

    calc = calculator_by_name("mace-mp", device=device, float64=True)
    report = correlation_report(calc, atoms, n_steps=steps)
    payload = {
        "calculator": "mace-mp",
        "structure": structure_name,
        "steps": steps,
        "device": device,
        **report,
        "environment": collect_environment(),
    }
    print(write_json(payload, out))


def run_channel_gating(n_supercell: int, vacancy: bool, steps: int, device: str, out: Path | None = None) -> None:
    """通道级门控:每个角动量通道单独积分梯度,回答"能量敏感性从哪些通道流过"。

    同时给出三层机理证据:
    1. 各通道的能量敏感性份额(|A_c| 占比);
    2. 通道归因和总归因的相关(哪个通道"解释"了总归因);
    3. 通道范数预测通道归因的相关,以及控制键长后的偏相关
       (说明通道信息是否超出"键长"这一 trivial 解释)。
    """

    from atomlens.channel_gating import channel_integrated_gradients
    from atomlens.channels import mace_message_channel_norms
    from atomlens.gating import _batch_and_pair_info
    from atomlens.metrics import partial_spearman, spearman_correlation
    from atomlens.structures import cspbi3_iodine_vacancy, cspbi3_supercell

    if vacancy:
        atoms = cspbi3_iodine_vacancy(n=n_supercell)
        structure_name = f"cspbi3_iodine_vacancy_{n_supercell}x{n_supercell}x{n_supercell}"
    else:
        atoms = cspbi3_supercell(n=n_supercell)
        structure_name = f"cspbi3_supercell_{n_supercell}x{n_supercell}x{n_supercell}"

    calc = calculator_by_name("mace-mp", device=device, float64=True)
    gate_result = channel_integrated_gradients(calc, atoms, n_steps=steps)
    channel_result = mace_message_channel_norms(calc, atoms)

    data, pair_ids = _batch_and_pair_info(calc, atoms)
    if not np.array_equal(pair_ids, gate_result["pair_ids"]):
        raise RuntimeError("pair ids changed between runs, cannot align channel norms")

    def per_pair(per_edge_values: np.ndarray) -> np.ndarray:
        """每条有向边的量 -> 每个原子对的平均(正反两条边取平均)。"""

        sums = np.zeros(gate_result["n_pairs"])
        counts = np.zeros(gate_result["n_pairs"])
        np.add.at(sums, pair_ids, per_edge_values)
        np.add.at(counts, pair_ids, 1.0)
        return sums / np.clip(counts, 1, None)

    distances = gate_result["distances"]
    abs_total = np.abs(gate_result["attribution_total"])
    channels = gate_result["channels"]
    rows = []
    for layer_index, layer in enumerate(channel_result["layers"]):
        for name in channels:
            attr_c = np.abs(gate_result["attribution_per_channel"][name])
            norm_c = per_pair(layer["edge_l_norms"][int(name[1:])])
            share = float(attr_c.sum() / max(abs_total.sum(), 1e-30))
            rows.append(
                {
                    "layer": layer_index,
                    "channel": name,
                    "sensitivity_share": share,
                    "spearman_attr_vs_total": spearman_correlation(attr_c, abs_total),
                    "spearman_norm_vs_attr": spearman_correlation(norm_c, attr_c),
                    "partial_spearman_norm_vs_attr": partial_spearman(norm_c, attr_c, distances),
                }
            )

    payload = {
        "calculator": "mace-mp",
        "structure": structure_name,
        "steps": steps,
        "device": device,
        "n_pairs": gate_result["n_pairs"],
        "energy_full": gate_result["energy_full"],
        "energy_closed": gate_result["energy_closed"],
        "completeness": gate_result["completeness"],
        "rows": rows,
        "pairs": [
            {
                "i": int(gate_result["idx_i"][p]),
                "j": int(gate_result["idx_j"][p]),
                "distance": float(distances[p]),
                **{f"attr_{name}": float(gate_result["attribution_per_channel"][name][p]) for name in channels},
                "attr_total": float(gate_result["attribution_total"][p]),
            }
            for p in range(gate_result["n_pairs"])
        ],
        "environment": collect_environment(),
    }
    print(write_json(payload, out))


if __name__ == "__main__":
    main()
