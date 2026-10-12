from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_report(
    attribution_path: Path,
    convergence_path: Path,
    output_path: Path,
    attribution_figure: Path | None = None,
    convergence_figure: Path | None = None,
    structure_file: Path | None = None,
) -> None:
    attribution = _read_json(attribution_path)
    convergence = _read_json(convergence_path)
    lines = [
        "# Atomlens Phase 1 Report",
        "",
        "## Run Summary",
        "",
        f"- Calculator: `{attribution.get('calculator', 'unknown')}`",
        f"- Structure path: `{attribution.get('structure_pair', 'unknown')}`",
        f"- Attribution steps: `{attribution.get('steps', 'unknown')}`",
        f"- Energy delta: `{_fmt(attribution.get('energy_delta'))}`",
        f"- Attribution total: `{_fmt(attribution.get('attribution_total'))}`",
        f"- Completeness residual: `{_fmt(attribution.get('residual'))}`",
        "",
        "## Top Atomic Contributions",
        "",
        "| rank | atom | attribution |",
        "| ---: | --- | ---: |",
    ]

    for rank, row in enumerate(attribution.get("ranking", [])[:10], start=1):
        atom = f"{row.get('symbol', '?')}{row.get('index', '?')}"
        lines.append(f"| {rank} | `{atom}` | `{_fmt(row.get('value'))}` |")

    lines.extend(
        [
            "",
            "## Convergence",
            "",
            "| steps | energy delta | attribution total | residual | max abs atomic |",
            "| ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in convergence.get("rows", []):
        lines.append(
            "| {steps} | `{energy}` | `{total}` | `{residual}` | `{max_atomic}` |".format(
                steps=row.get("steps"),
                energy=_fmt(row.get("energy_delta")),
                total=_fmt(row.get("attribution_total")),
                residual=_fmt(row.get("residual")),
                max_atomic=_fmt(row.get("max_abs_atomic")),
            )
        )

    lines.extend(["", "## Figures", ""])
    if attribution_figure is not None:
        lines.append(f"- Attribution plot: `{_path_text(attribution_figure)}`")
    if convergence_figure is not None:
        lines.append(f"- Convergence plot: `{_path_text(convergence_figure)}`")
    if structure_file is not None:
        lines.append(f"- Structure with attribution: `{_path_text(structure_file)}`")
    if attribution_figure is None and convergence_figure is None:
        lines.append("- No figure paths were provided.")

    environment = attribution.get("environment") or convergence.get("environment") or {}
    packages = environment.get("packages", {})
    lines.extend(["", "## Environment", ""])
    if packages:
        for name in sorted(packages):
            lines.append(f"- `{name}`: `{packages[name]}`")
    else:
        lines.append("- Environment metadata is not available.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.8e}"
    return str(value)


def _path_text(path: Path) -> str:
    return path.as_posix()
