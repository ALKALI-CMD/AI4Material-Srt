from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def validate_results(
    attribution_path: Path,
    convergence_path: Path | None = None,
    max_residual: float = 1e-6,
) -> dict[str, Any]:
    checks = []
    attribution = _read_json(attribution_path)
    checks.append(_check_abs("attribution_residual", attribution.get("residual"), max_residual))

    ranking = attribution.get("ranking")
    checks.append(
        {
            "name": "attribution_ranking_nonempty",
            "passed": isinstance(ranking, list) and len(ranking) > 0,
            "value": len(ranking) if isinstance(ranking, list) else None,
            "threshold": 1,
        }
    )

    if convergence_path is not None:
        convergence = _read_json(convergence_path)
        rows = convergence.get("rows")
        rows_ok = isinstance(rows, list) and len(rows) > 0
        checks.append(
            {
                "name": "convergence_rows_nonempty",
                "passed": rows_ok,
                "value": len(rows) if isinstance(rows, list) else None,
                "threshold": 1,
            }
        )
        if rows_ok:
            max_seen = max(abs(float(row.get("residual", float("inf")))) for row in rows)
            checks.append(_check_abs("convergence_max_residual", max_seen, max_residual))

    return {
        "passed": all(bool(check["passed"]) for check in checks),
        "max_residual": max_residual,
        "checks": checks,
    }


def write_validation(payload: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _check_abs(name: str, value: Any, threshold: float) -> dict[str, Any]:
    if value is None:
        return {"name": name, "passed": False, "value": None, "threshold": threshold}
    numeric = abs(float(value))
    return {"name": name, "passed": numeric <= threshold, "value": numeric, "threshold": threshold}
