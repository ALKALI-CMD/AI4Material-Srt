from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def compare_attribution_files(left_path: Path, right_path: Path) -> dict[str, Any]:
    left = _read_json(left_path)
    right = _read_json(right_path)
    left_values = attribution_values(left)
    right_values = attribution_values(right)
    if set(left_values) != set(right_values):
        missing_left = sorted(set(right_values) - set(left_values))
        missing_right = sorted(set(left_values) - set(right_values))
        raise RuntimeError(f"atom index mismatch: missing_left={missing_left}, missing_right={missing_right}")

    indices = sorted(left_values)
    left_array = np.array([left_values[index] for index in indices], dtype=float)
    right_array = np.array([right_values[index] for index in indices], dtype=float)
    rows = [
        {
            "index": index,
            "left": float(left_values[index]),
            "right": float(right_values[index]),
            "difference": float(left_values[index] - right_values[index]),
            "abs_difference": float(abs(left_values[index] - right_values[index])),
        }
        for index in indices
    ]
    rows.sort(key=lambda row: row["abs_difference"], reverse=True)

    return {
        "left_calculator": left.get("calculator", "left"),
        "right_calculator": right.get("calculator", "right"),
        "structure_pair": left.get("structure_pair") or right.get("structure_pair"),
        "atom_count": len(indices),
        "pearson": _pearson(left_array, right_array),
        "spearman": _pearson(_rank(left_array), _rank(right_array)),
        "sign_agreement": _sign_agreement(left_array, right_array),
        "left_energy_delta": left.get("energy_delta"),
        "right_energy_delta": right.get("energy_delta"),
        "rows": rows,
    }


def write_comparison(payload: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def attribution_values(payload: dict[str, Any]) -> dict[int, float]:
    ranking = payload.get("ranking")
    if not isinstance(ranking, list):
        raise RuntimeError("attribution payload does not contain a ranking list")
    values = {}
    for row in ranking:
        index = int(row["index"])
        if index in values:
            raise RuntimeError(f"duplicate atom index {index}")
        values[index] = float(row["value"])
    return values


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 2 or np.allclose(left, left[0]) or np.allclose(right, right[0]):
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    return ranks


def _sign_agreement(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.mean(np.sign(left) == np.sign(right)))
