from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.io import write


def write_attribution_structure(atoms: Atoms, attribution_path: Path, output_path: Path) -> None:
    payload = json.loads(attribution_path.read_text(encoding="utf-8"))
    atoms = atoms.copy()
    atoms.arrays["atomlens_attribution"] = attribution_array(payload, len(atoms))
    atoms.info["atomlens_calculator"] = str(payload.get("calculator", "unknown"))
    atoms.info["atomlens_structure_pair"] = str(payload.get("structure_pair", "unknown"))
    atoms.info["atomlens_energy_delta"] = float(payload.get("energy_delta", 0.0))
    atoms.info["atomlens_residual"] = float(payload.get("residual", 0.0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write(output_path, atoms, format="extxyz")


def attribution_array(payload: dict[str, Any], atom_count: int) -> np.ndarray:
    ranking = payload.get("ranking")
    if not isinstance(ranking, list):
        raise RuntimeError("attribution payload does not contain a ranking list")
    values = np.zeros(atom_count, dtype=float)
    seen = set()
    for row in ranking:
        index = int(row["index"])
        if index < 0 or index >= atom_count:
            raise RuntimeError(f"ranking index {index} is outside atom count {atom_count}")
        if index in seen:
            raise RuntimeError(f"ranking index {index} appears more than once")
        values[index] = float(row["value"])
        seen.add(index)
    return values
