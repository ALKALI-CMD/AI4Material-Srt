from __future__ import annotations

import numpy as np
from ase import Atoms


def translated(atoms: Atoms, vector: tuple[float, float, float]) -> Atoms:
    shifted = atoms.copy()
    shifted.translate(vector)
    return shifted


def rotated_z(atoms: Atoms, angle_degrees: float, rotate_cell: bool = True) -> Atoms:
    """Rotate positions, and optionally the cell, around the z axis."""

    angle = np.deg2rad(angle_degrees)
    matrix = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rotated = atoms.copy()
    rotated.set_positions(atoms.get_positions() @ matrix.T)
    if rotate_cell:
        rotated.set_cell(atoms.cell.array @ matrix.T, scale_atoms=False)
    return rotated
