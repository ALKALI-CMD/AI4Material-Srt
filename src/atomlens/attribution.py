from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator


CalculatorFactory = Callable[[], Calculator]


@dataclass(frozen=True)
class PathAttributionResult:
    """Integrated force attribution along a straight Cartesian path."""

    atomic: np.ndarray
    energy_start: float
    energy_end: float
    residual: float
    alphas: np.ndarray

    @property
    def total(self) -> float:
        return float(np.sum(self.atomic))

    @property
    def energy_delta(self) -> float:
        return float(self.energy_end - self.energy_start)


def integrated_force_attribution(
    atoms_start: Atoms,
    atoms_end: Atoms,
    calculator_factory: CalculatorFactory,
    n_steps: int = 64,
    require_same_cell: bool = True,
) -> PathAttributionResult:
    """Attribute an energy difference to atoms by integrating forces.

    For a fixed-composition path R(alpha) from alpha=0 to 1,

        E(1) - E(0) = - sum_i dR_i dot integral_0^1 F_i(alpha) d alpha.

    The returned per-atom values sum to the model energy change up to numerical
    quadrature error. This is a model-output attribution, not a unique physical
    decomposition of the material energy.
    """

    _validate_pair(atoms_start, atoms_end, require_same_cell=require_same_cell)
    if n_steps < 2:
        raise ValueError("n_steps must be at least 2")

    start = atoms_start.get_positions()
    end = atoms_end.get_positions()
    displacement = end - start
    alphas = np.linspace(0.0, 1.0, n_steps)
    forces = []
    energies = []

    for alpha in alphas:
        atoms = atoms_start.copy()
        atoms.set_positions(start + alpha * displacement)
        atoms.calc = calculator_factory()
        energies.append(float(atoms.get_potential_energy()))
        forces.append(np.asarray(atoms.get_forces(), dtype=float))

    # numpy 2.x 改名为 trapezoid,1.x 叫 trapz,两个都兼容
    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    force_integral = trapezoid(np.stack(forces, axis=0), x=alphas, axis=0)
    atomic = -np.einsum("ij,ij->i", displacement, force_integral)
    residual = float(np.sum(atomic) - (energies[-1] - energies[0]))

    return PathAttributionResult(
        atomic=atomic,
        energy_start=energies[0],
        energy_end=energies[-1],
        residual=residual,
        alphas=alphas,
    )


def _validate_pair(
    atoms_start: Atoms,
    atoms_end: Atoms,
    require_same_cell: bool,
) -> None:
    if len(atoms_start) != len(atoms_end):
        raise ValueError("atoms_start and atoms_end must contain the same atom count")
    if list(atoms_start.get_chemical_symbols()) != list(atoms_end.get_chemical_symbols()):
        raise ValueError("atoms_start and atoms_end must have the same ordered symbols")
    if tuple(atoms_start.get_pbc()) != tuple(atoms_end.get_pbc()):
        raise ValueError("atoms_start and atoms_end must have matching PBC flags")
    if require_same_cell and not np.allclose(atoms_start.cell.array, atoms_end.cell.array):
        raise ValueError("cell changes are not enabled for this attribution path")


def summarize_atomic_values(symbols: Iterable[str], values: np.ndarray) -> list[dict[str, float | str | int]]:
    rows = []
    for index, (symbol, value) in enumerate(zip(symbols, values, strict=True)):
        rows.append({"index": index, "symbol": symbol, "value": float(value)})
    return sorted(rows, key=lambda row: abs(float(row["value"])), reverse=True)
