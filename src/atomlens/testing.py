from __future__ import annotations

import numpy as np
from ase.calculators.calculator import Calculator, all_changes


class HarmonicCalculator(Calculator):
    """Simple analytic calculator used for attribution tests."""

    implemented_properties = ["energy", "forces"]

    def __init__(self, reference_positions: np.ndarray, k: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.reference_positions = np.asarray(reference_positions, dtype=float)
        self.k = float(k)

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        displacement = self.atoms.get_positions() - self.reference_positions
        self.results["energy"] = 0.5 * self.k * float(np.sum(displacement * displacement))
        self.results["forces"] = -self.k * displacement
