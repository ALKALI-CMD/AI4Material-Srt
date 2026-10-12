import numpy as np
import pytest
from ase import Atoms

from atomlens.attribution import integrated_force_attribution
from atomlens.compare import attribution_values
from atomlens.export import attribution_array
from atomlens.symmetry import rotated_z, translated
from atomlens.testing import HarmonicCalculator
from atomlens.validation import validate_results


def test_integrated_force_attribution_completes_energy_delta():
    start = Atoms("H2", positions=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)])
    end = Atoms("H2", positions=[(0.2, 0.0, 0.0), (1.0, -0.3, 0.0)])
    reference = start.get_positions()

    result = integrated_force_attribution(
        start,
        end,
        calculator_factory=lambda: HarmonicCalculator(reference_positions=reference, k=2.0),
        n_steps=17,
    )

    assert result.energy_delta == pytest.approx(0.13)
    assert result.total == pytest.approx(result.energy_delta)
    assert abs(result.residual) < 1e-12
    assert result.atomic == pytest.approx(np.array([0.04, 0.09]))


def test_integrated_force_attribution_rejects_changed_atom_order():
    start = Atoms("HO", positions=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)])
    end = Atoms("OH", positions=[(0.0, 0.0, 0.1), (1.0, 0.0, 0.0)])

    with pytest.raises(ValueError, match="same ordered symbols"):
        integrated_force_attribution(
            start,
            end,
            calculator_factory=lambda: HarmonicCalculator(reference_positions=start.get_positions()),
        )


def test_integrated_force_attribution_is_translation_invariant_for_matching_reference():
    start = Atoms("H2", positions=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)])
    end = Atoms("H2", positions=[(0.2, 0.0, 0.0), (1.0, -0.3, 0.0)])
    shift = (10.0, -2.0, 0.5)

    base = integrated_force_attribution(
        start,
        end,
        calculator_factory=lambda: HarmonicCalculator(reference_positions=start.get_positions()),
        n_steps=17,
    )
    moved_start = translated(start, shift)
    moved_end = translated(end, shift)
    moved = integrated_force_attribution(
        moved_start,
        moved_end,
        calculator_factory=lambda: HarmonicCalculator(reference_positions=moved_start.get_positions()),
        n_steps=17,
    )

    assert moved.atomic == pytest.approx(base.atomic)
    assert moved.energy_delta == pytest.approx(base.energy_delta)


def test_integrated_force_attribution_is_rotation_invariant_for_matching_reference():
    start = Atoms("H2", positions=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)])
    end = Atoms("H2", positions=[(0.2, 0.0, 0.0), (1.0, -0.3, 0.0)])

    base = integrated_force_attribution(
        start,
        end,
        calculator_factory=lambda: HarmonicCalculator(reference_positions=start.get_positions()),
        n_steps=17,
    )
    rotated_start = rotated_z(start, 33.0)
    rotated_end = rotated_z(end, 33.0)
    rotated = integrated_force_attribution(
        rotated_start,
        rotated_end,
        calculator_factory=lambda: HarmonicCalculator(reference_positions=rotated_start.get_positions()),
        n_steps=17,
    )

    assert rotated.atomic == pytest.approx(base.atomic)
    assert rotated.energy_delta == pytest.approx(base.energy_delta)


def test_attribution_array_maps_ranking_by_atom_index():
    payload = {
        "ranking": [
            {"index": 2, "value": 3.0},
            {"index": 0, "value": -1.0},
            {"index": 1, "value": 2.0},
        ]
    }

    assert attribution_array(payload, atom_count=3) == pytest.approx(np.array([-1.0, 2.0, 3.0]))


def test_attribution_values_rejects_duplicate_indices():
    payload = {"ranking": [{"index": 0, "value": 1.0}, {"index": 0, "value": 2.0}]}

    with pytest.raises(RuntimeError, match="duplicate atom index"):
        attribution_values(payload)


def test_validate_results_flags_large_residual(tmp_path):
    attribution_path = tmp_path / "bad.json"
    attribution_path.write_text(
        '{"residual": 0.1, "ranking": [{"index": 0, "value": 1.0}]}',
        encoding="utf-8",
    )

    result = validate_results(attribution_path, max_residual=1e-3)

    assert result["passed"] is False
