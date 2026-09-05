"""Checks of the reconstruction and selection against analytically known cases.

These are the calculations every downstream result depends on, so they are
tested on configurations whose answer can be written down by hand.
"""

import numpy as np
import pandas as pd
import pytest

from src.physics.dalitz import compute_dalitz_variables, compute_pair_mass
from src.physics.selection import apply_baseline_selection
from src.reconstruction.invariant_mass import KAON_MASS_MEV, compute_b_mass


def _candidate(momenta):
    """Build a one-row candidate table from {'H1': (px, py, pz), ...}."""
    row = {}
    for hadron, (px, py, pz) in momenta.items():
        row[f"{hadron}_PX"] = [px]
        row[f"{hadron}_PY"] = [py]
        row[f"{hadron}_PZ"] = [pz]
    return pd.DataFrame(row)


def test_three_daughters_at_rest_give_three_kaon_masses():
    # With no momentum, the invariant mass of the system is the sum of the
    # daughter masses.
    df = _candidate({"H1": (0.0, 0.0, 0.0), "H2": (0.0, 0.0, 0.0), "H3": (0.0, 0.0, 0.0)})
    mass = compute_b_mass(df).to_numpy()
    assert mass == pytest.approx(3.0 * KAON_MASS_MEV)


def test_invariant_mass_is_rotation_invariant():
    momenta = {"H1": (1200.0, -300.0, 8000.0), "H2": (-450.0, 900.0, 5200.0), "H3": (70.0, 40.0, 3100.0)}
    rotated = {h: (-py, px, pz) for h, (px, py, pz) in momenta.items()}  # 90 degrees about z

    mass = compute_b_mass(_candidate(momenta)).to_numpy()
    mass_rotated = compute_b_mass(_candidate(rotated)).to_numpy()

    assert mass_rotated == pytest.approx(mass)


def test_back_to_back_pair_mass_matches_closed_form():
    # Two daughters with opposite momenta: m^2 = (2E)^2, so m = 2 sqrt(p^2 + mK^2).
    p = 2500.0
    df = _candidate({"H1": (p, 0.0, 0.0), "H2": (-p, 0.0, 0.0), "H3": (0.0, 0.0, 0.0)})
    expected = 2.0 * np.sqrt(p**2 + KAON_MASS_MEV**2)

    pair_mass = compute_pair_mass(df, "H1", "H2").to_numpy()

    assert pair_mass == pytest.approx(expected)


def test_pair_mass_does_not_depend_on_daughter_order():
    df = _candidate({"H1": (800.0, 120.0, 4000.0), "H2": (-300.0, 640.0, 2500.0), "H3": (0.0, 0.0, 900.0)})

    assert compute_pair_mass(df, "H1", "H2").to_numpy() == pytest.approx(
        compute_pair_mass(df, "H2", "H1").to_numpy()
    )


def test_dalitz_variables_are_squared_masses_in_gev2():
    df = _candidate({"H1": (800.0, 120.0, 4000.0), "H2": (-300.0, 640.0, 2500.0), "H3": (0.0, 0.0, 900.0)})

    dalitz = compute_dalitz_variables(df)
    pair_mass_mev = compute_pair_mass(df, "H1", "H2").to_numpy()

    assert set(dalitz.columns) == {"m_K1K2_sq", "m_K2K3_sq", "m_K1K3_sq"}
    assert dalitz["m_K1K2_sq"].to_numpy() == pytest.approx(pair_mass_mev**2 / 1e6)


def test_baseline_selection_rejects_each_failing_requirement():
    df = pd.DataFrame(
        {
            "B_VertexChi2": [1.0, 9.0, 1.0, 1.0, 1.0],
            "B_FlightDistance": [50.0, 50.0, 5.0, 50.0, 50.0],
            "H1_ProbK": [0.9, 0.9, 0.9, 0.1, 0.9],
            "H2_ProbK": [0.9, 0.9, 0.9, 0.9, 0.9],
            "H3_ProbK": [0.9, 0.9, 0.9, 0.9, 0.2],
        }
    )

    selected = apply_baseline_selection(df)

    # Only the first candidate satisfies every requirement.
    assert list(selected.index) == [0]


def test_baseline_selection_thresholds_are_configurable():
    df = pd.DataFrame(
        {
            "B_VertexChi2": [7.0],
            "B_FlightDistance": [10.0],
            "H1_ProbK": [0.4],
            "H2_ProbK": [0.4],
            "H3_ProbK": [0.4],
        }
    )

    assert len(apply_baseline_selection(df)) == 0
    assert len(
        apply_baseline_selection(df, vertex_chi2_max=8.0, flight_distance_min=5.0, probk_min=0.3)
    ) == 1
