"""Invariant-mass reconstruction utilities for B -> hhh candidates."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


KAON_MASS_MEV = 493.677


def _energy(px, py, pz, mass: float):
    """Return relativistic energy in MeV for momentum components in MeV."""
    momentum2 = px**2 + py**2 + pz**2
    return np.sqrt(momentum2 + mass**2)


def compute_b_mass(
    df,
    hadron_mass: float = KAON_MASS_MEV,
    momentum_columns: Mapping[str, tuple[str, str, str]] | None = None,
):
    """Compute the invariant mass of a three-hadron B candidate.

    Parameters
    ----------
    df:
        Table containing the daughter momentum components. A pandas
        ``DataFrame`` is expected in the notebooks, but any object supporting
        column access works.
    hadron_mass:
        Mass hypothesis assigned to each daughter track, in MeV. The default
        is the charged-kaon mass, appropriate for a B -> K K K demonstration.
    momentum_columns:
        Optional mapping from daughter label to ``(PX, PY, PZ)`` column names.

    Returns
    -------
    array-like
        Reconstructed invariant mass in MeV.
    """
    columns = momentum_columns or {
        "H1": ("H1_PX", "H1_PY", "H1_PZ"),
        "H2": ("H2_PX", "H2_PY", "H2_PZ"),
        "H3": ("H3_PX", "H3_PY", "H3_PZ"),
    }

    total_energy = 0
    total_px = 0
    total_py = 0
    total_pz = 0

    for px_col, py_col, pz_col in columns.values():
        px = df[px_col]
        py = df[py_col]
        pz = df[pz_col]
        total_energy = total_energy + _energy(px, py, pz, hadron_mass)
        total_px = total_px + px
        total_py = total_py + py
        total_pz = total_pz + pz

    mass2 = total_energy**2 - (total_px**2 + total_py**2 + total_pz**2)
    return np.sqrt(np.clip(mass2, a_min=0, a_max=None))
