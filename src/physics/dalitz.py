"""Dalitz-plot kinematics for three-body B -> hhh candidates."""

from __future__ import annotations

import numpy as np

from src.reconstruction.invariant_mass import KAON_MASS_MEV


def _track_energy(df, hadron: str, mass: float):
    px = df[f"{hadron}_PX"]
    py = df[f"{hadron}_PY"]
    pz = df[f"{hadron}_PZ"]
    return np.sqrt(px**2 + py**2 + pz**2 + mass**2)


def compute_pair_mass(
    df,
    first: str,
    second: str,
    mass: float = KAON_MASS_MEV,
    squared: bool = False,
):
    """Compute the invariant mass of a two-track subsystem.

    Parameters
    ----------
    df:
        Candidate table containing ``H*_PX``, ``H*_PY``, and ``H*_PZ`` columns.
    first, second:
        Hadron labels such as ``"H1"`` and ``"H2"``.
    mass:
        Mass hypothesis assigned to each daughter, in MeV.
    squared:
        If true, return the invariant mass squared in GeV^2/c^4. Otherwise,
        return the invariant mass in MeV/c^2.
    """
    total_energy = _track_energy(df, first, mass) + _track_energy(df, second, mass)
    total_px = df[f"{first}_PX"] + df[f"{second}_PX"]
    total_py = df[f"{first}_PY"] + df[f"{second}_PY"]
    total_pz = df[f"{first}_PZ"] + df[f"{second}_PZ"]

    mass2_mev = total_energy**2 - (total_px**2 + total_py**2 + total_pz**2)
    mass2_mev = np.clip(mass2_mev, a_min=0, a_max=None)
    if squared:
        return mass2_mev / 1_000_000.0
    return np.sqrt(mass2_mev)


def compute_dalitz_variables(df, mass: float = KAON_MASS_MEV):
    """Return pairwise K K invariant-mass-squared variables for Dalitz plots."""
    import pandas as pd

    return pd.DataFrame(
        {
            "m_K1K2_sq": compute_pair_mass(df, "H1", "H2", mass=mass, squared=True),
            "m_K2K3_sq": compute_pair_mass(df, "H2", "H3", mass=mass, squared=True),
            "m_K1K3_sq": compute_pair_mass(df, "H1", "H3", mass=mass, squared=True),
        },
        index=df.index,
    )
