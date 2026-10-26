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
    """Return pairwise invariant-mass-squared variables by track label.

    The labels ``H1``, ``H2``, ``H3`` are not charge-ordered, so each column
    mixes same-sign and opposite-sign pairs across candidates. These are useful
    for checking the reconstruction, not as Dalitz coordinates; use
    ``compute_charge_ordered_dalitz_variables`` for any physics comparison.
    """
    import pandas as pd

    return pd.DataFrame(
        {
            "m_K1K2_sq": compute_pair_mass(df, "H1", "H2", mass=mass, squared=True),
            "m_K2K3_sq": compute_pair_mass(df, "H2", "H3", mass=mass, squared=True),
            "m_K1K3_sq": compute_pair_mass(df, "H1", "H3", mass=mass, squared=True),
        },
        index=df.index,
    )


OPPOSITE_SIGN_LOW = "m_KK_low_sq"
OPPOSITE_SIGN_HIGH = "m_KK_high_sq"
SAME_SIGN = "m_KK_same_sign_sq"


def compute_charge_ordered_dalitz_variables(df, mass: float = KAON_MASS_MEV):
    """Return the charge-ordered Dalitz variables of B -> K K K candidates.

    A B+ -> K+ K+ K- candidate has two opposite-sign pairs, K+K-, and one
    same-sign pair, K+K+. Only the opposite-sign pairs carry the resonant
    structure, and they are the variables that map onto each other under CP:
    the K-K+ pairs of the conjugate B- decay. The two same-sign kaons are
    identical particles, so the plane is folded by ordering the two
    opposite-sign masses into ``m_KK_low_sq <= m_KK_high_sq``.

    The track labels ``H1``, ``H2``, ``H3`` carry no physical meaning, so pair
    masses built from them directly (``compute_dalitz_variables``) mix same-sign
    and opposite-sign pairs from one candidate to the next. Here the pairs are
    identified from the daughter charges instead: the opposite-sign pairs are
    the two that contain the track whose charge is opposite to the total.

    Candidates whose total charge is not +-1 have no such track and get NaN.
    Squared masses are in GeV^2/c^4.
    """
    import pandas as pd

    charges = np.column_stack(
        [np.asarray(df[f"{hadron}_Charge"], dtype=float) for hadron in ("H1", "H2", "H3")]
    )
    total_charge = charges.sum(axis=1)
    is_odd_track = charges == -total_charge[:, None]
    valid = (np.abs(total_charge) == 1) & (is_odd_track.sum(axis=1) == 1)
    odd_track = np.argmax(is_odd_track, axis=1)

    m12 = np.asarray(compute_pair_mass(df, "H1", "H2", mass=mass, squared=True), dtype=float)
    m23 = np.asarray(compute_pair_mass(df, "H2", "H3", mass=mass, squared=True), dtype=float)
    m13 = np.asarray(compute_pair_mass(df, "H1", "H3", mass=mass, squared=True), dtype=float)

    # Index 0, 1, 2 of ``odd_track`` is H1, H2, H3; the opposite-sign pairs are
    # the two that contain it, the same-sign pair is the one that does not.
    first = np.choose(odd_track, [m12, m12, m13])
    second = np.choose(odd_track, [m13, m23, m23])
    same_sign = np.choose(odd_track, [m23, m13, m12])

    return pd.DataFrame(
        {
            OPPOSITE_SIGN_LOW: np.where(valid, np.minimum(first, second), np.nan),
            OPPOSITE_SIGN_HIGH: np.where(valid, np.maximum(first, second), np.nan),
            SAME_SIGN: np.where(valid, same_sign, np.nan),
        },
        index=getattr(df, "index", None),
    )
