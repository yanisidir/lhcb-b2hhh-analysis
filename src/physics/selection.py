"""Shared loading and baseline-selection helpers for B -> hhh candidates."""

from __future__ import annotations

from pathlib import Path

from src.reconstruction.invariant_mass import compute_b_mass


B2HHH_BRANCHES = [
    "H1_PX",
    "H1_PY",
    "H1_PZ",
    "H2_PX",
    "H2_PY",
    "H2_PZ",
    "H3_PX",
    "H3_PY",
    "H3_PZ",
    "H1_Charge",
    "H2_Charge",
    "H3_Charge",
    "H1_ProbK",
    "H2_ProbK",
    "H3_ProbK",
    "H1_ProbPi",
    "H2_ProbPi",
    "H3_ProbPi",
    "B_VertexChi2",
    "B_FlightDistance",
]


def load_b2hhh_dataframe(data_path: str | Path, branches: list[str] | None = None):
    """Load B -> hhh candidates from the ROOT ``DecayTree`` into pandas."""
    import uproot

    branches = branches or B2HHH_BRANCHES
    with uproot.open(data_path) as root_file:
        tree = root_file["DecayTree"]
        return tree.arrays(branches, library="pd")


def add_b_mass(df, mass_column: str = "B_mass"):
    """Return a copy of the candidate table with reconstructed B mass."""
    df_mass = df.copy()
    df_mass[mass_column] = compute_b_mass(df_mass)
    return df_mass


def apply_baseline_selection(
    df,
    vertex_chi2_max: float = 5.0,
    flight_distance_min: float = 20.0,
    probk_min: float = 0.5,
):
    """Apply the shared quality and kaon-PID selection used in the notebooks."""
    selection_mask = (
        (df["B_VertexChi2"] < vertex_chi2_max)
        & (df["B_FlightDistance"] > flight_distance_min)
        & (df["H1_ProbK"] > probk_min)
        & (df["H2_ProbK"] > probk_min)
        & (df["H3_ProbK"] > probk_min)
    )
    return df.loc[selection_mask].copy()


def prepare_selected_candidates(data_path: str | Path):
    """Load candidates, reconstruct B mass, and apply the baseline selection."""
    df = load_b2hhh_dataframe(data_path)
    df = add_b_mass(df)
    return apply_baseline_selection(df)


def split_by_charge(df):
    """Return the B+ and B- candidates, by the total charge of the three daughters."""
    total_charge = df["H1_Charge"] + df["H2_Charge"] + df["H3_Charge"]
    return df.loc[total_charge == 1].copy(), df.loc[total_charge == -1].copy()


def load_charge_samples(data_paths):
    """Return ``{polarity: (B+ masses, B- masses)}`` after the baseline selection."""
    samples = {}
    for polarity, data_path in data_paths.items():
        b_plus, b_minus = split_by_charge(prepare_selected_candidates(data_path))
        samples[polarity] = (b_plus["B_mass"].to_numpy(), b_minus["B_mass"].to_numpy())
    return samples
