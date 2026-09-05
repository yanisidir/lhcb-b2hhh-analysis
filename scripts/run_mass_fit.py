#!/usr/bin/env python
"""Run the first nominal B -> hhh invariant-mass fit."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.fitting.fit_mass import fit_mass_distribution
from src.physics.selection import prepare_selected_candidates
from src.plotting.fit_plots import plot_fit
from src.utils.io import DEFAULT_DATA_PATH, display_path, write_json


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "fits"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help="Input ROOT ntuple.")
    parser.add_argument("--bins", type=int, default=80, help="Number of fitted mass bins.")
    parser.add_argument(
        "--signal-model",
        choices=("gaussian", "double_gaussian"),
        default="gaussian",
        help="Signal mass model used in the nominal fit.",
    )
    parser.add_argument(
        "--background-model",
        choices=("exponential", "linear"),
        default="exponential",
        help="Background mass model used in the nominal fit.",
    )
    parser.add_argument(
        "--fit-range",
        type=float,
        nargs=2,
        default=(5200.0, 5600.0),
        metavar=("LOW", "HIGH"),
        help="Fitted invariant-mass range in MeV/c^2.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for mass-fit outputs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_path = args.data.resolve()
    output_dir = args.output_dir.resolve()
    fit_range = tuple(args.fit_range)

    if not data_path.exists():
        raise FileNotFoundError(f"Input ROOT file not found: {data_path}")

    selected = prepare_selected_candidates(data_path)
    masses = selected["B_mass"].to_numpy()
    result = fit_mass_distribution(
        masses,
        fit_range=fit_range,
        bins=args.bins,
        signal_model=args.signal_model,
        background_model=args.background_model,
    )
    result.update(
        {
            "note": (
                "First nominal mass fit with configurable Gaussian-family signal "
                "and finite-range background models; not yet a full systematic study."
            ),
            "data_path": str(data_path.relative_to(PROJECT_ROOT)),
            "mass_column": "B_mass",
            "selection": "src.physics.selection.prepare_selected_candidates",
        }
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "mass_fit_results.json"
    plot_path = output_dir / "mass_fit_plot.png"

    write_json(results_path, result)
    plot_fit(masses, result, plot_path, fit_range=fit_range, bins=args.bins)

    print(f"Selected candidates in fit: {result['fit_status']['n_candidates']:,}")
    print(f"Signal model:             {result['signal_model']}")
    print(f"Background model:         {result['background_model']}")
    print(f"Signal yield:             {result['signal_yield']:.1f}")
    print(f"Background yield:         {result['background_yield']:.1f}")
    print(f"Mass mean:                {result['mass_mean']:.2f} MeV/c^2")
    print(f"Mass resolution:          {result['mass_resolution']:.2f} MeV/c^2")
    if result["signal_model"] == "double_gaussian":
        print(f"Narrow resolution:        {result['mass_resolution_narrow']:.2f} MeV/c^2")
        print(f"Wide resolution:          {result['mass_resolution_wide']:.2f} MeV/c^2")
        print(f"Narrow signal fraction:   {result['signal_fraction']:.3f}")
    chi2_ndof = result["fit_quality"]["chi2_ndof"]
    print(f"Chi2 / ndof:              {chi2_ndof:.3f}" if chi2_ndof is not None else "Chi2 / ndof:              n/a")
    print(f"Fit backend:              {result['fit_status']['backend']}")
    print(f"Wrote {display_path(results_path)}")
    print(f"Wrote {display_path(plot_path)}")


if __name__ == "__main__":
    main()
