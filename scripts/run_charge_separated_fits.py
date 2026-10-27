#!/usr/bin/env python
"""Fit the B+ and B- spectra of each magnet polarity and extract the raw asymmetry.

For each polarity the two charge spectra are fitted simultaneously, with
shared shape parameters and separate yields (see ``src/fitting/simultaneous.py``
for why). The raw asymmetry is a parameter of each fit, and the nominal value is
the equal-weight average over the two polarities.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.fitting.simultaneous import charge_view, fit_polarities
from src.physics.selection import load_charge_samples
from src.plotting.fit_plots import plot_fit
from src.utils.io import DEFAULT_DATA_PATHS, display_path, write_json


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "fits" / "charge_separated"
CHARGE_LABELS = {"plus": r"$B^+$", "minus": r"$B^-$"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--polarities",
        nargs="+",
        choices=tuple(DEFAULT_DATA_PATHS),
        default=list(DEFAULT_DATA_PATHS),
        help="Magnet polarities to fit and average.",
    )
    parser.add_argument("--bins", type=int, default=80, help="Number of fitted mass bins.")
    parser.add_argument(
        "--signal-model",
        choices=("gaussian", "double_gaussian"),
        default="gaussian",
        help="Signal mass model used in the nominal fits.",
    )
    parser.add_argument(
        "--background-model",
        choices=("exponential", "linear"),
        default="exponential",
        help="Background mass model used in the nominal fits.",
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
        help="Directory for the fit outputs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir.resolve()
    fit_range = tuple(args.fit_range)
    data_paths = {polarity: DEFAULT_DATA_PATHS[polarity] for polarity in args.polarities}
    for path in data_paths.values():
        if not path.exists():
            raise FileNotFoundError(f"Input ROOT file not found: {path}")

    samples = load_charge_samples(data_paths)
    results, combination = fit_polarities(
        samples,
        fit_range=fit_range,
        bins=args.bins,
        signal_model=args.signal_model,
        background_model=args.background_model,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for polarity, result in results.items():
        plus_masses, minus_masses = samples[polarity]
        result.update(
            {
                "polarity": polarity,
                "data_path": str(display_path(data_paths[polarity])),
                "mass_column": "B_mass",
                "charge_definition": "total_charge = H1_Charge + H2_Charge + H3_Charge",
                "selection": "src.physics.selection.prepare_selected_candidates",
                "n_selected_plus": int(len(plus_masses)),
                "n_selected_minus": int(len(minus_masses)),
            }
        )
        written.append(write_json(output_dir / f"{polarity}_fit_results.json", result))
        for charge, masses in (("plus", plus_masses), ("minus", minus_masses)):
            path = output_dir / f"{polarity}_B_{charge}_fit_plot.png"
            plot_fit(
                masses,
                charge_view(result, charge),
                path,
                fit_range=fit_range,
                bins=args.bins,
                title=f"{CHARGE_LABELS[charge]} candidates, {polarity}",
            )
            written.append(path)

    asymmetry = {
        **combination,
        "fit_model_used": f"{args.signal_model} + {args.background_model}",
        "fit_range": list(fit_range),
        "bins": int(args.bins),
        "fit": "simultaneous B+/B- fit per polarity, shared shape parameters",
        "combination": "equal-weight average over magnet polarities",
        "note": (
            "Raw fitted-yield asymmetry only; no corrections for detector, production, "
            "trigger, reconstruction, or selection-induced charge asymmetries are applied. "
            "Averaging the polarities cancels polarity-odd detection asymmetries to first order."
        ),
    }
    written.append(write_json(output_dir / "raw_asymmetry.json", asymmetry))

    for polarity, result in results.items():
        status = "valid" if result["fit_status"]["valid"] else "NOT VALID"
        print(
            f"{polarity:11s} A_raw = {result['A_raw']:+.5f} +/- {result['sigma_A_raw']:.5f}   "
            f"N+ = {result['N_plus']:.1f}  N- = {result['N_minus']:.1f}   ({status})"
        )
    print(f"{'Average':11s} A_raw = {combination['A_raw']:+.5f} +/- {combination['sigma_A_raw']:.5f}")
    for path in written:
        print(f"Wrote {display_path(path)}")


if __name__ == "__main__":
    main()
