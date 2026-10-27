#!/usr/bin/env python
"""Fit-range variations of the polarity-averaged raw asymmetry, with the nominal model.

A change of range adds or removes candidates, so part of every shift is
statistical; the Barlow correction removes it. This is the one-at-a-time range
diagnostic; the systematic itself is the joint model x range envelope of
``run_joint_variations.py``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.fitting.acceptance import DEFAULT_MAX_CHI2_NDOF, barlow_shift
from src.physics.selection import load_charge_samples
from src.utils.io import DEFAULT_DATA_PATHS, display_path, write_json
from studies.systematics.configurations import (
    FIT_RANGE_VARIATIONS,
    evaluate_configuration,
    nominal_model,
    write_rows_csv,
    write_variation_plot,
)


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "systematics"
EMPTY_BARLOW = {"sigma_barlow": None, "barlow_significance": None, "barlow_corrected_shift": None}


def summarize(rows, bins: int, max_chi2_ndof: float):
    nominal = next(row for row in rows if row["is_nominal"])
    for row in rows:
        row["delta_A_raw"] = None if row["A_raw"] is None else row["A_raw"] - nominal["A_raw"]
        row.update(
            EMPTY_BARLOW
            if row["is_nominal"]
            else barlow_shift(row["delta_A_raw"], row["sigma_A_raw"], nominal["sigma_A_raw"])
        )
    variations = [row for row in rows if not row["is_nominal"]]
    accepted = [row for row in variations if row["accepted_for_systematic"]]
    raw_shifts = [abs(row["delta_A_raw"]) for row in accepted if row["delta_A_raw"] is not None]
    corrected = [row["barlow_corrected_shift"] for row in accepted if row["barlow_corrected_shift"] is not None]
    return {
        "study": "fit-range variations, one at a time, with the nominal model",
        "nominal_model": nominal["model_name"],
        "nominal_fit_range": [nominal["fit_range_low"], nominal["fit_range_high"]],
        "nominal_A_raw": nominal["A_raw"],
        "all_tested_fit_ranges": rows,
        "accepted_variations": accepted,
        "rejected_variations": [row for row in variations if not row["accepted_for_systematic"]],
        "max_abs_delta_A_raw_accepted": max(raw_shifts) if raw_shifts else None,
        "recommended_fit_range_systematic": max(corrected) if corrected else None,
        "systematic_definition": (
            "Largest Barlow-corrected shift sqrt(max(delta^2 - sigma_barlow^2, 0)) "
            "over the accepted variations, sigma_barlow = sqrt(|sigma_var^2 - sigma_nom^2|)."
        ),
        "role": "diagnostic; the quoted systematic is the joint model x range envelope",
        "max_chi2_ndof": max_chi2_ndof,
        "bins": int(bins),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bins", type=int, default=80, help="Number of fitted mass bins.")
    parser.add_argument(
        "--max-chi2-ndof",
        type=float,
        default=DEFAULT_MAX_CHI2_NDOF,
        help="Maximum chi2/ndof allowed for accepted variations.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir.resolve()
    samples = load_charge_samples(DEFAULT_DATA_PATHS)
    rows = []
    for variation in FIT_RANGE_VARIATIONS:
        rows.append(evaluate_configuration(samples, nominal_model(), variation, args.bins, args.max_chi2_ndof))
        print(f"{variation['range_name']:26s} A_raw = {rows[-1]['A_raw']:+.5f}   accepted = {rows[-1]['accepted_for_systematic']}")
    summary = summarize(rows, args.bins, args.max_chi2_ndof)

    csv_path = output_dir / "fit_range_variations.csv"
    summary_path = output_dir / "fit_range_summary.json"
    plot_path = output_dir / "fit_range_variations.png"
    write_rows_csv(rows, csv_path)
    write_json(summary_path, summary)
    write_variation_plot(rows, "range_name", "Fit-range variation of the raw asymmetry", plot_path)
    print(f"Largest Barlow-corrected range shift: {summary['recommended_fit_range_systematic']:.6f}")
    for path in (csv_path, summary_path, plot_path):
        print(f"Wrote {display_path(path)}")


if __name__ == "__main__":
    main()
