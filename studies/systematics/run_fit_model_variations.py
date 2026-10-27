#!/usr/bin/env python
"""Fit-model variations of the polarity-averaged raw asymmetry, on the nominal range.

This is the one-at-a-time model diagnostic. The systematic itself is the joint
model x range envelope of ``run_joint_variations.py``, because the two sources
are not independent.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.fitting.acceptance import DEFAULT_MAX_CHI2_NDOF
from src.physics.selection import load_charge_samples
from src.utils.io import DEFAULT_DATA_PATHS, display_path, write_json
from studies.systematics.configurations import (
    evaluate_configuration,
    implemented_models,
    nominal_range,
    pending_models,
    write_rows_csv,
    write_variation_plot,
)


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "systematics"


def summarize(rows, bins: int, max_chi2_ndof: float):
    nominal = next(row for row in rows if row["is_nominal"])
    for row in rows:
        row["delta_A_raw"] = None if row["A_raw"] is None else row["A_raw"] - nominal["A_raw"]
    variations = [row for row in rows if not row["is_nominal"]]
    accepted = [row for row in variations if row["accepted_for_systematic"]]
    shifts = [abs(row["delta_A_raw"]) for row in accepted if row["delta_A_raw"] is not None]
    return {
        "study": "fit-model variations, one at a time, on the nominal fit range",
        "nominal_model": nominal["model_name"],
        "fit_range": [nominal["fit_range_low"], nominal["fit_range_high"]],
        "nominal_A_raw": nominal["A_raw"],
        "all_tested_model_variations": rows,
        "accepted_variations": accepted,
        "rejected_variations": [row for row in variations if not row["accepted_for_systematic"]],
        "pending_model_variations": [
            {"model_name": model["model_name"], "reason": model["pending_reason"]} for model in pending_models()
        ],
        "max_abs_delta_A_raw_accepted": max(shifts) if shifts else None,
        "recommended_model_systematic": max(shifts) if shifts else None,
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
    for model in implemented_models():
        rows.append(evaluate_configuration(samples, model, nominal_range(), args.bins, args.max_chi2_ndof))
        print(f"{model['model_name']:36s} A_raw = {rows[-1]['A_raw']:+.5f}   accepted = {rows[-1]['accepted_for_systematic']}")
    summary = summarize(rows, args.bins, args.max_chi2_ndof)

    csv_path = output_dir / "fit_model_variations.csv"
    summary_path = output_dir / "fit_model_summary.json"
    plot_path = output_dir / "fit_model_variations.png"
    write_rows_csv(rows, csv_path)
    write_json(summary_path, summary)
    write_variation_plot(rows, "model_name", "Fit-model variation of the raw asymmetry", plot_path)
    print(f"Largest accepted model shift: {summary['recommended_model_systematic']:.6f}")
    for path in (csv_path, summary_path, plot_path):
        print(f"Wrote {display_path(path)}")


if __name__ == "__main__":
    main()
