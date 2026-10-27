#!/usr/bin/env python
"""Joint fit-model x fit-range variations of the raw asymmetry.

The fit-model and fit-range systematics are not independent on this sample:
the sensitivity to the lower fit edge only appears with the nominal
Gaussian + exponential model, whose steep background absorbs a low-mass
shoulder in the B+ spectrum, and disappears with the other models. Adding the
two one-at-a-time shifts in quadrature would therefore assume an independence
the data contradict. This study refits every model on every fit range and
assigns the largest shift from the nominal over the accepted configurations,
which measures the joint variation directly. Every configuration is fitted on
both magnet polarities, simultaneously in charge, and averaged like the nominal.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.fitting.acceptance import DEFAULT_MAX_CHI2_NDOF
from src.physics.selection import load_charge_samples
from src.plotting.mass_plots import configure_matplotlib_cache, set_hep_style
from src.utils.io import DEFAULT_DATA_PATHS, display_path, write_json
from studies.systematics.configurations import (
    FIT_RANGE_VARIATIONS,
    evaluate_configuration,
    implemented_models,
    write_rows_csv,
)


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "systematics"


def summarize(rows, bins: int, max_chi2_ndof: float):
    nominal = next((row for row in rows if row["is_nominal"]), None)
    if nominal is None or nominal["A_raw"] is None:
        raise ValueError("The nominal configuration did not produce an asymmetry.")
    nominal_a_raw = nominal["A_raw"]

    for row in rows:
        row["delta_A_raw"] = None if row["A_raw"] is None else row["A_raw"] - nominal_a_raw

    accepted = [row for row in rows if row["accepted_for_systematic"] and row["A_raw"] is not None]
    accepted_values = np.array([row["A_raw"] for row in accepted], dtype=float)
    shifts = np.abs(accepted_values - nominal_a_raw)
    largest = accepted[int(np.argmax(shifts))] if len(accepted) else None

    return {
        "study": "joint fit-model x fit-range variations",
        "nominal_configuration": {
            "model_name": nominal["model_name"],
            "range_name": nominal["range_name"],
        },
        "nominal_A_raw": nominal_a_raw,
        "n_configurations": len(rows),
        "n_accepted_configurations": len(accepted),
        "accepted_A_raw_min": float(accepted_values.min()) if len(accepted) else None,
        "accepted_A_raw_max": float(accepted_values.max()) if len(accepted) else None,
        "max_abs_delta_A_raw_accepted": float(shifts.max()) if len(accepted) else None,
        "largest_shift_configuration": (
            {"model_name": largest["model_name"], "range_name": largest["range_name"]}
            if largest is not None
            else None
        ),
        "recommended_fit_systematic": float(shifts.max()) if len(accepted) else None,
        "systematic_definition": (
            "Largest |A_raw - A_raw(nominal)| over every accepted fit-model x fit-range "
            "configuration. It replaces the quadrature sum of the one-at-a-time model "
            "and range shifts, which are not independent on this sample. The range "
            "shifts inside it are not Barlow-corrected, so it is mildly conservative."
        ),
        "configurations": rows,
        "fit_quality_criteria": {
            "max_chi2_ndof": max_chi2_ndof,
            "require_converged_B_plus": True,
            "require_converged_B_minus": True,
            "require_positive_yields": True,
            "require_positive_yield_uncertainties": True,
        },
        "bins": int(bins),
    }


def write_grid_plot(rows, models, ranges, nominal_a_raw: float, output_path: Path):
    configure_matplotlib_cache()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = np.full((len(models), len(ranges)), np.nan)
    accepted = np.zeros_like(grid, dtype=bool)
    for row in rows:
        i = [model["model_name"] for model in models].index(row["model_name"])
        j = [variation["range_name"] for variation in ranges].index(row["range_name"])
        if row["A_raw"] is not None:
            grid[i, j] = row["A_raw"] - nominal_a_raw
        accepted[i, j] = row["accepted_for_systematic"]

    limit = float(np.nanmax(np.abs(grid))) if np.isfinite(grid).any() else 1.0
    set_hep_style()
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    mesh = ax.imshow(grid, cmap="coolwarm", vmin=-limit, vmax=limit, aspect="auto")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if np.isfinite(grid[i, j]):
                label = f"{grid[i, j]:+.3f}" + ("" if accepted[i, j] else "\n(rejected)")
                ax.text(j, i, label, ha="center", va="center", fontsize=9)
    ax.set_xticks(range(len(ranges)))
    ax.set_xticklabels(
        [f"{v['fit_range'][0]:.0f}-{v['fit_range'][1]:.0f}" for v in ranges], rotation=20, ha="right"
    )
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([model["model_name"].replace("_", " ") for model in models])
    ax.set_xlabel(r"Fit range [MeV/$c^2$]")
    ax.set_title(r"$A_{\rm raw} - A_{\rm raw}^{\rm nominal}$ over fit models and ranges")
    colorbar = fig.colorbar(mesh, ax=ax)
    colorbar.set_label(r"$\Delta A_{\rm raw}$")
    ax.grid(False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bins", type=int, default=80, help="Number of fitted mass bins.")
    parser.add_argument(
        "--max-chi2-ndof",
        type=float,
        default=DEFAULT_MAX_CHI2_NDOF,
        help="Maximum chi2/ndof allowed for accepted configurations.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for joint-variation outputs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if args.bins <= 0:
        raise ValueError("--bins must be positive.")

    samples = load_charge_samples(DEFAULT_DATA_PATHS)
    models = implemented_models()
    rows = [
        evaluate_configuration(samples, model, variation, args.bins, args.max_chi2_ndof)
        for model in models
        for variation in FIT_RANGE_VARIATIONS
    ]
    summary = summarize(rows, args.bins, args.max_chi2_ndof)

    csv_path = output_dir / "joint_variations.csv"
    summary_path = output_dir / "joint_variations_summary.json"
    plot_path = output_dir / "joint_variations.png"
    write_rows_csv(rows, csv_path)
    write_json(summary_path, summary)
    write_grid_plot(rows, models, FIT_RANGE_VARIATIONS, summary["nominal_A_raw"], plot_path)

    print(
        f"Accepted configurations: {summary['n_accepted_configurations']} / {summary['n_configurations']}"
    )
    print(
        f"Accepted A_raw envelope: [{summary['accepted_A_raw_min']:.5f}, {summary['accepted_A_raw_max']:.5f}]"
    )
    print(f"Recommended fit systematic: {summary['recommended_fit_systematic']:.6f}")
    for path in (csv_path, summary_path, plot_path):
        print(f"Wrote {display_path(path)}")


if __name__ == "__main__":
    main()
