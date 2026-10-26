#!/usr/bin/env python
"""Joint fit-model x fit-range variations of the raw asymmetry.

The fit-model and fit-range systematics are not independent on this sample:
the sensitivity to the lower fit edge only appears with the nominal
Gaussian + exponential model, whose steep background absorbs a low-mass
shoulder in the B+ spectrum, and disappears with the other models. Adding the
two one-at-a-time shifts in quadrature would therefore assume an independence
the data contradict. This study refits every model on every fit range and
assigns the largest shift from the nominal over the accepted configurations,
which measures the joint variation directly.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.fitting.acceptance import fit_quality_acceptance, status_summary
from src.fitting.asymmetry import raw_asymmetry
from src.fitting.fit_mass import fit_mass_distribution
from src.physics.selection import prepare_selected_candidates
from src.plotting.mass_plots import configure_matplotlib_cache, set_hep_style
from src.utils.io import DEFAULT_DATA_PATH, display_path, json_safe, write_json
from studies.systematics.run_fit_model_variations import MODEL_VARIATIONS
from studies.systematics.run_fit_range_variations import FIT_RANGE_VARIATIONS


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "systematics"


def implemented_models():
    """Model variations the fitter supports; pending ones are skipped."""
    return [variation for variation in MODEL_VARIATIONS if "pending_reason" not in variation]


def _fit(masses, fit_range, bins: int, signal_model: str, background_model: str):
    try:
        return fit_mass_distribution(
            masses,
            fit_range=fit_range,
            bins=bins,
            signal_model=signal_model,
            background_model=background_model,
        )
    except Exception as exc:
        return {
            "signal_yield": None,
            "parameter_uncertainties": {"signal_yield": None},
            "fit_quality": {"chi2_ndof": None},
            "signal_model": signal_model,
            "background_model": background_model,
            "fit_status": {
                "success": False,
                "valid": False,
                "backend": None,
                "message": f"{type(exc).__name__}: {exc}",
                "fit_range": list(fit_range),
                "n_fitted_candidates": 0,
            },
        }


def configuration_row(model, fit_range_variation, plus_masses, minus_masses, bins: int, max_chi2_ndof: float):
    """Fit both charges in one model x range configuration and judge it."""
    fit_range = fit_range_variation["fit_range"]
    plus = _fit(plus_masses, fit_range, bins, model["signal_model"], model["background_model"])
    minus = _fit(minus_masses, fit_range, bins, model["signal_model"], model["background_model"])

    if plus["signal_yield"] is None or minus["signal_yield"] is None:
        asymmetry = {"N_plus": None, "sigma_N_plus": None, "N_minus": None, "sigma_N_minus": None,
                     "A_raw": None, "sigma_A_raw": None}
    else:
        asymmetry = raw_asymmetry(plus, minus, 0, 0)

    plus_status = status_summary(plus)
    minus_status = status_summary(minus)
    row = {
        "model_name": model["model_name"],
        "signal_model": model["signal_model"],
        "background_model": model["background_model"],
        "range_name": fit_range_variation["range_name"],
        "fit_range_low": fit_range[0],
        "fit_range_high": fit_range[1],
        "is_nominal": bool(
            model["variation_role"] == "nominal" and fit_range_variation["is_nominal"]
        ),
        "N_plus": asymmetry["N_plus"],
        "sigma_N_plus": asymmetry["sigma_N_plus"],
        "N_minus": asymmetry["N_minus"],
        "sigma_N_minus": asymmetry["sigma_N_minus"],
        "A_raw": asymmetry["A_raw"],
        "sigma_A_raw": asymmetry["sigma_A_raw"],
        "chi2_ndof_plus": plus.get("fit_quality", {}).get("chi2_ndof"),
        "chi2_ndof_minus": minus.get("fit_quality", {}).get("chi2_ndof"),
        "fit_success_plus": plus_status["success"],
        "fit_success_minus": minus_status["success"],
        "fit_valid_plus": plus_status["valid"],
        "fit_valid_minus": minus_status["valid"],
    }
    accepted, reason = fit_quality_acceptance(row, max_chi2_ndof)
    row["accepted_for_systematic"] = accepted
    row["rejection_reason"] = reason
    return row


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


def write_csv(rows, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(json_safe(row) for row in rows)


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
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help="Input ROOT ntuple.")
    parser.add_argument("--bins", type=int, default=80, help="Number of fitted mass bins.")
    parser.add_argument(
        "--max-chi2-ndof",
        type=float,
        default=2.5,
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
    data_path = args.data.resolve()
    output_dir = args.output_dir.resolve()
    if args.bins <= 0:
        raise ValueError("--bins must be positive.")
    if not data_path.exists():
        raise FileNotFoundError(f"Input ROOT file not found: {data_path}")

    selected = prepare_selected_candidates(data_path)
    total_charge = selected["H1_Charge"] + selected["H2_Charge"] + selected["H3_Charge"]
    plus_masses = selected.loc[total_charge == 1, "B_mass"].to_numpy()
    minus_masses = selected.loc[total_charge == -1, "B_mass"].to_numpy()

    models = implemented_models()
    rows = [
        configuration_row(model, variation, plus_masses, minus_masses, args.bins, args.max_chi2_ndof)
        for model in models
        for variation in FIT_RANGE_VARIATIONS
    ]
    summary = summarize(rows, args.bins, args.max_chi2_ndof)

    csv_path = output_dir / "joint_variations.csv"
    summary_path = output_dir / "joint_variations_summary.json"
    plot_path = output_dir / "joint_variations.png"
    write_csv(rows, csv_path)
    write_json(summary_path, summary)
    write_grid_plot(rows, models, FIT_RANGE_VARIATIONS, summary["nominal_A_raw"], plot_path)

    print(
        f"Accepted configurations: {summary['n_accepted_configurations']} / {summary['n_configurations']}"
    )
    print(
        f"Accepted A_raw envelope: [{summary['accepted_A_raw_min']:.5f}, {summary['accepted_A_raw_max']:.5f}]"
    )
    print(f"Recommended fit systematic: {summary['recommended_fit_systematic']:.6f}")
    print(f"Wrote {display_path(csv_path)}")
    print(f"Wrote {display_path(summary_path)}")
    print(f"Wrote {display_path(plot_path)}")


if __name__ == "__main__":
    main()
