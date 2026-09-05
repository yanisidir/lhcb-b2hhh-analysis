#!/usr/bin/env python
"""Fit-range variation study for charge-separated raw asymmetry extraction."""

from __future__ import annotations

import argparse
import csv
import json
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


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "systematics"
SIGNAL_MODEL = "gaussian"
BACKGROUND_MODEL = "exponential"
FIT_RANGE_VARIATIONS = [
    {
        "range_name": "nominal_5200_5600",
        "fit_range": (5200.0, 5600.0),
        "variation_role": "nominal",
        "is_nominal": True,
    },
    {
        "range_name": "narrower_low_5230_5600",
        "fit_range": (5230.0, 5600.0),
        "variation_role": "candidate_systematic",
        "is_nominal": False,
    },
    {
        "range_name": "narrower_high_5200_5550",
        "fit_range": (5200.0, 5550.0),
        "variation_role": "candidate_systematic",
        "is_nominal": False,
    },
    {
        "range_name": "narrower_both_5230_5550",
        "fit_range": (5230.0, 5550.0),
        "variation_role": "candidate_systematic",
        "is_nominal": False,
    },
    {
        "range_name": "wider_high_5200_5700",
        "fit_range": (5200.0, 5700.0),
        "variation_role": "candidate_systematic",
        "is_nominal": False,
    },
]


def _fit_charge_sample(masses, fit_range, bins: int):
    try:
        return fit_mass_distribution(
            masses,
            fit_range=fit_range,
            bins=bins,
            signal_model=SIGNAL_MODEL,
            background_model=BACKGROUND_MODEL,
        )
    except Exception as exc:
        return _failed_fit_result(f"{type(exc).__name__}: {exc}", fit_range, bins)


def _failed_fit_result(message: str, fit_range, bins: int):
    return {
        "signal_yield": None,
        "parameter_uncertainties": {"signal_yield": None},
        "fit_quality": {"chi2_ndof": None},
        "fit_status": {
            "backend": None,
            "success": False,
            "valid": False,
            "message": message,
            "fit_range": list(fit_range),
            "bins": int(bins),
            "n_fitted_candidates": 0,
        },
    }


def _raw_asymmetry_safe(plus_result, minus_result, n_plus_candidates: int, n_minus_candidates: int):
    if plus_result.get("signal_yield") is None or minus_result.get("signal_yield") is None:
        return {
            "N_plus": plus_result.get("signal_yield"),
            "sigma_N_plus": _signal_yield_uncertainty(plus_result),
            "N_minus": minus_result.get("signal_yield"),
            "sigma_N_minus": _signal_yield_uncertainty(minus_result),
            "A_raw": None,
            "sigma_A_raw": None,
            "fit_model_used": f"{SIGNAL_MODEL} + {BACKGROUND_MODEL}",
            "n_candidates_plus": int(n_plus_candidates),
            "n_candidates_minus": int(n_minus_candidates),
        }
    return raw_asymmetry(
        plus_result,
        minus_result,
        n_plus_candidates=n_plus_candidates,
        n_minus_candidates=n_minus_candidates,
    )


def _format_float(value):
    return "n/a" if value is None else f"{value:.5f}"


def _variation_row(
    variation,
    plus_result,
    minus_result,
    n_plus_candidates: int,
    n_minus_candidates: int,
    max_chi2_ndof: float,
):
    low, high = variation["fit_range"]
    asymmetry = _raw_asymmetry_safe(
        plus_result,
        minus_result,
        n_plus_candidates=n_plus_candidates,
        n_minus_candidates=n_minus_candidates,
    )
    plus_status = status_summary(plus_result)
    minus_status = status_summary(minus_result)
    row = {
        "range_name": variation["range_name"],
        "fit_range_low": low,
        "fit_range_high": high,
        "signal_model": SIGNAL_MODEL,
        "background_model": BACKGROUND_MODEL,
        "variation_role": variation["variation_role"],
        "is_nominal": variation["is_nominal"],
        "n_candidates_plus": int(n_plus_candidates),
        "n_candidates_minus": int(n_minus_candidates),
        "n_fitted_candidates_plus": plus_result.get("fit_status", {}).get("n_fitted_candidates"),
        "n_fitted_candidates_minus": minus_result.get("fit_status", {}).get("n_fitted_candidates"),
        "N_plus": asymmetry["N_plus"],
        "sigma_N_plus": asymmetry["sigma_N_plus"],
        "N_minus": asymmetry["N_minus"],
        "sigma_N_minus": asymmetry["sigma_N_minus"],
        "A_raw": asymmetry["A_raw"],
        "sigma_A_raw": asymmetry["sigma_A_raw"],
        "chi2_ndof_plus": plus_result.get("fit_quality", {}).get("chi2_ndof"),
        "chi2_ndof_minus": minus_result.get("fit_quality", {}).get("chi2_ndof"),
        "fit_success_plus": plus_status["success"],
        "fit_success_minus": minus_status["success"],
        "fit_valid_plus": plus_status["valid"],
        "fit_valid_minus": minus_status["valid"],
        "fit_backend_plus": plus_status["backend"],
        "fit_backend_minus": minus_status["backend"],
        "fit_status_plus": plus_status["message"],
        "fit_status_minus": minus_status["message"],
    }
    accepted, rejection_reason = fit_quality_acceptance(row, max_chi2_ndof)
    row["accepted_for_systematic"] = accepted
    row["rejection_reason"] = rejection_reason
    return row


def run_variations(plus_masses, minus_masses, bins: int, max_chi2_ndof: float):
    rows = []
    for variation in FIT_RANGE_VARIATIONS:
        fit_range = variation["fit_range"]
        plus_in_range = plus_masses[
            (plus_masses >= fit_range[0]) & (plus_masses <= fit_range[1])
        ]
        minus_in_range = minus_masses[
            (minus_masses >= fit_range[0]) & (minus_masses <= fit_range[1])
        ]
        plus_result = _fit_charge_sample(plus_masses, fit_range, bins)
        minus_result = _fit_charge_sample(minus_masses, fit_range, bins)
        rows.append(
            _variation_row(
                variation,
                plus_result,
                minus_result,
                n_plus_candidates=len(plus_in_range),
                n_minus_candidates=len(minus_in_range),
                max_chi2_ndof=max_chi2_ndof,
            )
        )
        print(
            f"{variation['range_name']}: "
            f"A_raw={_format_float(rows[-1]['A_raw'])}, "
            f"sigma={_format_float(rows[-1]['sigma_A_raw'])}, "
            f"accepted={rows[-1]['accepted_for_systematic']}"
        )
    return rows


def write_csv(rows, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(json_safe(row) for row in rows)


def _summary_entry(row, nominal_a_raw):
    delta = row["A_raw"] - nominal_a_raw if row["A_raw"] is not None else None
    return {
        "range_name": row["range_name"],
        "fit_range": [row["fit_range_low"], row["fit_range_high"]],
        "signal_model": row["signal_model"],
        "background_model": row["background_model"],
        "A_raw": row["A_raw"],
        "sigma_A_raw": row["sigma_A_raw"],
        "delta_A_raw": delta,
        "is_nominal": row["is_nominal"],
        "accepted_for_systematic": row["accepted_for_systematic"],
        "rejection_reason": row["rejection_reason"],
        "n_fitted_candidates_plus": row["n_fitted_candidates_plus"],
        "n_fitted_candidates_minus": row["n_fitted_candidates_minus"],
        "fit_success_plus": row["fit_success_plus"],
        "fit_success_minus": row["fit_success_minus"],
        "chi2_ndof_plus": row["chi2_ndof_plus"],
        "chi2_ndof_minus": row["chi2_ndof_minus"],
    }


def summarize_variations(rows, bins: int, max_chi2_ndof: float):
    nominal = next((row for row in rows if row["is_nominal"]), None)
    if nominal is None:
        raise ValueError("No nominal fit range was run.")

    nominal_a_raw = nominal["A_raw"]
    tested_ranges = [_summary_entry(row, nominal_a_raw) for row in rows]
    accepted_ranges = [
        entry
        for entry in tested_ranges
        if entry["accepted_for_systematic"] and not entry["is_nominal"]
    ]
    rejected_ranges = [
        entry
        for entry in tested_ranges
        if not entry["accepted_for_systematic"]
    ]
    accepted_deltas = [
        abs(entry["delta_A_raw"])
        for entry in accepted_ranges
        if entry["delta_A_raw"] is not None
    ]
    recommended = max(accepted_deltas) if accepted_deltas else None
    status = (
        "complete with accepted fit-range variations"
        if accepted_deltas
        else "pending additional accepted fit-range variations"
    )

    return {
        "status": status,
        "signal_model": SIGNAL_MODEL,
        "background_model": BACKGROUND_MODEL,
        "nominal_range_name": nominal["range_name"],
        "nominal_fit_range": [nominal["fit_range_low"], nominal["fit_range_high"]],
        "nominal_A_raw": nominal_a_raw,
        "all_tested_fit_ranges": tested_ranges,
        "accepted_fit_range_variations": accepted_ranges,
        "rejected_fit_range_variations": rejected_ranges,
        "max_abs_delta_A_raw_accepted": recommended,
        "recommended_fit_range_systematic": recommended,
        "fit_quality_criteria": {
            "max_chi2_ndof": max_chi2_ndof,
            "require_converged_B_plus": True,
            "require_converged_B_minus": True,
            "require_positive_yields": True,
            "require_positive_yield_uncertainties": True,
        },
        "bins": int(bins),
        "note": (
            "First fit-range systematic for the raw fitted-yield asymmetry. "
            "Only accepted charge-separated fits are used for the recommended "
            "fit-range systematic uncertainty."
        ),
    }


def write_variation_plot(rows, output_path: Path):
    configure_matplotlib_cache()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    set_hep_style()
    labels = [row["range_name"].replace("_", "\n") for row in rows]
    values = np.array([row["A_raw"] for row in rows], dtype=float)
    errors = np.array([row["sigma_A_raw"] for row in rows], dtype=float)
    nominal = next(row for row in rows if row["is_nominal"])
    colors = [
        "#4c78a8" if row["accepted_for_systematic"] else "#d62728"
        for row in rows
    ]

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    x = np.arange(len(rows))
    for i, row in enumerate(rows):
        ax.errorbar(
            x[i],
            values[i],
            yerr=errors[i],
            fmt="o",
            color=colors[i],
            capsize=4,
        )
    ax.axhline(nominal["A_raw"], color="#4c78a8", linewidth=2.0, label="Nominal")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(r"$A_{\rm raw}$")
    ax.set_title("Fit-range variation of raw asymmetry")
    ax.legend(frameon=False)
    ax.grid(True, axis="y", alpha=0.25)
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
        help="Maximum chi2/ndof allowed for accepted systematic variations.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for fit-range systematic outputs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_path = args.data.resolve()
    output_dir = args.output_dir.resolve()
    if args.bins <= 0:
        raise ValueError("--bins must be positive.")
    if args.max_chi2_ndof <= 0:
        raise ValueError("--max-chi2-ndof must be positive.")
    if not data_path.exists():
        raise FileNotFoundError(f"Input ROOT file not found: {data_path}")

    selected = prepare_selected_candidates(data_path).copy()
    selected["total_charge"] = (
        selected["H1_Charge"] + selected["H2_Charge"] + selected["H3_Charge"]
    )
    b_plus = selected.loc[selected["total_charge"] == 1].copy()
    b_minus = selected.loc[selected["total_charge"] == -1].copy()

    rows = run_variations(
        b_plus["B_mass"].to_numpy(),
        b_minus["B_mass"].to_numpy(),
        bins=args.bins,
        max_chi2_ndof=args.max_chi2_ndof,
    )
    summary = summarize_variations(rows, args.bins, args.max_chi2_ndof)

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "fit_range_variations.csv"
    summary_path = output_dir / "fit_range_summary.json"
    plot_path = output_dir / "fit_range_variations.png"
    write_csv(rows, csv_path)
    write_json(summary_path, summary)
    write_variation_plot(rows, plot_path)

    recommended = summary["recommended_fit_range_systematic"]
    if recommended is None:
        print("Recommended fit-range systematic: pending accepted fit-range variations")
    else:
        print(f"Recommended fit-range systematic: {recommended:.6f}")
    print(f"Wrote {display_path(csv_path)}")
    print(f"Wrote {display_path(summary_path)}")
    print(f"Wrote {display_path(plot_path)}")


if __name__ == "__main__":
    main()
