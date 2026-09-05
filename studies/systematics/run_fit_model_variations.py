#!/usr/bin/env python
"""Fit-model variation study for charge-separated raw asymmetry extraction."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
for path in (PROJECT_ROOT, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_charge_separated_fits import _raw_asymmetry, _signal_yield_uncertainty
from run_mass_fit import DEFAULT_DATA_PATH, _display_path, _json_safe
from src.fitting.fit_mass import fit_mass_distribution
from src.physics.selection import prepare_selected_candidates
from src.plotting.mass_plots import set_hep_style


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "systematics"
MODEL_VARIATIONS = [
    {
        "model_name": "nominal_gaussian_exponential",
        "signal_model": "gaussian",
        "background_model": "exponential",
        "variation_role": "nominal",
        "is_nominal": True,
    },
    {
        "model_name": "gaussian_linear_background",
        "signal_model": "gaussian",
        "background_model": "linear",
        "variation_role": "candidate_systematic",
        "is_nominal": False,
    },
    {
        "model_name": "double_gaussian_exponential",
        "signal_model": "double_gaussian",
        "background_model": "exponential",
        "variation_role": "candidate_systematic",
        "is_nominal": False,
    },
    {
        "model_name": "double_gaussian_linear_background",
        "signal_model": "double_gaussian",
        "background_model": "linear",
        "variation_role": "candidate_systematic",
        "is_nominal": False,
    },
    {
        "model_name": "double_gaussian_polynomial_background",
        "signal_model": "double_gaussian",
        "background_model": "polynomial",
        "variation_role": "candidate_systematic",
        "is_nominal": False,
        "implemented": False,
        "pending_reason": "Pending: polynomial background support requires extending the fit-model interface.",
    },
    {
        "model_name": "crystal_ball_exponential",
        "signal_model": "crystal_ball",
        "background_model": "exponential",
        "variation_role": "candidate_systematic",
        "is_nominal": False,
        "implemented": False,
        "pending_reason": "Pending: Crystal Ball signal support requires extending the signal-model interface.",
    },
]


def _configure_matplotlib_cache():
    cache_root = Path(tempfile.gettempdir()) / "lhcb-b2hhh-analysis"
    matplotlib_cache = cache_root / "matplotlib"
    xdg_cache = cache_root / "xdg-cache"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    xdg_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache))


def _status_summary(fit_result):
    status = fit_result.get("fit_status", {})
    return {
        "success": bool(status.get("success")),
        "valid": bool(status.get("valid")),
        "backend": status.get("backend"),
        "message": status.get("message"),
    }


def _fit_charge_sample(
    masses,
    fit_range,
    bins: int,
    signal_model: str,
    background_model: str,
):
    return fit_mass_distribution(
        masses,
        fit_range=fit_range,
        bins=bins,
        signal_model=signal_model,
        background_model=background_model,
    )


def _positive(value) -> bool:
    return value is not None and np.isfinite(value) and value > 0.0


def _fit_quality_acceptance(row, max_chi2_ndof: float):
    reasons = []
    if not row["fit_success_plus"] or not row["fit_valid_plus"]:
        reasons.append("B+ fit did not converge")
    if not row["fit_success_minus"] or not row["fit_valid_minus"]:
        reasons.append("B- fit did not converge")
    if row["chi2_ndof_plus"] is None or row["chi2_ndof_plus"] >= max_chi2_ndof:
        reasons.append(f"B+ chi2/ndof >= {max_chi2_ndof:g}")
    if row["chi2_ndof_minus"] is None or row["chi2_ndof_minus"] >= max_chi2_ndof:
        reasons.append(f"B- chi2/ndof >= {max_chi2_ndof:g}")
    if not _positive(row["N_plus"]):
        reasons.append("B+ fitted yield is not positive")
    if not _positive(row["N_minus"]):
        reasons.append("B- fitted yield is not positive")
    if not _positive(row["sigma_N_plus"]):
        reasons.append("B+ fitted yield uncertainty is not positive")
    if not _positive(row["sigma_N_minus"]):
        reasons.append("B- fitted yield uncertainty is not positive")
    if row["variation_role"] == "stress_test":
        reasons.append("stress-test model, not accepted for systematic assignment")

    return len(reasons) == 0, "; ".join(reasons) if reasons else ""


def _variation_row(
    variation,
    plus_result,
    minus_result,
    n_plus_candidates: int,
    n_minus_candidates: int,
    max_chi2_ndof: float,
):
    asymmetry = _raw_asymmetry(
        plus_result,
        minus_result,
        n_plus_candidates=n_plus_candidates,
        n_minus_candidates=n_minus_candidates,
    )
    plus_status = _status_summary(plus_result)
    minus_status = _status_summary(minus_result)
    row = {
        "model_name": variation["model_name"],
        "signal_model": variation["signal_model"],
        "background_model": variation["background_model"],
        "variation_role": variation["variation_role"],
        "is_nominal": variation["is_nominal"],
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
    accepted, rejection_reason = _fit_quality_acceptance(row, max_chi2_ndof)
    row["accepted_for_systematic"] = accepted
    row["rejection_reason"] = rejection_reason
    return row


def run_variations(
    plus_masses,
    minus_masses,
    n_plus_candidates: int,
    n_minus_candidates: int,
    fit_range,
    bins: int,
    max_chi2_ndof: float,
):
    rows = []
    for variation in MODEL_VARIATIONS:
        if not variation.get("implemented", True):
            print(f"{variation['model_name']}: pending ({variation['pending_reason']})")
            continue
        try:
            plus_result = _fit_charge_sample(
                plus_masses,
                fit_range,
                bins,
                variation["signal_model"],
                variation["background_model"],
            )
            minus_result = _fit_charge_sample(
                minus_masses,
                fit_range,
                bins,
                variation["signal_model"],
                variation["background_model"],
            )
        except NotImplementedError as exc:
            print(f"{variation['model_name']}: pending ({exc})")
            continue
        rows.append(
            _variation_row(
                variation,
                plus_result,
                minus_result,
                n_plus_candidates,
                n_minus_candidates,
                max_chi2_ndof,
            )
        )
        print(
            f"{variation['model_name']}: "
            f"A_raw={rows[-1]['A_raw']:.5f}, "
            f"sigma={rows[-1]['sigma_A_raw']:.5f}, "
            f"accepted={rows[-1]['accepted_for_systematic']}"
        )
    return rows


def write_csv(rows, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(_json_safe(row) for row in rows)


def _summary_entry(row, nominal_a_raw):
    delta = row["A_raw"] - nominal_a_raw if row["A_raw"] is not None else None
    return {
        "model_name": row["model_name"],
        "signal_model": row["signal_model"],
        "background_model": row["background_model"],
        "variation_role": row["variation_role"],
        "A_raw": row["A_raw"],
        "sigma_A_raw": row["sigma_A_raw"],
        "delta_A_raw": delta,
        "is_nominal": row["is_nominal"],
        "accepted_for_systematic": row["accepted_for_systematic"],
        "rejection_reason": row["rejection_reason"],
        "fit_success_plus": row["fit_success_plus"],
        "fit_success_minus": row["fit_success_minus"],
        "chi2_ndof_plus": row["chi2_ndof_plus"],
        "chi2_ndof_minus": row["chi2_ndof_minus"],
    }


def summarize_variations(rows, fit_range, bins: int, max_chi2_ndof: float):
    nominal = next((row for row in rows if row["is_nominal"]), None)
    if nominal is None:
        raise ValueError("No nominal model variation was run.")

    nominal_a_raw = nominal["A_raw"]
    tested_variations = [_summary_entry(row, nominal_a_raw) for row in rows]
    accepted_variations = [
        entry
        for entry in tested_variations
        if entry["accepted_for_systematic"] and not entry["is_nominal"]
    ]
    rejected_stress_tests = [
        entry
        for entry in tested_variations
        if entry["variation_role"] == "stress_test" and not entry["accepted_for_systematic"]
    ]
    rejected_candidate_variations = [
        entry
        for entry in tested_variations
        if (
            entry["variation_role"] == "candidate_systematic"
            and not entry["accepted_for_systematic"]
        )
    ]
    accepted_deltas = [
        abs(entry["delta_A_raw"])
        for entry in accepted_variations
        if entry["delta_A_raw"] is not None
    ]

    pending_variations = [
        {
            "model_name": variation["model_name"],
            "signal_model": variation["signal_model"],
            "background_model": variation["background_model"],
            "variation_role": variation["variation_role"],
            "reason": variation["pending_reason"],
        }
        for variation in MODEL_VARIATIONS
        if not variation.get("implemented", True)
    ]

    recommended = max(accepted_deltas) if accepted_deltas else None
    status = (
        "complete with accepted model variations"
        if accepted_deltas
        else "pending additional accepted model variations"
    )
    max_abs_delta_all_tested = 0.0
    all_deltas = [
        abs(entry["delta_A_raw"])
        for entry in tested_variations
        if entry["delta_A_raw"] is not None
    ]
    if all_deltas:
        max_abs_delta_all_tested = max(all_deltas)

    return {
        "status": status,
        "nominal_model": nominal["model_name"],
        "nominal_A_raw": nominal_a_raw,
        "all_tested_model_variations": tested_variations,
        "accepted_systematic_variations": accepted_variations,
        "rejected_stress_test_variations": rejected_stress_tests,
        "rejected_candidate_systematic_variations": rejected_candidate_variations,
        "pending_model_variations": pending_variations,
        "max_abs_delta_A_raw_all_tested": max_abs_delta_all_tested,
        "max_abs_delta_A_raw_accepted": recommended,
        "recommended_model_systematic": recommended,
        "fit_quality_criteria": {
            "max_chi2_ndof": max_chi2_ndof,
            "require_converged_B_plus": True,
            "require_converged_B_minus": True,
            "require_positive_yields": True,
            "require_positive_yield_uncertainties": True,
        },
        "fit_range": list(fit_range),
        "bins": int(bins),
        "note": (
            "First fit-model systematic with quality filtering. Stress-test models are "
            "reported but are not used as systematic uncertainties unless they provide "
            "an acceptable description of both charge-separated mass spectra."
        ),
    }


def write_variation_plot(rows, output_path: Path):
    _configure_matplotlib_cache()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    set_hep_style()
    labels = [row["model_name"].replace("_", "\n") for row in rows]
    values = np.array([row["A_raw"] for row in rows], dtype=float)
    errors = np.array([row["sigma_A_raw"] for row in rows], dtype=float)
    nominal = next(row for row in rows if row["is_nominal"])
    colors = [
        "#4c78a8" if row["accepted_for_systematic"] else "#d62728"
        for row in rows
    ]

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    x = np.arange(len(rows))
    for i, row in enumerate(rows):
        ax.errorbar(
            x[i],
            values[i],
            yerr=errors[i],
            fmt="o",
            color=colors[i],
            capsize=4,
            label=(
                "Accepted quality"
                if row["accepted_for_systematic"]
                and not row["is_nominal"]
                and not any(
                    previous["accepted_for_systematic"] and not previous["is_nominal"]
                    for previous in rows[:i]
                )
                else None
            ),
        )
    ax.axhline(nominal["A_raw"], color="#4c78a8", linewidth=2.0, label="Nominal")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(r"$A_{\rm raw}$")
    ax.set_title("Fit-model variation of raw asymmetry")
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
        help="Directory for fit-model systematic outputs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_path = args.data.resolve()
    output_dir = args.output_dir.resolve()
    fit_range = tuple(float(value) for value in args.fit_range)
    if not fit_range[0] < fit_range[1]:
        raise ValueError("--fit-range must be an increasing LOW HIGH interval.")
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
        n_plus_candidates=len(b_plus),
        n_minus_candidates=len(b_minus),
        fit_range=fit_range,
        bins=args.bins,
        max_chi2_ndof=args.max_chi2_ndof,
    )
    summary = summarize_variations(rows, fit_range, args.bins, args.max_chi2_ndof)

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "fit_model_variations.csv"
    summary_path = output_dir / "fit_model_summary.json"
    plot_path = output_dir / "fit_model_variations.png"
    write_csv(rows, csv_path)
    summary_path.write_text(json.dumps(_json_safe(summary), indent=2) + "\n", encoding="utf-8")
    write_variation_plot(rows, plot_path)

    recommended = summary["recommended_model_systematic"]
    if recommended is None:
        print("Recommended model systematic: pending additional accepted model variations")
    else:
        print(f"Recommended model systematic: {recommended:.6f}")
    print(f"Wrote {_display_path(csv_path)}")
    print(f"Wrote {_display_path(summary_path)}")
    print(f"Wrote {_display_path(plot_path)}")


if __name__ == "__main__":
    main()
