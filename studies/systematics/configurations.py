"""Fit configurations shared by the systematic studies, and how each is evaluated.

A configuration is a mass model and a fit range. Evaluating it means fitting
every magnet polarity simultaneously in charge, averaging the raw asymmetry
over the polarities exactly as the nominal does, and applying the
pre-registered acceptance criteria to every fit.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from src.fitting.acceptance import DEFAULT_MAX_CHI2_NDOF, simultaneous_fit_acceptance
from src.fitting.simultaneous import fit_polarities
from src.plotting.mass_plots import configure_matplotlib_cache, set_hep_style
from src.utils.io import json_safe


NOMINAL_FIT_RANGE = (5200.0, 5600.0)

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
        "pending_reason": "Pending: polynomial background support requires extending the fit-model interface.",
    },
    {
        "model_name": "crystal_ball_exponential",
        "signal_model": "crystal_ball",
        "background_model": "exponential",
        "variation_role": "candidate_systematic",
        "is_nominal": False,
        "pending_reason": "Pending: Crystal Ball signal support requires extending the signal-model interface.",
    },
]

FIT_RANGE_VARIATIONS = [
    {"range_name": "nominal_5200_5600", "fit_range": (5200.0, 5600.0), "is_nominal": True},
    {"range_name": "narrower_low_5230_5600", "fit_range": (5230.0, 5600.0), "is_nominal": False},
    {"range_name": "narrower_high_5200_5550", "fit_range": (5200.0, 5550.0), "is_nominal": False},
    {"range_name": "narrower_both_5230_5550", "fit_range": (5230.0, 5550.0), "is_nominal": False},
    {"range_name": "wider_high_5200_5700", "fit_range": (5200.0, 5700.0), "is_nominal": False},
]


def implemented_models():
    """Model variations the fitter supports; pending ones are skipped."""
    return [model for model in MODEL_VARIATIONS if "pending_reason" not in model]


def pending_models():
    return [model for model in MODEL_VARIATIONS if "pending_reason" in model]


def nominal_model():
    return next(model for model in MODEL_VARIATIONS if model["is_nominal"])


def nominal_range():
    return next(variation for variation in FIT_RANGE_VARIATIONS if variation["is_nominal"])


def evaluate_configuration(
    samples_by_polarity,
    model,
    range_variation,
    bins: int = 80,
    max_chi2_ndof: float = DEFAULT_MAX_CHI2_NDOF,
) -> dict:
    """Fit one model x range configuration on every polarity and judge it."""
    fit_range = range_variation["fit_range"]
    results, combination = fit_polarities(
        samples_by_polarity,
        fit_range=fit_range,
        bins=bins,
        signal_model=model["signal_model"],
        background_model=model["background_model"],
    )
    accepted, reason = simultaneous_fit_acceptance(results, max_chi2_ndof)
    row = {
        "model_name": model["model_name"],
        "signal_model": model["signal_model"],
        "background_model": model["background_model"],
        "range_name": range_variation["range_name"],
        "fit_range_low": fit_range[0],
        "fit_range_high": fit_range[1],
        "is_nominal": bool(model["is_nominal"] and range_variation["is_nominal"]),
        "A_raw": combination["A_raw"],
        "sigma_A_raw": combination["sigma_A_raw"],
    }
    for polarity, result in results.items():
        quality = result.get("fit_quality", {})
        row.update(
            {
                f"A_raw_{polarity}": result["A_raw"],
                f"sigma_A_raw_{polarity}": result["sigma_A_raw"],
                f"fit_valid_{polarity}": bool(result["fit_status"].get("valid")),
                f"chi2_ndof_plus_{polarity}": quality.get("plus", {}).get("chi2_ndof"),
                f"chi2_ndof_minus_{polarity}": quality.get("minus", {}).get("chi2_ndof"),
            }
        )
    row["accepted_for_systematic"] = accepted
    row["rejection_reason"] = reason
    return row


def write_rows_csv(rows, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(json_safe(row) for row in rows)


def write_variation_plot(rows, label_key: str, title: str, output_path: Path):
    """Plot the polarity-averaged raw asymmetry of each variation against the nominal."""
    configure_matplotlib_cache()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    set_hep_style()
    nominal = next(row for row in rows if row["is_nominal"])
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    for index, row in enumerate(rows):
        if row["A_raw"] is None:
            continue
        ax.errorbar(
            index,
            row["A_raw"],
            yerr=row["sigma_A_raw"],
            fmt="o",
            color="#4c78a8" if row["accepted_for_systematic"] else "#d62728",
            capsize=4,
        )
    ax.axhline(nominal["A_raw"], color="#4c78a8", linewidth=2.0, label="Nominal")
    ax.set_xticks(np.arange(len(rows)))
    ax.set_xticklabels([row[label_key].replace("_", "\n") for row in rows])
    ax.set_ylabel(r"$A_{\rm raw}$, average of polarities")
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
