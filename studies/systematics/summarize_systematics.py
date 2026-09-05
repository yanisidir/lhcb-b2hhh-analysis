#!/usr/bin/env python
"""Summarize statistical and systematic uncertainties on the raw asymmetry."""

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

from src.plotting.mass_plots import configure_matplotlib_cache, set_hep_style
from src.utils.io import display_path, json_safe, write_json


DEFAULT_RAW_ASYMMETRY_PATH = (
    PROJECT_ROOT / "results" / "fits" / "charge_separated" / "raw_asymmetry.json"
)
DEFAULT_FIT_MODEL_SUMMARY_PATH = (
    PROJECT_ROOT / "results" / "systematics" / "fit_model_summary.json"
)
DEFAULT_FIT_RANGE_SUMMARY_PATH = (
    PROJECT_ROOT / "results" / "systematics" / "fit_range_summary.json"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "systematics"


def _load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Required input file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _finite_uncertainty(value, label: str) -> float:
    if value is None:
        raise ValueError(f"{label} is missing; run the corresponding systematic study first.")
    value = float(value)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{label} must be a finite non-negative number.")
    return value


def _report_text(
    nominal_a_raw: float,
    statistical_uncertainty: float,
    total_systematic_uncertainty: float,
) -> str:
    return (
        f"A_raw = {nominal_a_raw:.5f} +/- {statistical_uncertainty:.5f} (stat.) "
        f"+/- {total_systematic_uncertainty:.5f} (syst.)"
    )


def build_summary(raw_asymmetry, fit_model_summary, fit_range_summary):
    nominal_a_raw = float(raw_asymmetry["A_raw"])
    statistical_uncertainty = _finite_uncertainty(
        raw_asymmetry.get("sigma_A_raw"),
        "statistical uncertainty",
    )
    fit_model_systematic = _finite_uncertainty(
        fit_model_summary.get("recommended_model_systematic"),
        "fit-model systematic",
    )
    fit_range_systematic = _finite_uncertainty(
        fit_range_summary.get("recommended_fit_range_systematic"),
        "fit-range systematic",
    )

    total_systematic_uncertainty = float(
        np.hypot(fit_model_systematic, fit_range_systematic)
    )
    total_uncertainty = float(
        np.hypot(statistical_uncertainty, total_systematic_uncertainty)
    )

    return {
        "nominal_A_raw": nominal_a_raw,
        "statistical_uncertainty": statistical_uncertainty,
        "fit_model_systematic": fit_model_systematic,
        "fit_range_systematic": fit_range_systematic,
        "total_systematic_uncertainty": total_systematic_uncertainty,
        "total_uncertainty": total_uncertainty,
        "report_text": _report_text(
            nominal_a_raw,
            statistical_uncertainty,
            total_systematic_uncertainty,
        ),
        "inputs": {
            "raw_asymmetry": str(display_path(DEFAULT_RAW_ASYMMETRY_PATH)),
            "fit_model_summary": str(display_path(DEFAULT_FIT_MODEL_SUMMARY_PATH)),
            "fit_range_summary": str(display_path(DEFAULT_FIT_RANGE_SUMMARY_PATH)),
        },
        "notes": (
            "Total systematic and total uncertainties are computed by adding "
            "independent uncertainty components in quadrature. The quoted "
            "asymmetry is raw and is not corrected for detector, production, "
            "trigger, reconstruction, or selection-induced charge asymmetries."
        ),
    }


def write_summary_csv(summary, output_path: Path):
    rows = [
        ("nominal_A_raw", summary["nominal_A_raw"]),
        ("statistical_uncertainty", summary["statistical_uncertainty"]),
        ("fit_model_systematic", summary["fit_model_systematic"]),
        ("fit_range_systematic", summary["fit_range_systematic"]),
        ("total_systematic_uncertainty", summary["total_systematic_uncertainty"]),
        ("total_uncertainty", summary["total_uncertainty"]),
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["quantity", "value"])
        writer.writerows(rows)


def write_breakdown_plot(summary, output_path: Path):
    configure_matplotlib_cache()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [
        "Statistical",
        "Fit model",
        "Fit range",
        "Total systematic",
    ]
    values = np.array(
        [
            summary["statistical_uncertainty"],
            summary["fit_model_systematic"],
            summary["fit_range_systematic"],
            summary["total_systematic_uncertainty"],
        ],
        dtype=float,
    )
    colors = ["#4c78a8", "#f58518", "#54a24b", "#b279a2"]

    set_hep_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    x = np.arange(len(values))
    ax.bar(x, values, color=colors, width=0.62)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel(r"Uncertainty on $A_{\rm raw}$")
    ax.set_title("Raw-asymmetry uncertainty breakdown")
    ax.grid(True, axis="y", alpha=0.25)

    for index, value in enumerate(values):
        ax.text(index, value, f"{value:.4f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-asymmetry",
        type=Path,
        default=DEFAULT_RAW_ASYMMETRY_PATH,
        help="Input raw-asymmetry JSON from the nominal charge-separated fits.",
    )
    parser.add_argument(
        "--fit-model-summary",
        type=Path,
        default=DEFAULT_FIT_MODEL_SUMMARY_PATH,
        help="Input fit-model systematic summary JSON.",
    )
    parser.add_argument(
        "--fit-range-summary",
        type=Path,
        default=DEFAULT_FIT_RANGE_SUMMARY_PATH,
        help="Input fit-range systematic summary JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for systematic summary outputs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    raw_asymmetry_path = args.raw_asymmetry.resolve()
    fit_model_summary_path = args.fit_model_summary.resolve()
    fit_range_summary_path = args.fit_range_summary.resolve()
    output_dir = args.output_dir.resolve()

    raw_asymmetry = _load_json(raw_asymmetry_path)
    fit_model_summary = _load_json(fit_model_summary_path)
    fit_range_summary = _load_json(fit_range_summary_path)

    summary = build_summary(raw_asymmetry, fit_model_summary, fit_range_summary)
    summary["inputs"] = {
        "raw_asymmetry": str(display_path(raw_asymmetry_path)),
        "fit_model_summary": str(display_path(fit_model_summary_path)),
        "fit_range_summary": str(display_path(fit_range_summary_path)),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "systematic_summary.json"
    csv_path = output_dir / "systematic_summary.csv"
    plot_path = output_dir / "systematic_breakdown.png"

    write_json(json_path, summary)
    write_summary_csv(summary, csv_path)
    write_breakdown_plot(summary, plot_path)

    print(summary["report_text"])
    print(f"Total uncertainty: {summary['total_uncertainty']:.5f}")
    print(f"Wrote {display_path(json_path)}")
    print(f"Wrote {display_path(csv_path)}")
    print(f"Wrote {display_path(plot_path)}")


if __name__ == "__main__":
    main()
