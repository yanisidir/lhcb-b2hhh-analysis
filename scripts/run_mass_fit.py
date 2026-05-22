#!/usr/bin/env python
"""Run the first nominal B -> hhh invariant-mass fit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.fitting.fit_mass import fit_mass_distribution
from src.fitting.mass_models import signal_background_binned_model
from src.physics.selection import prepare_selected_candidates
from src.plotting.mass_plots import set_hep_style


DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "B2HHH_MagnetDown.root"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "fits"
SIGNAL_LABELS = {
    "gaussian": "Gaussian",
    "double_gaussian": "Double Gaussian",
}
BACKGROUND_LABELS = {
    "exponential": "exponential background",
    "linear": "linear background",
}


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _display_path(path: Path) -> Path:
    try:
        return path.relative_to(PROJECT_ROOT)
    except ValueError:
        return path


def _model_counts(
    bin_edges,
    result,
    fit_range: tuple[float, float],
    signal_yield: float | None = None,
    background_yield: float | None = None,
):
    signal_model = result.get("signal_model", "gaussian")
    background_model = result.get("background_model", "exponential")
    signal_yield = result["signal_yield"] if signal_yield is None else signal_yield
    background_yield = result["background_yield"] if background_yield is None else background_yield

    if signal_model == "double_gaussian":
        return signal_background_binned_model(
            bin_edges,
            signal_yield,
            background_yield,
            result["mass_mean"],
            result["mass_resolution_narrow"],
            result["background_slope"],
            fit_range=fit_range,
            signal_model=signal_model,
            background_model=background_model,
            sigma_wide=result["mass_resolution_wide"],
            signal_fraction=result["signal_fraction"],
        )

    return signal_background_binned_model(
        bin_edges,
        signal_yield,
        background_yield,
        result["mass_mean"],
        result["mass_resolution"],
        result["background_slope"],
        fit_range=fit_range,
        signal_model=signal_model,
        background_model=background_model,
    )


def _plot_fit(masses, result, output_path: Path, fit_range: tuple[float, float], bins: int):
    import os
    import tempfile

    cache_root = Path(tempfile.gettempdir()) / "lhcb-b2hhh-analysis"
    matplotlib_cache = cache_root / "matplotlib"
    xdg_cache = cache_root / "xdg-cache"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    xdg_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    set_hep_style()
    counts, bin_edges = np.histogram(masses, bins=bins, range=fit_range)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    expected = _model_counts(bin_edges, result, fit_range)
    signal = _model_counts(bin_edges, result, fit_range, background_yield=0.0)
    background = _model_counts(bin_edges, result, fit_range, signal_yield=0.0)

    pulls = np.zeros_like(expected, dtype=float)
    nonzero = counts > 0
    pulls[nonzero] = (counts[nonzero] - expected[nonzero]) / np.sqrt(counts[nonzero])

    signal_label = (
        f"{SIGNAL_LABELS.get(result.get('signal_model'), result.get('signal_model'))}"
        f" + {BACKGROUND_LABELS.get(result.get('background_model', 'exponential'), result.get('background_model'))}"
    )

    fig, (ax, ax_pull) = plt.subplots(
        2,
        1,
        figsize=(7.4, 6.2),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [3.2, 1.0], "hspace": 0.06},
    )
    ax.hist(
        masses,
        bins=bins,
        range=fit_range,
        histtype="stepfilled",
        alpha=0.62,
        label="Selected candidates",
    )
    ax.stairs(expected, bin_edges, color="black", linewidth=2.0, label=signal_label)
    ax.stairs(signal, bin_edges, color="#4c78a8", linewidth=1.7, linestyle="--", label="Signal")
    ax.stairs(
        background,
        bin_edges,
        color="#f58518",
        linewidth=1.7,
        linestyle=":",
        label="Background",
    )
    ax.set_ylabel("Candidates / bin")
    ax.set_title("First nominal B-candidate invariant-mass fit")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)

    ax_pull.axhline(0.0, color="black", linewidth=1.0)
    ax_pull.axhline(3.0, color="gray", linewidth=0.8, linestyle=":")
    ax_pull.axhline(-3.0, color="gray", linewidth=0.8, linestyle=":")
    ax_pull.scatter(bin_centers, pulls, s=14, color="#333333")
    ax_pull.set_xlabel(r"$m(hhh)$ [MeV/$c^2$]")
    ax_pull.set_ylabel("Pull")
    ax_pull.set_ylim(-5.5, 5.5)
    ax_pull.grid(True, alpha=0.25)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help="Input ROOT ntuple.")
    parser.add_argument("--bins", type=int, default=80, help="Number of fitted mass bins.")
    parser.add_argument(
        "--signal-model",
        choices=("gaussian", "double_gaussian"),
        default="double_gaussian",
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
        default=(5000.0, 5600.0),
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

    results_path.write_text(json.dumps(_json_safe(result), indent=2) + "\n", encoding="utf-8")
    _plot_fit(masses, result, plot_path, fit_range=fit_range, bins=args.bins)

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
    print(f"Wrote {_display_path(results_path)}")
    print(f"Wrote {_display_path(plot_path)}")


if __name__ == "__main__":
    main()
