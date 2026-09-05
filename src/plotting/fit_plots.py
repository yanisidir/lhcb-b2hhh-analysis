"""Plot helpers for invariant-mass fits: model curves, pulls and summary boxes."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.fitting.mass_models import signal_background_binned_model
from src.plotting.mass_plots import configure_matplotlib_cache, set_hep_style


SIGNAL_LABELS = {
    "gaussian": "Gaussian",
    "double_gaussian": "Double Gaussian",
}
BACKGROUND_LABELS = {
    "exponential": "exponential background",
    "linear": "linear background",
}


def model_counts(
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


def fit_summary_text(result) -> str:
    """Return the fitted yield and fit quality, for annotation on the fit plot."""
    yield_uncertainty = result.get("parameter_uncertainties", {}).get("signal_yield")
    if yield_uncertainty is None:
        signal = f"$N_\\mathrm{{sig}} = {result['signal_yield']:.0f}$"
    else:
        signal = f"$N_\\mathrm{{sig}} = {result['signal_yield']:.0f} \\pm {yield_uncertainty:.0f}$"

    lines = [signal, f"$\\mu = {result['mass_mean']:.1f}$ MeV/$c^2$"]
    chi2_ndof = result["fit_quality"]["chi2_ndof"]
    if chi2_ndof is not None:
        lines.append(f"$\\chi^2/\\mathrm{{ndof}} = {chi2_ndof:.2f}$")
    return "\n".join(lines)


def plot_fit(
    masses,
    result,
    output_path: Path,
    fit_range: tuple[float, float],
    bins: int,
    title: str = "Nominal B-candidate invariant-mass fit",
):
    configure_matplotlib_cache()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    set_hep_style()
    counts, bin_edges = np.histogram(masses, bins=bins, range=fit_range)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    expected = model_counts(bin_edges, result, fit_range)
    signal = model_counts(bin_edges, result, fit_range, background_yield=0.0)
    background = model_counts(bin_edges, result, fit_range, signal_yield=0.0)

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
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.text(
        0.025,
        0.95,
        fit_summary_text(result),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        linespacing=1.5,
    )
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
