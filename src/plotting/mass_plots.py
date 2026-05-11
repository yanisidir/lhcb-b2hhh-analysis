"""Publication-style plotting helpers for reconstructed B-mass spectra."""

from __future__ import annotations

from pathlib import Path


def set_hep_style():
    """Apply a compact, readable matplotlib style for analysis figures."""
    import matplotlib.pyplot as plt

    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "legend.fontsize": 10,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _save_figure(fig, output_path: str | Path | None):
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")


def plot_mass_distribution(
    masses,
    bins: int = 120,
    mass_range: tuple[float, float] = (5000, 5800),
    title: str = "Reconstructed B-candidate invariant mass",
    label: str = "Candidates",
    output_path: str | Path | None = None,
):
    """Plot a reconstructed invariant-mass distribution."""
    import matplotlib.pyplot as plt

    set_hep_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.hist(masses, bins=bins, range=mass_range, histtype="stepfilled", alpha=0.72)
    ax.set_xlabel(r"$m(hhh)$ [MeV/$c^2$]")
    ax.set_ylabel("Candidates")
    ax.set_title(title)
    ax.legend([label], frameon=False)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    _save_figure(fig, output_path)
    return fig, ax


def plot_mass_fit(
    masses,
    model_x,
    model_y,
    bins: int = 100,
    mass_range: tuple[float, float] = (5200, 5350),
    title: str = "B-candidate mass fit after selection",
    output_path: str | Path | None = None,
):
    """Plot the mass spectrum with an overlaid signal-plus-background model."""
    import matplotlib.pyplot as plt

    set_hep_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.hist(
        masses,
        bins=bins,
        range=mass_range,
        histtype="stepfilled",
        alpha=0.65,
        label="Selected candidates",
    )
    ax.plot(model_x, model_y, color="black", linewidth=2.0, label="Gaussian + linear background")
    ax.set_xlabel(r"$m(hhh)$ [MeV/$c^2$]")
    ax.set_ylabel("Candidates")
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    _save_figure(fig, output_path)
    return fig, ax
