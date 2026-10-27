#!/usr/bin/env python
"""Test whether the B+ and B- mass spectra have the same shape, per polarity.

The MagnetDown B+ spectrum shows an excess on the low-mass flank of the peak,
between 5230 and 5260 MeV/c^2, that the B- spectrum does not. This study asks
whether that is a feature of the data or a fluctuation, in two ways:

* a likelihood-ratio test of the simultaneous fit, where the two charges share
  their shape parameters, against separate fits, where each charge has its
  own -- calibrated with pseudo-experiments generated under the shared-shape
  hypothesis rather than read off the asymptotic chi2 distribution;
* the same test on the independent MagnetUp sample, which a genuine physics
  effect would also show in B+, a polarity-dependent detector effect would
  show in B-, and a fluctuation would not show at all.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.fitting.fit_mass import fit_mass_distribution
from src.fitting.mass_models import signal_background_binned_model
from src.fitting.simultaneous import charge_view, fit_charge_simultaneous
from src.physics.selection import load_charge_samples
from src.plotting.mass_plots import configure_matplotlib_cache, set_hep_style
from src.utils.io import DEFAULT_DATA_PATHS, display_path, write_json
from studies.toy_mc.run_raw_asymmetry_toys import generate_toy_masses


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "charge_shape"
FIT_RANGE = (5200.0, 5600.0)
BINS = 80
FLANK = (5220.0, 5260.0)
# mean, resolution and background slope are shared in the simultaneous fit
N_SHARED_SHAPE_PARAMETERS = 3


def shape_test_statistic(plus_masses, minus_masses):
    """Return 2 * (NLL_shared - NLL_separate), the shared fit, and whether all fits are valid."""
    shared = fit_charge_simultaneous(plus_masses, minus_masses, fit_range=FIT_RANGE, bins=BINS)
    plus = fit_mass_distribution(plus_masses, fit_range=FIT_RANGE, bins=BINS)
    minus = fit_mass_distribution(minus_masses, fit_range=FIT_RANGE, bins=BINS)
    separate_nll = plus["fit_status"]["negative_log_likelihood"] + minus["fit_status"]["negative_log_likelihood"]
    statistic = 2.0 * (shared["fit_status"]["negative_log_likelihood"] - separate_nll)
    valid = shared["fit_status"]["valid"] and plus["fit_status"]["valid"] and minus["fit_status"]["valid"]
    return statistic, shared, (plus, minus), bool(valid)


def toy_p_value(shared, observed: float, n_toys: int, rng):
    """Fraction of shared-shape pseudo-experiments with a statistic at least as large."""
    statistics = []
    for _ in range(n_toys):
        plus, _, _ = generate_toy_masses(rng, charge_view(shared, "plus"), FIT_RANGE)
        minus, _, _ = generate_toy_masses(rng, charge_view(shared, "minus"), FIT_RANGE)
        statistic, _, _, valid = shape_test_statistic(plus, minus)
        if valid:
            statistics.append(statistic)
    statistics = np.asarray(statistics)
    n_above = int(np.count_nonzero(statistics >= observed))
    p_value = n_above / len(statistics)
    return {
        "toy_p_value": p_value,
        "toy_p_value_uncertainty": float(np.sqrt(max(p_value * (1 - p_value), 1.0 / len(statistics)) / len(statistics))),
        "number_of_toys": int(len(statistics)),
        "mean_toy_statistic": float(np.mean(statistics)),
    }


def analyse_polarity(plus_masses, minus_masses, n_toys: int, rng):
    from scipy.stats import chi2

    statistic, shared, (plus, minus), valid = shape_test_statistic(plus_masses, minus_masses)
    in_flank = lambda masses: int(np.count_nonzero((masses >= FLANK[0]) & (masses < FLANK[1])))
    _, edges = np.histogram([], bins=BINS, range=FIT_RANGE)
    flank_bins = (edges[:-1] >= FLANK[0]) & (edges[1:] <= FLANK[1])
    expected = {}
    for charge in ("plus", "minus"):
        view = charge_view(shared, charge)
        # Expected counts of the shared-shape fit in the flank, from its binned model.
        counts = signal_background_binned_model(
            edges,
            view["signal_yield"],
            view["background_yield"],
            view["mass_mean"],
            view["mass_resolution"],
            view["background_slope"],
            fit_range=FIT_RANGE,
        )
        expected[charge] = float(counts[flank_bins].sum())

    return {
        "shape_test_statistic": float(statistic),
        "n_shared_shape_parameters": N_SHARED_SHAPE_PARAMETERS,
        "asymptotic_p_value": float(chi2.sf(statistic, N_SHARED_SHAPE_PARAMETERS)),
        **toy_p_value(shared, statistic, n_toys, rng),
        "all_fits_valid": valid,
        "flank": list(FLANK),
        "flank_observed": {"plus": in_flank(plus_masses), "minus": in_flank(minus_masses)},
        "flank_expected_shared_shape": expected,
        "separate_fit_peak_position": {
            "plus": [plus["mass_mean"], plus["parameter_uncertainties"]["mass_mean"]],
            "minus": [minus["mass_mean"], minus["parameter_uncertainties"]["mass_mean"]],
        },
        "separate_fit_background_yield": {
            "plus": [plus["background_yield"], plus["parameter_uncertainties"]["background_yield"]],
            "minus": [minus["background_yield"], minus["parameter_uncertainties"]["background_yield"]],
        },
        "separate_fit_background_slope": {
            "plus": [plus["background_slope"], plus["parameter_uncertainties"]["background_slope"]],
            "minus": [minus["background_slope"], minus["parameter_uncertainties"]["background_slope"]],
        },
    }


def write_flank_plot(samples, output_path: Path):
    """B+ and B- spectra on the low-mass flank, per polarity, B- scaled to the B+ peak."""
    configure_matplotlib_cache()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    set_hep_style()
    edges = np.arange(5150.0, 5330.0, 10.0)
    centres = 0.5 * (edges[1:] + edges[:-1])
    fig, axes = plt.subplots(1, len(samples), figsize=(5.6 * len(samples), 4.4), sharey=True)
    for ax, (polarity, (plus, minus)) in zip(np.atleast_1d(axes), samples.items()):
        n_plus, _ = np.histogram(plus, edges)
        n_minus, _ = np.histogram(minus, edges)
        peak = (centres > 5260) & (centres < 5310)
        scale = n_plus[peak].sum() / max(n_minus[peak].sum(), 1)
        ax.errorbar(centres - 1.2, n_plus, yerr=np.sqrt(n_plus), fmt="o", ms=4, label=r"$B^+$")
        ax.errorbar(
            centres + 1.2,
            n_minus * scale,
            yerr=np.sqrt(n_minus) * scale,
            fmt="s",
            ms=4,
            label=rf"$B^-$ $\times$ {scale:.2f}",
        )
        ax.axvspan(*FLANK, color="0.9", zorder=0)
        ax.set_yscale("log")
        ax.set_title(polarity)
        ax.set_xlabel(r"$m(hhh)$ [MeV/$c^2$]")
        ax.legend(frameon=False, loc="upper left")
    np.atleast_1d(axes)[0].set_ylabel(r"Candidates / 10 MeV$/c^2$")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-toys", type=int, default=1000, help="Pseudo-experiments per polarity.")
    parser.add_argument("--seed", type=int, default=2024, help="Random-number seed.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir.resolve()
    samples = load_charge_samples(DEFAULT_DATA_PATHS)
    rng = np.random.default_rng(args.seed)

    start = time.monotonic()
    by_polarity = {
        polarity: analyse_polarity(plus, minus, args.n_toys, rng) for polarity, (plus, minus) in samples.items()
    }
    summary = {
        "study": "B+/B- mass-shape compatibility, per magnet polarity",
        "fit_range": list(FIT_RANGE),
        "bins": BINS,
        "random_seed": args.seed,
        "by_polarity": by_polarity,
        "note": (
            "The flank window was chosen after looking at the MagnetDown spectrum, so its "
            "local significance overstates the evidence; MagnetUp is the independent test."
        ),
    }
    summary_path = write_json(output_dir / "charge_shape_summary.json", summary)
    plot_path = output_dir / "low_mass_flank.png"
    write_flank_plot(samples, plot_path)

    for polarity, result in by_polarity.items():
        observed, expected = result["flank_observed"], result["flank_expected_shared_shape"]
        print(
            f"{polarity:11s} shape test 2dNLL = {result['shape_test_statistic']:5.2f}   "
            f"toy p = {result['toy_p_value']:.3f} +/- {result['toy_p_value_uncertainty']:.3f}   "
            f"flank B+ {observed['plus']} (exp. {expected['plus']:.1f})  B- {observed['minus']} (exp. {expected['minus']:.1f})"
        )
    print(f"({time.monotonic() - start:.0f}s)")
    print(f"Wrote {display_path(summary_path)}")
    print(f"Wrote {display_path(plot_path)}")


if __name__ == "__main__":
    main()
