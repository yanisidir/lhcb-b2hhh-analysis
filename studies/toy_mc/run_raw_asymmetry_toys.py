#!/usr/bin/env python
"""Toy-MC validation of the nominal raw-asymmetry extraction.

Each pseudo-experiment generates B+ and B- spectra for every magnet polarity
from that polarity's simultaneous fit, refits them with the same simultaneous
fit, and averages the polarities as the nominal does. The toys check that the
procedure recovers the injected asymmetry with a correctly calibrated
uncertainty, and give the goodness-of-fit p-value of each data fit.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.fitting.simultaneous import charge_view, fit_polarities
from src.plotting.mass_plots import configure_matplotlib_cache, set_hep_style
from src.utils.io import display_path, json_safe, write_json


DEFAULT_BASELINE_DIR = PROJECT_ROOT / "results" / "fits" / "charge_separated"
POLARITIES = ("MagnetDown", "MagnetUp")
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "toy_mc" / "raw_asymmetry"


def _sample_truncated_gaussian(rng, mean: float, sigma: float, size: int, fit_range):
    low, high = fit_range
    samples = []
    remaining = size
    while remaining > 0:
        draw = rng.normal(mean, sigma, size=max(remaining * 2, 64))
        accepted = draw[(draw >= low) & (draw <= high)]
        if len(accepted) == 0:
            continue
        samples.append(accepted[:remaining])
        remaining -= min(len(accepted), remaining)
    return np.concatenate(samples) if samples else np.empty(0, dtype=float)


def sample_signal_masses(rng, fit_result, n_signal: int, fit_range):
    """Sample signal masses from the fitted Gaussian or double-Gaussian PDF."""
    if n_signal <= 0:
        return np.empty(0, dtype=float)

    signal_model = fit_result["signal_model"]
    mean = float(fit_result["mass_mean"])
    if signal_model == "double_gaussian":
        fraction = float(fit_result["signal_fraction"])
        n_narrow = rng.binomial(n_signal, fraction)
        n_wide = n_signal - n_narrow
        narrow = _sample_truncated_gaussian(
            rng,
            mean,
            float(fit_result["mass_resolution_narrow"]),
            n_narrow,
            fit_range,
        )
        wide = _sample_truncated_gaussian(
            rng,
            mean,
            float(fit_result["mass_resolution_wide"]),
            n_wide,
            fit_range,
        )
        return np.concatenate([narrow, wide])

    return _sample_truncated_gaussian(
        rng,
        mean,
        float(fit_result["mass_resolution"]),
        n_signal,
        fit_range,
    )


def sample_background_masses(rng, fit_result, n_background: int, fit_range):
    """Sample background masses from the fitted finite-range background PDF.

    The shape is taken from ``fit_result["background_model"]``, so the toys are
    generated from the same background the baseline fit used. The two models
    parameterise ``background_slope`` differently -- MeV^-1 for the exponential,
    dimensionless for the linear one -- so generating one with the other's
    slope would silently produce the wrong sample.
    """
    if n_background <= 0:
        return np.empty(0, dtype=float)

    low, high = fit_range
    width = high - low
    slope = float(fit_result["background_slope"])
    background_model = fit_result.get("background_model", "exponential")
    u = rng.random(n_background)

    if background_model == "exponential":
        if abs(slope) < 1.0e-12:
            return low + width * u
        return low + np.log1p(u * np.expm1(slope * width)) / slope

    if background_model == "linear":
        if abs(slope) > 1.0:
            raise ValueError("Linear background slope must satisfy |slope| <= 1.")
        # Invert CDF(t) = t + slope * (t^2 - t) on t = (m - low) / width, using
        # the numerically stable root so that slope -> 0 degrades to t = u.
        discriminant = (1.0 - slope) ** 2 + 4.0 * slope * u
        denominator = (1.0 - slope) + np.sqrt(np.clip(discriminant, 0.0, None))
        # The denominator only vanishes for slope = 1 at u = 0, where t = 0.
        t = np.where(denominator > 0.0, 2.0 * u / np.where(denominator > 0.0, denominator, 1.0), 0.0)
        return low + width * np.clip(t, 0.0, 1.0)

    raise ValueError(
        f"Unsupported background model for toy generation: {background_model!r}."
    )


def generate_toy_masses(rng, fit_result, fit_range):
    """Generate one extended pseudo-experiment from fitted mass-model yields."""
    n_signal = int(rng.poisson(float(fit_result["signal_yield"])))
    n_background = int(rng.poisson(float(fit_result["background_yield"])))
    signal = sample_signal_masses(rng, fit_result, n_signal, fit_range)
    background = sample_background_masses(rng, fit_result, n_background, fit_range)
    masses = np.concatenate([signal, background])
    rng.shuffle(masses)
    return masses, n_signal, n_background


def load_baseline_results(baseline_dir: Path, polarities=POLARITIES):
    """Load the simultaneous data fit of each polarity, used as toy truth."""
    results = {}
    for polarity in polarities:
        path = baseline_dir / f"{polarity}_fit_results.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing baseline file: {path}. Run scripts/run_charge_separated_fits.py first.")
        results[polarity] = json.loads(path.read_text(encoding="utf-8"))
    return results


def _safe_pull(fitted, injected: float, sigma):
    if fitted is None or sigma is None or sigma <= 0:
        return None
    return (fitted - injected) / sigma


def run_toy(toy_index: int, rng, truths, fit_range, bins: int, signal_model: str, background_model: str):
    """Generate every polarity from its fitted model, refit, and average."""
    samples = {}
    for polarity, truth in truths.items():
        plus, _, _ = generate_toy_masses(rng, charge_view(truth, "plus"), fit_range)
        minus, _, _ = generate_toy_masses(rng, charge_view(truth, "minus"), fit_range)
        samples[polarity] = (plus, minus)
    results, combination = fit_polarities(
        samples, fit_range=fit_range, bins=bins, signal_model=signal_model, background_model=background_model
    )

    injected = float(np.mean([truth["A_raw"] for truth in truths.values()]))
    row = {"toy_index": toy_index, "injected_A_raw": injected}
    all_valid = True
    for polarity, result in results.items():
        valid = bool(result["fit_status"].get("valid"))
        all_valid = all_valid and valid
        quality = result.get("fit_quality", {})
        row.update(
            {
                f"fitted_A_raw_{polarity}": result["A_raw"],
                f"sigma_A_raw_{polarity}": result["sigma_A_raw"],
                f"A_raw_pull_{polarity}": _safe_pull(result["A_raw"], truths[polarity]["A_raw"], result["sigma_A_raw"]),
                f"fit_valid_{polarity}": valid,
                f"chi2_likelihood_ratio_{polarity}": quality.get("chi2_likelihood_ratio"),
            }
        )
    row.update(
        {
            "fitted_A_raw": combination["A_raw"],
            "sigma_A_raw": combination["sigma_A_raw"],
            "A_raw_pull": _safe_pull(combination["A_raw"], injected, combination["sigma_A_raw"]),
            "fit_success": all_valid,
            "valid_for_pull_studies": all_valid and combination["sigma_A_raw"] is not None,
        }
    )
    return row


def write_toy_results(rows, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(json_safe(row) for row in rows)


def _finite_array(rows, key: str, valid_only: bool = True):
    values = []
    for row in rows:
        if valid_only and not row["valid_for_pull_studies"]:
            continue
        value = row.get(key)
        if value is not None and np.isfinite(value):
            values.append(float(value))
    return np.asarray(values, dtype=float)


def _pull_statistics(rows, key_suffix: str, injected: float):
    fitted = _finite_array(rows, f"fitted_A_raw{key_suffix}")
    sigma = _finite_array(rows, f"sigma_A_raw{key_suffix}")
    pulls = _finite_array(rows, f"A_raw_pull{key_suffix}")
    mean_fitted = float(np.mean(fitted)) if len(fitted) else None
    return {
        "injected_A_raw": injected,
        "mean_fitted_A_raw": mean_fitted,
        "bias": None if mean_fitted is None else mean_fitted - injected,
        "std_fitted_A_raw": float(np.std(fitted, ddof=1)) if len(fitted) > 1 else None,
        "mean_fitted_uncertainty": float(np.mean(sigma)) if len(sigma) else None,
        "mean_A_raw_pull": float(np.mean(pulls)) if len(pulls) else None,
        "A_raw_pull_width": float(np.std(pulls, ddof=1)) if len(pulls) > 1 else None,
    }


def goodness_of_fit_p_values(rows, truths) -> dict:
    """Toy-based p-value of each polarity's likelihood-ratio chi2.

    The toys are generated from the fitted model and refitted, so the fraction
    whose chi2 is at least the data's is the p-value with the fit itself and
    the low-count bins accounted for, which the asymptotic chi2 formula is not.
    """
    result = {}
    for polarity, truth in truths.items():
        observed = truth.get("fit_quality", {}).get("chi2_likelihood_ratio")
        toys = _finite_array(rows, f"chi2_likelihood_ratio_{polarity}")
        if observed is None or len(toys) == 0:
            result[polarity] = {"observed_chi2_likelihood_ratio": observed, "toy_p_value": None}
            continue
        n_above = int(np.count_nonzero(toys >= float(observed)))
        p_value = n_above / len(toys)
        result[polarity] = {
            "observed_chi2_likelihood_ratio": float(observed),
            "ndof_likelihood_ratio": truth["fit_quality"].get("ndof_likelihood_ratio"),
            "toy_p_value": p_value,
            "toy_p_value_uncertainty": float(np.sqrt(max(p_value * (1 - p_value), 1.0 / len(toys)) / len(toys))),
            "number_of_toys": int(len(toys)),
        }
    return result


def summarize_toys(rows, truths, seed: int, baseline_dir: Path, fit_range, bins: int, signal_model, background_model):
    n_failures = sum(not row["fit_success"] for row in rows)
    n_valid = sum(row["valid_for_pull_studies"] for row in rows)
    injected = float(np.mean([truth["A_raw"] for truth in truths.values()]))
    return {
        "number_of_toys": len(rows),
        "number_of_fit_failures": int(n_failures),
        "number_valid_for_pull_studies": int(n_valid),
        "fit_failure_rate": n_failures / len(rows) if rows else None,
        "polarity_average": _pull_statistics(rows, "", injected),
        "by_polarity": {
            polarity: _pull_statistics(rows, f"_{polarity}", float(truth["A_raw"]))
            for polarity, truth in truths.items()
        },
        "goodness_of_fit": goodness_of_fit_p_values(rows, truths),
        "fit_range": list(fit_range),
        "bins": int(bins),
        "signal_model": signal_model,
        "background_model": background_model,
        "random_seed": seed,
        "baseline_dir": str(display_path(baseline_dir)),
        "note": (
            "Toys generated from each polarity's simultaneous B+/B- fit, refitted the same way and "
            "averaged over polarities. This validates the statistical machinery only, not "
            "detector-level systematics, and cannot detect a bias from the model itself being wrong."
        ),
    }


def _histogram(ax, values, bins: int, title: str, xlabel: str, color: str):
    ax.hist(values, bins=bins, histtype="stepfilled", alpha=0.72, color=color)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Toys")
    ax.grid(True, alpha=0.25)


def write_plots(rows, output_dir: Path, injected_a_raw: float):
    configure_matplotlib_cache()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    set_hep_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    _histogram(ax, _finite_array(rows, "fitted_A_raw"), 30, r"Toy fitted $A_{\rm raw}$, average of polarities",
               r"Fitted $A_{\rm raw}$", "#4c78a8")
    ax.axvline(injected_a_raw, color="black", linewidth=2.0, label="Injected value")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "A_raw_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    _histogram(ax, _finite_array(rows, "A_raw_pull"), 30, r"Toy $A_{\rm raw}$ pull, average of polarities",
               r"$(A_{\rm fit} - A_{\rm inj}) / \sigma_A$", "#72b7b2")
    ax.axvline(0.0, color="black", linewidth=1.8)
    fig.tight_layout()
    fig.savefig(output_dir / "pull_A_raw_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-toys", type=int, default=100, help="Number of pseudo-experiments to generate.")
    parser.add_argument("--seed", type=int, default=12345, help="Random-number seed.")
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=DEFAULT_BASELINE_DIR,
        help="Directory holding the per-polarity simultaneous data fits.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for toy outputs.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.n_toys <= 0:
        raise ValueError("--n-toys must be positive.")
    baseline_dir = args.baseline_dir.resolve()
    output_dir = args.output_dir.resolve()
    truths = load_baseline_results(baseline_dir)

    reference = next(iter(truths.values()))
    fit_range = tuple(float(value) for value in reference["fit_status"]["fit_range"])
    bins = int(reference["fit_status"]["bins"])
    signal_model = reference["signal_model"]
    background_model = reference["background_model"]
    for polarity, truth in truths.items():
        if (truth["signal_model"], truth["background_model"]) != (signal_model, background_model):
            raise ValueError(f"{polarity} was fitted with a different mass model; toys would not reproduce the analysis.")

    rng = np.random.default_rng(args.seed)
    rows = []
    start_time = time.monotonic()
    progress_every = max(1, args.n_toys // 10)
    for toy_index in range(args.n_toys):
        rows.append(run_toy(toy_index, rng, truths, fit_range, bins, signal_model, background_model))
        if (toy_index + 1) % progress_every == 0:
            print(f"Completed {toy_index + 1}/{args.n_toys} toys ({time.monotonic() - start_time:.0f}s)")

    summary = summarize_toys(rows, truths, args.seed, baseline_dir, fit_range, bins, signal_model, background_model)
    write_toy_results(rows, output_dir / "toy_results.csv")
    write_json(output_dir / "toy_summary.json", summary)
    write_plots(rows, output_dir, summary["polarity_average"]["injected_A_raw"])

    average = summary["polarity_average"]
    print(f"Fit failures:             {summary['number_of_fit_failures']}")
    print(f"Bias (average):           {average['bias']:+.6f}")
    print(f"Pull mean / width:        {average['mean_A_raw_pull']:+.3f} / {average['A_raw_pull_width']:.3f}")
    print(f"Fitted sigma vs spread:   {average['mean_fitted_uncertainty']:.5f} / {average['std_fitted_A_raw']:.5f}")
    for polarity, gof in summary["goodness_of_fit"].items():
        print(f"Goodness of fit {polarity:10s} toy p = {gof['toy_p_value']:.3f} +/- {gof['toy_p_value_uncertainty']:.3f}")
    print(f"Wrote {display_path(output_dir / 'toy_summary.json')}")


if __name__ == "__main__":
    main()
