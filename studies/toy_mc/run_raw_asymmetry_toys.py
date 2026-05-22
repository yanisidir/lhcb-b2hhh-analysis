#!/usr/bin/env python
"""Toy-MC validation for fitted raw charge asymmetry extraction."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.fitting.fit_mass import fit_mass_distribution
from src.plotting.mass_plots import set_hep_style


DEFAULT_BASELINE_DIR = PROJECT_ROOT / "results" / "fits" / "charge_separated"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "toy_mc" / "raw_asymmetry"


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


def _configure_matplotlib_cache():
    cache_root = Path(tempfile.gettempdir()) / "lhcb-b2hhh-analysis"
    matplotlib_cache = cache_root / "matplotlib"
    xdg_cache = cache_root / "xdg-cache"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    xdg_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache))


def _load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing baseline file: {path}. Run scripts/run_charge_separated_fits.py first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def load_baseline_results(baseline_dir: Path):
    """Load nominal charge-separated fit results used as toy truth."""
    plus = _load_json(baseline_dir / "B_plus_fit_results.json")
    minus = _load_json(baseline_dir / "B_minus_fit_results.json")
    return plus, minus


def compute_raw_asymmetry(n_plus: float, sigma_plus: float | None, n_minus: float, sigma_minus: float | None):
    """Compute A_raw and its independent-yield propagated uncertainty."""
    denominator = n_minus + n_plus
    if denominator <= 0:
        return None, None

    asymmetry = (n_minus - n_plus) / denominator
    if sigma_plus is None or sigma_minus is None:
        return asymmetry, None

    d_a_d_n_plus = -2.0 * n_minus / denominator**2
    d_a_d_n_minus = 2.0 * n_plus / denominator**2
    sigma = np.sqrt((d_a_d_n_plus * sigma_plus) ** 2 + (d_a_d_n_minus * sigma_minus) ** 2)
    return asymmetry, float(sigma)


def signal_yield_uncertainty(fit_result) -> float | None:
    if fit_result is None:
        return None
    uncertainty = fit_result.get("parameter_uncertainties", {}).get("signal_yield")
    return None if uncertainty is None else float(uncertainty)


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
    """Sample background masses from the finite-range exponential PDF."""
    if n_background <= 0:
        return np.empty(0, dtype=float)

    low, high = fit_range
    width = high - low
    slope = float(fit_result["background_slope"])
    u = rng.random(n_background)
    if abs(slope) < 1.0e-12:
        return low + width * u
    return low + np.log1p(u * np.expm1(slope * width)) / slope


def generate_toy_masses(rng, fit_result, fit_range):
    """Generate one extended pseudo-experiment from fitted mass-model yields."""
    n_signal = int(rng.poisson(float(fit_result["signal_yield"])))
    n_background = int(rng.poisson(float(fit_result["background_yield"])))
    signal = sample_signal_masses(rng, fit_result, n_signal, fit_range)
    background = sample_background_masses(rng, fit_result, n_background, fit_range)
    masses = np.concatenate([signal, background])
    rng.shuffle(masses)
    return masses, n_signal, n_background


def _safe_pull(fitted: float | None, injected: float, sigma: float | None):
    if fitted is None or sigma is None or sigma <= 0:
        return None
    return (fitted - injected) / sigma


def _failed_fit_result(message: str):
    return {
        "signal_yield": None,
        "parameter_uncertainties": {"signal_yield": None},
        "fit_quality": {"chi2_ndof": None},
        "fit_status": {
            "backend": None,
            "success": False,
            "valid": False,
            "message": message,
        },
    }


def _fit_toy_sample(masses, fit_range, bins: int, signal_model: str):
    try:
        return fit_mass_distribution(masses, fit_range=fit_range, bins=bins, signal_model=signal_model)
    except Exception as exc:
        return _failed_fit_result(f"{type(exc).__name__}: {exc}")


def _fit_signal_yield(fit_result):
    value = fit_result.get("signal_yield")
    return None if value is None else float(value)


def _fit_success(fit_result):
    status = fit_result.get("fit_status", {})
    return bool(status.get("success")) and bool(status.get("valid"))


def run_toy(toy_index: int, rng, plus_truth, minus_truth, fit_range, bins: int, signal_model: str):
    plus_masses, generated_signal_plus, generated_background_plus = generate_toy_masses(
        rng,
        plus_truth,
        fit_range,
    )
    minus_masses, generated_signal_minus, generated_background_minus = generate_toy_masses(
        rng,
        minus_truth,
        fit_range,
    )

    plus_fit = _fit_toy_sample(plus_masses, fit_range, bins, signal_model)
    minus_fit = _fit_toy_sample(minus_masses, fit_range, bins, signal_model)

    injected_n_plus = float(plus_truth["signal_yield"])
    injected_n_minus = float(minus_truth["signal_yield"])
    injected_a_raw, _ = compute_raw_asymmetry(injected_n_plus, None, injected_n_minus, None)

    fitted_n_plus = _fit_signal_yield(plus_fit)
    fitted_n_minus = _fit_signal_yield(minus_fit)
    sigma_n_plus = signal_yield_uncertainty(plus_fit)
    sigma_n_minus = signal_yield_uncertainty(minus_fit)
    if fitted_n_plus is None or fitted_n_minus is None:
        fitted_a_raw = None
        sigma_a_raw = None
    else:
        fitted_a_raw, sigma_a_raw = compute_raw_asymmetry(
            fitted_n_plus,
            sigma_n_plus,
            fitted_n_minus,
            sigma_n_minus,
        )

    plus_success = _fit_success(plus_fit)
    minus_success = _fit_success(minus_fit)
    valid_for_pull_studies = (
        plus_success
        and minus_success
        and fitted_a_raw is not None
        and sigma_a_raw is not None
        and sigma_a_raw > 0
        and sigma_n_plus is not None
        and sigma_n_plus > 0
        and sigma_n_minus is not None
        and sigma_n_minus > 0
    )

    return {
        "toy_index": toy_index,
        "injected_N_plus": injected_n_plus,
        "injected_N_minus": injected_n_minus,
        "generated_signal_plus": generated_signal_plus,
        "generated_background_plus": generated_background_plus,
        "generated_signal_minus": generated_signal_minus,
        "generated_background_minus": generated_background_minus,
        "fitted_N_plus": fitted_n_plus,
        "fitted_N_minus": fitted_n_minus,
        "sigma_N_plus": sigma_n_plus,
        "sigma_N_minus": sigma_n_minus,
        "injected_A_raw": injected_a_raw,
        "fitted_A_raw": fitted_a_raw,
        "sigma_A_raw": sigma_a_raw,
        "A_raw_pull": _safe_pull(fitted_a_raw, injected_a_raw, sigma_a_raw),
        "N_plus_pull": _safe_pull(fitted_n_plus, injected_n_plus, sigma_n_plus),
        "N_minus_pull": _safe_pull(fitted_n_minus, injected_n_minus, sigma_n_minus),
        "fit_success": plus_success and minus_success,
        "valid_for_pull_studies": valid_for_pull_studies,
        "fit_success_plus": plus_success,
        "fit_success_minus": minus_success,
        "fit_valid_plus": bool(plus_fit["fit_status"].get("valid")),
        "fit_valid_minus": bool(minus_fit["fit_status"].get("valid")),
        "fit_backend_plus": plus_fit["fit_status"].get("backend"),
        "fit_backend_minus": minus_fit["fit_status"].get("backend"),
        "fit_status_plus": plus_fit["fit_status"].get("message"),
        "fit_status_minus": minus_fit["fit_status"].get("message"),
        "chi2_ndof_plus": plus_fit.get("fit_quality", {}).get("chi2_ndof"),
        "chi2_ndof_minus": minus_fit.get("fit_quality", {}).get("chi2_ndof"),
    }


def write_toy_results(rows, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(_json_safe(row) for row in rows)


def _finite_array(rows, key: str, valid_only: bool = True):
    values = []
    for row in rows:
        if valid_only and not row["valid_for_pull_studies"]:
            continue
        value = row.get(key)
        if value is not None and np.isfinite(value):
            values.append(float(value))
    return np.asarray(values, dtype=float)


def _toy_statistics(rows, injected_a_raw: float, valid_only: bool):
    selected_rows = [row for row in rows if row["valid_for_pull_studies"]] if valid_only else rows
    fitted_a = _finite_array(selected_rows, "fitted_A_raw", valid_only=False)
    sigma_a = _finite_array(selected_rows, "sigma_A_raw", valid_only=False)
    pulls = _finite_array(selected_rows, "A_raw_pull", valid_only=False)
    n_plus_pulls = _finite_array(selected_rows, "N_plus_pull", valid_only=False)
    n_minus_pulls = _finite_array(selected_rows, "N_minus_pull", valid_only=False)
    mean_fitted = float(np.mean(fitted_a)) if len(fitted_a) else None
    bias = (mean_fitted - injected_a_raw) if mean_fitted is not None else None
    return {
        "number_of_toys": len(selected_rows),
        "number_with_fitted_A_raw": int(len(fitted_a)),
        "mean_fitted_A_raw": mean_fitted,
        "injected_A_raw": injected_a_raw,
        "bias": bias,
        "std_fitted_A_raw": float(np.std(fitted_a, ddof=1)) if len(fitted_a) > 1 else None,
        "mean_fitted_uncertainty": float(np.mean(sigma_a)) if len(sigma_a) else None,
        "mean_A_raw_pull": float(np.mean(pulls)) if len(pulls) else None,
        "A_raw_pull_width": float(np.std(pulls, ddof=1)) if len(pulls) > 1 else None,
        "mean_N_plus_pull": float(np.mean(n_plus_pulls)) if len(n_plus_pulls) else None,
        "N_plus_pull_width": float(np.std(n_plus_pulls, ddof=1)) if len(n_plus_pulls) > 1 else None,
        "mean_N_minus_pull": float(np.mean(n_minus_pulls)) if len(n_minus_pulls) else None,
        "N_minus_pull_width": float(np.std(n_minus_pulls, ddof=1)) if len(n_minus_pulls) > 1 else None,
    }


def summarize_toys(rows, injected_a_raw: float, seed: int, baseline_dir: Path, fit_range, bins: int):
    n_failures = sum(not row["fit_success"] for row in rows)
    n_valid = sum(row["valid_for_pull_studies"] for row in rows)

    return {
        "number_of_toys": len(rows),
        "injected_A_raw": injected_a_raw,
        "number_of_fit_failures": int(n_failures),
        "number_valid_for_pull_studies": int(n_valid),
        "fit_failure_rate": n_failures / len(rows) if rows else None,
        "valid_for_pull_studies_rate": n_valid / len(rows) if rows else None,
        "all_toys": _toy_statistics(rows, injected_a_raw, valid_only=False),
        "valid_toys": _toy_statistics(rows, injected_a_raw, valid_only=True),
        "fit_range": list(fit_range),
        "bins": int(bins),
        "random_seed": seed,
        "baseline_dir": str(_display_path(baseline_dir)),
        "note": (
            "Toy-MC validation generated from the nominal charge-separated mass model. "
            "This validates the statistical machinery only, not detector-level systematics."
        ),
    }


def _histogram(ax, values, bins: int, title: str, xlabel: str, color: str):
    ax.hist(values, bins=bins, histtype="stepfilled", alpha=0.72, color=color)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Toys")
    ax.grid(True, alpha=0.25)


def write_plots(rows, output_dir: Path, injected_a_raw: float):
    _configure_matplotlib_cache()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    set_hep_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    fitted_a = _finite_array(rows, "fitted_A_raw", valid_only=True)
    a_pulls = _finite_array(rows, "A_raw_pull", valid_only=True)
    n_plus_pulls = _finite_array(rows, "N_plus_pull", valid_only=True)
    n_minus_pulls = _finite_array(rows, "N_minus_pull", valid_only=True)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    _histogram(ax, fitted_a, bins=24, title=r"Toy fitted $A_{\rm raw}$", xlabel=r"Fitted $A_{\rm raw}$", color="#4c78a8")
    ax.axvline(injected_a_raw, color="black", linewidth=2.0, label="Injected value")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "A_raw_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    _histogram(ax, a_pulls, bins=24, title=r"Toy $A_{\rm raw}$ pull", xlabel=r"$(A_{\rm fit} - A_{\rm inj}) / \sigma_A$", color="#72b7b2")
    ax.axvline(0.0, color="black", linewidth=1.8)
    fig.tight_layout()
    fig.savefig(output_dir / "pull_A_raw_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.hist(n_plus_pulls, bins=24, histtype="stepfilled", alpha=0.55, label=r"$N_+$ pull")
    ax.hist(n_minus_pulls, bins=24, histtype="stepfilled", alpha=0.55, label=r"$N_-$ pull")
    ax.axvline(0.0, color="black", linewidth=1.8)
    ax.set_title("Toy signal-yield pulls")
    ax.set_xlabel(r"$(N_{\rm fit} - N_{\rm inj}) / \sigma_N$")
    ax.set_ylabel("Toys")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "yield_pull_distributions.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-toys", type=int, default=100, help="Number of pseudo-experiments to generate.")
    parser.add_argument("--seed", type=int, default=12345, help="Random-number seed.")
    parser.add_argument("--bins", type=int, default=None, help="Number of fitted mass bins.")
    parser.add_argument(
        "--fit-range",
        type=float,
        nargs=2,
        default=None,
        metavar=("LOW", "HIGH"),
        help="Fitted invariant-mass range in MeV/c^2.",
    )
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=DEFAULT_BASELINE_DIR,
        help="Directory containing nominal charge-separated fit results.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for toy-MC validation outputs.",
    )
    return parser.parse_args()


def _format_optional(value, precision: int = 5):
    return "n/a" if value is None else f"{value:.{precision}f}"


def main():
    args = parse_args()
    if args.n_toys <= 0:
        raise ValueError("--n-toys must be positive.")

    baseline_dir = args.baseline_dir.resolve()
    output_dir = args.output_dir.resolve()
    plus_truth, minus_truth = load_baseline_results(baseline_dir)

    fit_range = (
        tuple(float(value) for value in args.fit_range)
        if args.fit_range is not None
        else tuple(float(value) for value in plus_truth["fit_status"]["fit_range"])
    )
    bins = int(args.bins) if args.bins is not None else int(plus_truth["fit_status"]["bins"])
    if bins <= 0:
        raise ValueError("--bins must be positive.")
    if not fit_range[0] < fit_range[1]:
        raise ValueError("--fit-range must be an increasing LOW HIGH interval.")

    signal_model = plus_truth["signal_model"]
    injected_a_raw, _ = compute_raw_asymmetry(
        float(plus_truth["signal_yield"]),
        None,
        float(minus_truth["signal_yield"]),
        None,
    )

    rng = np.random.default_rng(args.seed)
    rows = []
    start_time = time.monotonic()
    progress_every = max(1, min(25, args.n_toys // 10))
    for toy_index in range(args.n_toys):
        rows.append(run_toy(toy_index, rng, plus_truth, minus_truth, fit_range, bins, signal_model))
        if (toy_index + 1) % progress_every == 0 or toy_index + 1 == args.n_toys:
            elapsed = time.monotonic() - start_time
            n_failures = sum(not row["fit_success"] for row in rows)
            n_valid = sum(row["valid_for_pull_studies"] for row in rows)
            print(
                f"Completed {toy_index + 1}/{args.n_toys} toys "
                f"({elapsed:.1f}s elapsed, failures={n_failures}, valid={n_valid})"
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_toys(rows, injected_a_raw, args.seed, baseline_dir, fit_range, bins)
    write_toy_results(rows, output_dir / "toy_results.csv")
    (output_dir / "toy_summary.json").write_text(
        json.dumps(_json_safe(summary), indent=2) + "\n",
        encoding="utf-8",
    )
    write_plots(rows, output_dir, injected_a_raw)

    valid_summary = summary["valid_toys"]
    print(f"Injected A_raw:           {injected_a_raw:.5f}")
    print(f"Mean fitted A_raw:        {_format_optional(valid_summary['mean_fitted_A_raw'])}")
    print(f"Bias:                     {_format_optional(valid_summary['bias'])}")
    print(f"Mean pull:                {_format_optional(valid_summary['mean_A_raw_pull'], precision=3)}")
    print(f"Pull width:               {_format_optional(valid_summary['A_raw_pull_width'], precision=3)}")
    print(f"Fit failure rate:         {_format_optional(summary['fit_failure_rate'], precision=3)}")
    print(f"Valid toy rate:           {_format_optional(summary['valid_for_pull_studies_rate'], precision=3)}")
    print(f"Wrote {_display_path(output_dir / 'toy_results.csv')}")
    print(f"Wrote {_display_path(output_dir / 'toy_summary.json')}")
    print(f"Wrote {_display_path(output_dir / 'A_raw_distribution.png')}")
    print(f"Wrote {_display_path(output_dir / 'pull_A_raw_distribution.png')}")
    print(f"Wrote {_display_path(output_dir / 'yield_pull_distributions.png')}")


if __name__ == "__main__":
    main()
