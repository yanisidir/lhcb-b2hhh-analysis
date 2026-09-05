"""Reusable binned invariant-mass fits for selected B -> hhh candidates."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from src.fitting.mass_models import (
    SUPPORTED_BACKGROUND_MODELS,
    SUPPORTED_SIGNAL_MODELS,
    signal_background_binned_model,
)


PARAMETER_LABELS_BY_MODEL = {
    "gaussian": {
        "signal_yield": "signal_yield",
        "background_yield": "background_yield",
        "mean": "mass_mean",
        "sigma": "mass_resolution",
        "slope": "background_slope",
    },
    "double_gaussian": {
        "signal_yield": "signal_yield",
        "background_yield": "background_yield",
        "mean": "mass_mean",
        "sigma_narrow": "mass_resolution_narrow",
        "sigma_wide": "mass_resolution_wide",
        "signal_fraction": "signal_fraction",
        "slope": "background_slope",
    },
}


def _validate_signal_model(signal_model: str) -> str:
    if signal_model not in SUPPORTED_SIGNAL_MODELS:
        choices = ", ".join(sorted(SUPPORTED_SIGNAL_MODELS))
        raise ValueError(f"signal_model must be one of: {choices}.")
    return signal_model


def _validate_background_model(background_model: str) -> str:
    if background_model not in SUPPORTED_BACKGROUND_MODELS:
        choices = ", ".join(sorted(SUPPORTED_BACKGROUND_MODELS))
        raise ValueError(f"background_model must be one of: {choices}.")
    return background_model


def _clean_masses(masses, fit_range: tuple[float, float]) -> np.ndarray:
    low, high = fit_range
    values = np.asarray(masses, dtype=float)
    mask = np.isfinite(values) & (values >= low) & (values <= high)
    return values[mask]


def _initial_values(
    values: np.ndarray,
    counts: np.ndarray,
    fit_range: tuple[float, float],
    signal_model: str,
    background_model: str,
):
    low, high = fit_range
    total = float(counts.sum())
    central_low = max(low, 5260.0)
    central_high = min(high, 5300.0)

    central_values = values[(values >= central_low) & (values <= central_high)]
    mean = float(np.median(central_values)) if len(central_values) else 5280.0
    mean = float(np.clip(mean, low + 1.0, high - 1.0))

    if len(central_values) > 2:
        sigma_seed = float(np.std(central_values))
    else:
        sigma_seed = 15.0
    sigma_seed = float(np.clip(sigma_seed, 5.0, 40.0))

    central_count = float(len(central_values))
    central_width = max(central_high - central_low, 1.0)
    sideband_width = max((high - low) - central_width, 1.0)
    sideband_count = max(total - central_count, 0.0)
    background_in_central = sideband_count * central_width / sideband_width
    signal_yield = max(central_count - background_in_central, 0.10 * total, 1.0)
    background_yield = max(total - signal_yield, 1.0)

    initial = {
        "signal_yield": signal_yield,
        "background_yield": background_yield,
        "mean": mean,
        "slope": -0.001 if background_model == "exponential" else 0.0,
    }

    if signal_model == "gaussian":
        initial["sigma"] = sigma_seed
    else:
        sigma_narrow = float(np.clip(0.65 * sigma_seed, 4.0, 25.0))
        sigma_wide = float(np.clip(1.90 * sigma_seed, sigma_narrow + 5.0, 90.0))
        initial.update(
            {
                "sigma_narrow": sigma_narrow,
                "sigma_wide": sigma_wide,
                "signal_fraction": 0.75,
            }
        )

    return initial


def _parameter_limits(
    signal_model: str,
    background_model: str,
    fit_range: tuple[float, float],
):
    low, high = fit_range
    limits = {
        "signal_yield": (0.0, None),
        "background_yield": (0.0, None),
        "mean": (low, high),
        "slope": (-0.05, 0.05) if background_model == "exponential" else (-0.95, 0.95),
    }

    if signal_model == "gaussian":
        limits["sigma"] = (1.0, 0.5 * (high - low))
    else:
        limits.update(
            {
                "sigma_narrow": (1.0, 0.25 * (high - low)),
                "sigma_wide": (1.0, 0.75 * (high - low)),
                "signal_fraction": (0.0, 1.0),
            }
        )

    return limits


def expected_counts_from_parameters(
    bin_edges,
    parameters: Mapping[str, float],
    fit_range: tuple[float, float],
    signal_model: str,
    background_model: str = "exponential",
):
    """Return expected bin counts for fitted mass-model parameters."""
    signal_model = _validate_signal_model(signal_model)
    background_model = _validate_background_model(background_model)

    if signal_model == "gaussian":
        return signal_background_binned_model(
            bin_edges,
            parameters["signal_yield"],
            parameters["background_yield"],
            parameters["mean"],
            parameters["sigma"],
            parameters["slope"],
            fit_range=fit_range,
            signal_model=signal_model,
            background_model=background_model,
        )

    return signal_background_binned_model(
        bin_edges,
        parameters["signal_yield"],
        parameters["background_yield"],
        parameters["mean"],
        parameters["sigma_narrow"],
        parameters["slope"],
        fit_range=fit_range,
        signal_model=signal_model,
        background_model=background_model,
        sigma_wide=parameters["sigma_wide"],
        signal_fraction=parameters["signal_fraction"],
    )


def _make_nll(
    counts: np.ndarray,
    bin_edges: np.ndarray,
    fit_range: tuple[float, float],
    signal_model: str,
    background_model: str,
):
    if signal_model == "gaussian":

        def nll(signal_yield, background_yield, mean, sigma, slope):
            if signal_yield < 0 or background_yield < 0 or sigma <= 0:
                return np.inf
            parameters = {
                "signal_yield": signal_yield,
                "background_yield": background_yield,
                "mean": mean,
                "sigma": sigma,
                "slope": slope,
            }
            return _poisson_nll(
                counts,
                bin_edges,
                fit_range,
                signal_model,
                background_model,
                parameters,
            )

        return nll

    def nll(
        signal_yield,
        background_yield,
        mean,
        sigma_narrow,
        sigma_wide,
        signal_fraction,
        slope,
    ):
        if (
            signal_yield < 0
            or background_yield < 0
            or sigma_narrow <= 0
            or sigma_wide <= 0
            or not 0.0 <= signal_fraction <= 1.0
        ):
            return np.inf
        parameters = {
            "signal_yield": signal_yield,
            "background_yield": background_yield,
            "mean": mean,
            "sigma_narrow": sigma_narrow,
            "sigma_wide": sigma_wide,
            "signal_fraction": signal_fraction,
            "slope": slope,
        }
        return _poisson_nll(
            counts,
            bin_edges,
            fit_range,
            signal_model,
            background_model,
            parameters,
        )

    return nll


def _poisson_nll(
    counts: np.ndarray,
    bin_edges: np.ndarray,
    fit_range: tuple[float, float],
    signal_model: str,
    background_model: str,
    parameters: Mapping[str, float],
):
    try:
        expected = expected_counts_from_parameters(
            bin_edges,
            parameters,
            fit_range,
            signal_model,
            background_model=background_model,
        )
    except ValueError:
        return np.inf

    expected = np.clip(expected, 1.0e-12, None)
    return float(np.sum(expected - counts * np.log(expected)))


def _uncertainties_from_minuit(
    errors: Mapping[str, float],
    parameter_labels: Mapping[str, str],
) -> dict[str, float | None]:
    return {
        label: (None if errors[name] is None else float(errors[name]))
        for name, label in parameter_labels.items()
    }


def _fit_with_iminuit(nll, initial, limits, parameter_labels):
    from iminuit import Minuit

    minuit = Minuit(nll, **initial)
    minuit.errordef = Minuit.LIKELIHOOD
    for name, limit in limits.items():
        minuit.limits[name] = limit

    minuit.migrad()
    minuit.hesse()

    values = {name: float(minuit.values[name]) for name in parameter_labels}
    uncertainties = _uncertainties_from_minuit(minuit.errors, parameter_labels)
    fmin = minuit.fmin
    fit_status = {
        "backend": "iminuit",
        "success": bool(minuit.valid),
        "valid": bool(minuit.valid),
        "accurate_covariance": bool(getattr(fmin, "has_accurate_covar", False)),
        "positive_definite_covariance": bool(getattr(fmin, "has_posdef_covar", False)),
        "edm": float(getattr(fmin, "edm", np.nan)),
        "nfcn": int(getattr(fmin, "nfcn", 0)),
        "message": "iminuit migrad completed",
    }
    return values, uncertainties, fit_status, float(minuit.fval)


def _numerical_hessian(func, x: np.ndarray, steps: np.ndarray):
    size = len(x)
    hessian = np.zeros((size, size), dtype=float)
    f0 = func(x)

    for i in range(size):
        step_i = np.zeros(size)
        step_i[i] = steps[i]
        hessian[i, i] = (func(x + step_i) - 2.0 * f0 + func(x - step_i)) / steps[i] ** 2

        for j in range(i + 1, size):
            step_j = np.zeros(size)
            step_j[j] = steps[j]
            value = (
                func(x + step_i + step_j)
                - func(x + step_i - step_j)
                - func(x - step_i + step_j)
                + func(x - step_i - step_j)
            ) / (4.0 * steps[i] * steps[j])
            hessian[i, j] = value
            hessian[j, i] = value

    return hessian


def _finite_difference_steps(
    values: Mapping[str, float],
    names: list[str],
    background_model: str,
) -> np.ndarray:
    steps = []
    for name in names:
        if name in {"signal_yield", "background_yield"}:
            steps.append(max(1.0, 0.01 * abs(values[name])))
        elif name in {"mean", "sigma", "sigma_narrow", "sigma_wide"}:
            steps.append(0.10)
        elif name == "signal_fraction":
            steps.append(0.01)
        elif name == "slope":
            steps.append(1.0e-5 if background_model == "exponential" else 1.0e-3)
        else:
            steps.append(1.0e-5)
    return np.array(steps, dtype=float)


def _clip_to_limits(value: float, limit: tuple[float | None, float | None]) -> float:
    low, high = limit
    if low is not None:
        value = max(value, low)
    if high is not None:
        value = min(value, high)
    return value


def _scipy_starting_points(initial, limits, names):
    starts = [initial]

    if {"sigma_narrow", "sigma_wide", "signal_fraction"}.issubset(initial):
        total = initial["signal_yield"] + initial["background_yield"]
        variants = [
            {"signal_fraction": 0.05, "sigma_narrow": 3.0, "sigma_wide": 20.0, "signal_share": 0.85},
            {"signal_fraction": 0.20, "sigma_narrow": 6.0, "sigma_wide": 26.0, "signal_share": 0.80},
            {"signal_fraction": 0.50, "sigma_narrow": 8.0, "sigma_wide": 34.0, "signal_share": 0.75},
            {"signal_fraction": 0.80, "sigma_narrow": 9.0, "sigma_wide": 45.0, "signal_share": 0.65},
        ]
        for variant in variants:
            start = dict(initial)
            start["signal_yield"] = total * variant["signal_share"]
            start["background_yield"] = total - start["signal_yield"]
            start["sigma_narrow"] = variant["sigma_narrow"]
            start["sigma_wide"] = variant["sigma_wide"]
            start["signal_fraction"] = variant["signal_fraction"]
            starts.append(start)

    clipped_starts = []
    for start in starts:
        clipped_starts.append(
            {
                name: _clip_to_limits(float(start[name]), limits[name])
                for name in names
            }
        )
    return clipped_starts


def _fit_with_scipy(nll, initial, limits, parameter_labels, background_model: str):
    from scipy.optimize import minimize

    names = list(parameter_labels)
    bounds = [limits[name] for name in names]

    def objective(x):
        return nll(*x)

    results = []
    for start in _scipy_starting_points(initial, limits, names):
        x0 = np.array([start[name] for name in names], dtype=float)
        results.append(minimize(objective, x0, method="L-BFGS-B", bounds=bounds))

    successful = [candidate for candidate in results if candidate.success]
    result_pool = successful or results
    result = min(result_pool, key=lambda candidate: candidate.fun)
    values = {name: float(value) for name, value in zip(names, result.x)}

    uncertainties = {label: None for label in parameter_labels.values()}
    if result.success:
        steps = _finite_difference_steps(values, names, background_model)
        try:
            hessian = _numerical_hessian(objective, result.x, steps)
            covariance = np.linalg.pinv(hessian)
            errors = np.sqrt(np.clip(np.diag(covariance), a_min=0.0, a_max=None))
            uncertainties = {
                parameter_labels[name]: float(error)
                for name, error in zip(names, errors)
            }
        except (FloatingPointError, np.linalg.LinAlgError, ValueError):
            pass

    fit_status = {
        "backend": "scipy",
        "success": bool(result.success),
        "valid": bool(result.success),
        "accurate_covariance": False,
        "positive_definite_covariance": False,
        "edm": None,
        "nfcn": int(result.nfev),
        "message": str(result.message),
    }
    return values, uncertainties, fit_status, float(result.fun)


def _fit_quality(
    counts: np.ndarray,
    expected: np.ndarray,
    n_parameters: int,
) -> dict[str, float | int | None]:
    valid = counts > 0
    pulls = np.zeros_like(expected, dtype=float)
    pulls[valid] = (counts[valid] - expected[valid]) / np.sqrt(counts[valid])
    chi2 = float(np.sum(pulls[valid] ** 2))
    ndof = int(np.count_nonzero(valid) - n_parameters)
    chi2_ndof = chi2 / ndof if ndof > 0 else None
    return {
        "chi2": chi2,
        "ndof": ndof,
        "chi2_ndof": chi2_ndof,
        "n_fitted_bins": int(np.count_nonzero(valid)),
    }


def _result_parameters(fit_values: Mapping[str, float], signal_model: str):
    result = {
        "signal_yield": fit_values["signal_yield"],
        "background_yield": fit_values["background_yield"],
        "mass_mean": fit_values["mean"],
        "background_slope": fit_values["slope"],
    }

    if signal_model == "gaussian":
        result["mass_resolution"] = fit_values["sigma"]
        return result

    signal_fraction = fit_values["signal_fraction"]
    sigma_narrow = fit_values["sigma_narrow"]
    sigma_wide = fit_values["sigma_wide"]
    if sigma_narrow > sigma_wide:
        sigma_narrow, sigma_wide = sigma_wide, sigma_narrow
        signal_fraction = 1.0 - signal_fraction

    effective_sigma = np.sqrt(
        signal_fraction * sigma_narrow**2 + (1.0 - signal_fraction) * sigma_wide**2
    )
    result.update(
        {
            "mass_resolution": float(effective_sigma),
            "mass_resolution_narrow": sigma_narrow,
            "mass_resolution_wide": sigma_wide,
            "signal_fraction": signal_fraction,
        }
    )
    return result


def fit_mass_distribution(
    masses,
    fit_range: tuple[float, float] = (5200, 5600),
    bins: int = 80,
    signal_model: str = "gaussian",
    background_model: str = "exponential",
):
    """Fit selected B-candidate invariant masses with a nominal mass model.

    The fit is an extended binned Poisson likelihood using either a Gaussian or
    common-mean double-Gaussian signal PDF for the B peak, with a finite-range
    normalized exponential or linear PDF for the combinatorial background.  The
    single Gaussian is the default: on this sample the wide component of the
    double Gaussian is degenerate with the combinatorial background, which it
    absorbs, and the double-Gaussian model is therefore used as a systematic
    variation rather than as the nominal shape.
    """
    signal_model = _validate_signal_model(signal_model)
    background_model = _validate_background_model(background_model)
    if bins <= 0:
        raise ValueError("bins must be positive.")

    fit_range = tuple(map(float, fit_range))
    low, high = fit_range
    if not low < high:
        raise ValueError("fit_range must be an increasing (low, high) interval.")

    values = _clean_masses(masses, fit_range)
    if len(values) == 0:
        raise ValueError("No finite masses were found inside the requested fit range.")

    counts, bin_edges = np.histogram(values, bins=bins, range=fit_range)
    initial = _initial_values(values, counts, fit_range, signal_model, background_model)
    limits = _parameter_limits(signal_model, background_model, fit_range)
    parameter_labels = PARAMETER_LABELS_BY_MODEL[signal_model]
    nll = _make_nll(counts, bin_edges, fit_range, signal_model, background_model)

    try:
        fit_values, uncertainties, fit_status, fval = _fit_with_iminuit(
            nll,
            initial,
            limits,
            parameter_labels,
        )
    except ImportError:
        fit_values, uncertainties, fit_status, fval = _fit_with_scipy(
            nll,
            initial,
            limits,
            parameter_labels,
            background_model,
        )

    expected = expected_counts_from_parameters(
        bin_edges,
        fit_values,
        fit_range,
        signal_model,
        background_model=background_model,
    )
    fit_quality = _fit_quality(counts, expected, len(parameter_labels))
    result = _result_parameters(fit_values, signal_model)
    result.update(
        {
            "signal_model": signal_model,
            "background_model": background_model,
            "parameter_uncertainties": uncertainties,
            "fit_quality": fit_quality,
            "fit_status": {
                **fit_status,
                "negative_log_likelihood": fval,
                "n_candidates": int(counts.sum()),
                "n_fitted_candidates": int(counts.sum()),
                "fit_range": [low, high],
                "bins": int(bins),
                "n_parameters": len(parameter_labels),
            },
        }
    )
    return result
