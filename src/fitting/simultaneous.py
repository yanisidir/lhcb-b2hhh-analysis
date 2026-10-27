"""Simultaneous B+/B- invariant-mass fit, and the combination of magnet polarities.

The B+ and B- spectra are fitted together with one set of shape parameters --
peak position, resolution, background slope -- and separate yields for each
charge. Fitting them separately instead lets each charge pick its own
background shape, so a fluctuation in one spectrum is absorbed by its
background and moves its signal yield: on MagnetDown the separate fits give the
B+ background a much steeper slope than the B- one, which is what made the raw
asymmetry depend on the fit range. A combinatorial background has no reason to
look different for the two charges, and neither has the mass resolution.

The raw asymmetry is a parameter of the fit, so its uncertainty includes the
correlation with the total yield directly.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from src.fitting.fit_mass import (
    _clean_masses,
    _fit_quality,
    _fit_with_iminuit,
    _signal_shape_parameters,
    _validate_background_model,
    _validate_signal_model,
)
from src.fitting.mass_models import signal_background_binned_model


SHAPE_LABELS_BY_MODEL = {
    "gaussian": {"sigma": "mass_resolution"},
    "double_gaussian": {
        "sigma_narrow": "mass_resolution_narrow",
        "sigma_wide": "mass_resolution_wide",
        "signal_fraction": "signal_fraction",
    },
}

# (signal_fraction, sigma_narrow, sigma_wide) starting points for the
# double-Gaussian shape, whose wide component trades off against the background.
DOUBLE_GAUSSIAN_SHAPE_STARTS = [
    (0.75, 13.0, 37.0),
    (0.50, 12.0, 30.0),
    (0.90, 18.0, 55.0),
    (0.70, 17.0, 40.0),
    (0.95, 19.0, 70.0),
]


def _parameter_labels(signal_model: str) -> dict[str, str]:
    return {
        "signal_yield": "signal_yield",
        "asymmetry": "A_raw",
        "mean": "mass_mean",
        **SHAPE_LABELS_BY_MODEL[signal_model],
        "slope": "background_slope",
        "background_yield_plus": "background_yield_plus",
        "background_yield_minus": "background_yield_minus",
    }


def _expected_counts(bin_edges, values: Mapping[str, float], fit_range, signal_model, background_model):
    """Return the expected B+ and B- bin counts for a set of fit parameters."""
    n_plus = values["signal_yield"] * (1.0 - values["asymmetry"]) / 2.0
    n_minus = values["signal_yield"] * (1.0 + values["asymmetry"]) / 2.0
    if signal_model == "gaussian":
        shape = {"sigma": values["sigma"]}
    else:
        shape = {
            "sigma": values["sigma_narrow"],
            "sigma_wide": values["sigma_wide"],
            "signal_fraction": values["signal_fraction"],
        }

    def spectrum(signal_yield, background_yield):
        return signal_background_binned_model(
            bin_edges,
            signal_yield,
            background_yield,
            values["mean"],
            shape["sigma"],
            values["slope"],
            fit_range=fit_range,
            signal_model=signal_model,
            background_model=background_model,
            sigma_wide=shape.get("sigma_wide"),
            signal_fraction=shape.get("signal_fraction", 1.0),
        )

    return (
        spectrum(n_plus, values["background_yield_plus"]),
        spectrum(n_minus, values["background_yield_minus"]),
    )


def _make_nll(plus_counts, minus_counts, bin_edges, fit_range, signal_model, background_model):
    def poisson(values):
        try:
            expected_plus, expected_minus = _expected_counts(
                bin_edges, values, fit_range, signal_model, background_model
            )
        except ValueError:
            return np.inf
        expected_plus = np.clip(expected_plus, 1.0e-12, None)
        expected_minus = np.clip(expected_minus, 1.0e-12, None)
        return float(
            np.sum(expected_plus - plus_counts * np.log(expected_plus))
            + np.sum(expected_minus - minus_counts * np.log(expected_minus))
        )

    # MINUIT reads the parameter names from the signature, so each model gets
    # an explicit one.
    if signal_model == "gaussian":

        def nll(signal_yield, asymmetry, mean, sigma, slope, background_yield_plus, background_yield_minus):
            return poisson(locals())

        return nll

    def nll(
        signal_yield,
        asymmetry,
        mean,
        sigma_narrow,
        sigma_wide,
        signal_fraction,
        slope,
        background_yield_plus,
        background_yield_minus,
    ):
        return poisson(locals())

    return nll


def _limits(signal_model: str, background_model: str, fit_range):
    low, high = fit_range
    limits = {
        "signal_yield": (0.0, None),
        "asymmetry": (-1.0, 1.0),
        "mean": (low, high),
        "slope": (-0.05, 0.05) if background_model == "exponential" else (-0.95, 0.95),
        "background_yield_plus": (0.0, None),
        "background_yield_minus": (0.0, None),
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


def _starts(plus_values, minus_values, signal_model, background_model, limits):
    total = len(plus_values) + len(minus_values)
    base = {
        "signal_yield": 0.9 * total,
        "asymmetry": 0.0,
        "mean": 5284.0,
        "slope": -0.004 if background_model == "exponential" else 0.0,
        "background_yield_plus": max(0.05 * len(plus_values), 1.0),
        "background_yield_minus": max(0.05 * len(minus_values), 1.0),
    }
    if signal_model == "gaussian":
        starts = [{**base, "sigma": 19.5}]
    else:
        starts = [
            {**base, "signal_fraction": fraction, "sigma_narrow": narrow, "sigma_wide": wide}
            for fraction, narrow, wide in DOUBLE_GAUSSIAN_SHAPE_STARTS
        ]

    def clip(name, value):
        low, high = limits[name]
        value = value if low is None else max(value, low)
        return value if high is None else min(value, high)

    return [{name: clip(name, float(value)) for name, value in start.items()} for start in starts]


def _charge_yield(signal_yield, asymmetry, covariance, sign: float):
    """N(B+) = N (1 - A) / 2 or N(B-) = N (1 + A) / 2, with its uncertainty."""
    value = signal_yield * (1.0 - sign * asymmetry) / 2.0
    if covariance is None:
        return value, None
    d_n = (1.0 - sign * asymmetry) / 2.0
    d_a = -sign * signal_yield / 2.0
    variance = (
        d_n**2 * covariance["signal_yield"]["signal_yield"]
        + d_a**2 * covariance["asymmetry"]["asymmetry"]
        + 2.0 * d_n * d_a * covariance["signal_yield"]["asymmetry"]
    )
    return value, float(np.sqrt(variance)) if variance > 0 else None


def fit_charge_simultaneous(
    plus_masses,
    minus_masses,
    fit_range: tuple[float, float] = (5200.0, 5600.0),
    bins: int = 80,
    signal_model: str = "gaussian",
    background_model: str = "exponential",
):
    """Fit the B+ and B- mass spectra together with shared shape parameters.

    The model for each charge is the same signal and background shape as in
    ``fit_mass_distribution``; the signal yields are parameterised by the total
    ``N`` and the raw asymmetry ``A = (N- - N+) / (N- + N+)``, and each charge
    has its own background yield.
    """
    signal_model = _validate_signal_model(signal_model)
    background_model = _validate_background_model(background_model)
    if bins <= 0:
        raise ValueError("bins must be positive.")
    fit_range = tuple(map(float, fit_range))
    low, high = fit_range
    if not low < high:
        raise ValueError("fit_range must be an increasing (low, high) interval.")

    plus_values = _clean_masses(plus_masses, fit_range)
    minus_values = _clean_masses(minus_masses, fit_range)
    if len(plus_values) == 0 or len(minus_values) == 0:
        raise ValueError("Both charges need candidates inside the fit range.")

    plus_counts, bin_edges = np.histogram(plus_values, bins=bins, range=fit_range)
    minus_counts, _ = np.histogram(minus_values, bins=bins, range=fit_range)
    labels = _parameter_labels(signal_model)
    limits = _limits(signal_model, background_model, fit_range)
    nll = _make_nll(plus_counts, minus_counts, bin_edges, fit_range, signal_model, background_model)

    values, uncertainties, fit_status, fval, covariance = _fit_with_iminuit(
        nll,
        _starts(plus_values, minus_values, signal_model, background_model, limits),
        limits,
        labels,
    )

    expected_plus, expected_minus = _expected_counts(
        bin_edges, values, fit_range, signal_model, background_model
    )
    n_parameters = len(labels)
    # Each spectrum depends on the shared shape, the total yield, the asymmetry
    # and its own background yield: every parameter but the other background.
    per_spectrum_parameters = n_parameters - 1
    quality_plus = _fit_quality(plus_counts, expected_plus, per_spectrum_parameters)
    quality_minus = _fit_quality(minus_counts, expected_minus, per_spectrum_parameters)
    combined_lr = quality_plus["chi2_likelihood_ratio"] + quality_minus["chi2_likelihood_ratio"]

    n_plus, sigma_n_plus = _charge_yield(values["signal_yield"], values["asymmetry"], covariance, +1.0)
    n_minus, sigma_n_minus = _charge_yield(values["signal_yield"], values["asymmetry"], covariance, -1.0)

    shape, uncertainties = _signal_shape_parameters(values, uncertainties, signal_model)
    result = {
        "signal_model": signal_model,
        "background_model": background_model,
        "signal_yield": values["signal_yield"],
        "A_raw": values["asymmetry"],
        "sigma_A_raw": uncertainties["A_raw"],
        "N_plus": n_plus,
        "sigma_N_plus": sigma_n_plus,
        "N_minus": n_minus,
        "sigma_N_minus": sigma_n_minus,
        "mass_mean": values["mean"],
        **shape,
        "background_slope": values["slope"],
        "background_yield_plus": values["background_yield_plus"],
        "background_yield_minus": values["background_yield_minus"],
        "parameter_uncertainties": uncertainties,
        "fit_quality": {
            "plus": quality_plus,
            "minus": quality_minus,
            "chi2_likelihood_ratio": combined_lr,
            "ndof_likelihood_ratio": int(2 * bins - n_parameters),
        },
        "fit_status": {
            **fit_status,
            "negative_log_likelihood": fval,
            "n_fitted_candidates_plus": int(plus_counts.sum()),
            "n_fitted_candidates_minus": int(minus_counts.sum()),
            "fit_range": [low, high],
            "bins": int(bins),
            "n_parameters": n_parameters,
        },
    }
    return result


def combine_polarities(results_by_polarity: Mapping[str, Mapping]) -> dict:
    """Average the raw asymmetry over magnet polarities with equal weights.

    Reversing the magnet swaps which side of the detector each charge bends
    towards, so a left-right detection asymmetry enters the two polarities with
    opposite signs. The unweighted average cancels it to first order; a
    precision-weighted average would not, whenever the polarities have
    different sample sizes. The polarities are independent samples, so the
    uncertainty is ``sqrt(sum sigma_i^2) / n``.
    """
    if not results_by_polarity:
        raise ValueError("At least one polarity is needed.")
    asymmetries = [result["A_raw"] for result in results_by_polarity.values()]
    sigmas = [result["sigma_A_raw"] for result in results_by_polarity.values()]
    n = len(asymmetries)
    if any(value is None for value in asymmetries):
        return {"A_raw": None, "sigma_A_raw": None, "n_polarities": n}
    sigma = None if any(value is None for value in sigmas) else float(np.sqrt(np.sum(np.square(sigmas))) / n)
    return {
        "A_raw": float(np.mean(asymmetries)),
        "sigma_A_raw": sigma,
        "n_polarities": n,
        "by_polarity": {
            name: {"A_raw": result["A_raw"], "sigma_A_raw": result["sigma_A_raw"]}
            for name, result in results_by_polarity.items()
        },
    }


def _failed_result(message: str, fit_range, bins: int, signal_model: str, background_model: str):
    return {
        "signal_model": signal_model,
        "background_model": background_model,
        "A_raw": None,
        "sigma_A_raw": None,
        "N_plus": None,
        "sigma_N_plus": None,
        "N_minus": None,
        "sigma_N_minus": None,
        "fit_quality": {"plus": {"chi2_ndof": None}, "minus": {"chi2_ndof": None}},
        "fit_status": {
            "success": False,
            "valid": False,
            "message": message,
            "fit_range": list(fit_range),
            "bins": int(bins),
        },
    }


def fit_polarities(
    samples_by_polarity: Mapping[str, tuple],
    fit_range: tuple[float, float] = (5200.0, 5600.0),
    bins: int = 80,
    signal_model: str = "gaussian",
    background_model: str = "exponential",
):
    """Fit every polarity simultaneously in charge and average the asymmetry.

    ``samples_by_polarity`` maps a polarity name to its ``(B+ masses, B- masses)``.
    A fit that raises is recorded as failed rather than stopping a scan of
    configurations. Returns ``(results_by_polarity, combination)``.
    """
    results = {}
    for polarity, (plus_masses, minus_masses) in samples_by_polarity.items():
        try:
            results[polarity] = fit_charge_simultaneous(
                plus_masses,
                minus_masses,
                fit_range=fit_range,
                bins=bins,
                signal_model=signal_model,
                background_model=background_model,
            )
        except Exception as exc:  # a failed configuration is a result, not a crash
            results[polarity] = _failed_result(
                f"{type(exc).__name__}: {exc}", fit_range, bins, signal_model, background_model
            )
    return results, combine_polarities(results)


def charge_view(result: Mapping, charge: str) -> dict:
    """Return one charge of a simultaneous fit in the single-spectrum format.

    The plotting and reporting helpers written for ``fit_mass_distribution``
    read ``signal_yield``, ``background_yield`` and ``fit_quality`` from a
    single-spectrum result; this builds that view for ``"plus"`` or ``"minus"``.
    """
    if charge not in ("plus", "minus"):
        raise ValueError("charge must be 'plus' or 'minus'.")
    view = {
        key: value
        for key, value in result.items()
        if key not in ("fit_quality", "parameter_uncertainties", "signal_yield")
    }
    view.update(
        {
            "signal_yield": result[f"N_{charge}"],
            "background_yield": result[f"background_yield_{charge}"],
            "parameter_uncertainties": {
                **result.get("parameter_uncertainties", {}),
                "signal_yield": result[f"sigma_N_{charge}"],
            },
            "fit_quality": result["fit_quality"][charge],
        }
    )
    return view
