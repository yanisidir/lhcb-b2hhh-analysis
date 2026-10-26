"""Mass-shape models for first nominal B-candidate invariant-mass fits.

The functions in this module describe compact HEP mass models: Gaussian and
double-Gaussian signal peaks for the reconstructed B meson, plus finite-range
background shapes for the combinatorial component.  The binned model
returns expected candidate counts per bin and is intended for the first nominal
fit, not for a full systematic signal-yield study.
"""

from __future__ import annotations

import numpy as np
from scipy.special import erf


SUPPORTED_SIGNAL_MODELS = {"gaussian", "double_gaussian"}
SUPPORTED_BACKGROUND_MODELS = {"exponential", "linear"}


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


def _validate_fit_range(fit_range: tuple[float, float]) -> tuple[float, float]:
    low, high = map(float, fit_range)
    if not low < high:
        raise ValueError("fit_range must be an increasing (low, high) interval.")
    return low, high


def _gaussian_cdf(mass, mean: float, sigma: float):
    if sigma <= 0:
        raise ValueError("Gaussian sigma must be positive.")
    z = (np.asarray(mass, dtype=float) - mean) / (np.sqrt(2.0) * sigma)
    return 0.5 * (1.0 + erf(z))


def gaussian_pdf(mass, mean: float, sigma: float):
    """Return the unit-normalized Gaussian signal PDF in MeV^-1.

    Parameters are the reconstructed B-meson mass peak position ``mean`` and
    detector mass resolution ``sigma``, both in MeV.
    """
    if sigma <= 0:
        raise ValueError("Gaussian sigma must be positive.")

    mass = np.asarray(mass, dtype=float)
    norm = sigma * np.sqrt(2.0 * np.pi)
    return np.exp(-0.5 * ((mass - mean) / sigma) ** 2) / norm


def double_gaussian_pdf(
    mass,
    mean: float,
    sigma_narrow: float,
    sigma_wide: float,
    signal_fraction: float,
):
    """Return a unit-normalized double-Gaussian signal PDF in MeV^-1.

    The two Gaussian components share a common B-meson mass mean.  The
    ``signal_fraction`` parameter gives the narrow-core fraction, with the
    wide-tail component receiving ``1 - signal_fraction``.
    """
    if not 0.0 <= signal_fraction <= 1.0:
        raise ValueError("signal_fraction must lie between 0 and 1.")

    return (
        signal_fraction * gaussian_pdf(mass, mean, sigma_narrow)
        + (1.0 - signal_fraction) * gaussian_pdf(mass, mean, sigma_wide)
    )


def gaussian_bin_probabilities(
    bin_edges,
    mean: float,
    sigma: float,
    fit_range: tuple[float, float] | None = None,
):
    """Integrate the Gaussian signal PDF over mass bins.

    If ``fit_range`` is provided, the bin probabilities are normalized inside
    that fitted mass interval, so the fitted signal yield corresponds to the
    number of signal candidates in the fit range.
    """
    bin_edges = np.asarray(bin_edges, dtype=float)
    integration_edges = bin_edges
    probabilities = np.diff(_gaussian_cdf(integration_edges, mean, sigma))

    if fit_range is not None:
        low, high = _validate_fit_range(fit_range)
        integration_edges = np.clip(bin_edges, low, high)
        probabilities = np.diff(_gaussian_cdf(integration_edges, mean, sigma))
        norm = _gaussian_cdf(high, mean, sigma) - _gaussian_cdf(low, mean, sigma)
        if norm <= 0:
            raise ValueError("Gaussian normalization is not positive in fit_range.")
        probabilities = probabilities / norm

    return probabilities


def double_gaussian_bin_probabilities(
    bin_edges,
    mean: float,
    sigma_narrow: float,
    sigma_wide: float,
    signal_fraction: float,
    fit_range: tuple[float, float] | None = None,
):
    """Integrate a common-mean double-Gaussian signal PDF over mass bins."""
    if not 0.0 <= signal_fraction <= 1.0:
        raise ValueError("signal_fraction must lie between 0 and 1.")

    narrow = gaussian_bin_probabilities(bin_edges, mean, sigma_narrow, fit_range)
    wide = gaussian_bin_probabilities(bin_edges, mean, sigma_wide, fit_range)
    return signal_fraction * narrow + (1.0 - signal_fraction) * wide


def exponential_background_pdf(mass, slope: float, fit_range: tuple[float, float]):
    """Return a finite-range normalized exponential background PDF in MeV^-1.

    The background shape is proportional to ``exp(slope * (m - low))`` inside
    ``fit_range`` and zero outside it.  The slope is expressed in MeV^-1.
    """
    low, high = _validate_fit_range(fit_range)
    mass = np.asarray(mass, dtype=float)
    width = high - low
    pdf = np.zeros_like(mass, dtype=float)
    in_range = (mass >= low) & (mass <= high)

    if abs(slope) < 1.0e-12:
        pdf[in_range] = 1.0 / width
        return pdf

    norm = np.expm1(slope * width) / slope
    pdf[in_range] = np.exp(slope * (mass[in_range] - low)) / norm
    return pdf


def exponential_background_cdf(mass, slope: float, fit_range: tuple[float, float]):
    """Return the normalized exponential background CDF over ``fit_range``."""
    low, high = _validate_fit_range(fit_range)
    mass = np.asarray(mass, dtype=float)
    clipped = np.clip(mass, low, high)
    width = high - low

    if abs(slope) < 1.0e-12:
        return (clipped - low) / width

    return np.expm1(slope * (clipped - low)) / np.expm1(slope * width)


def exponential_background_bin_probabilities(
    bin_edges,
    slope: float,
    fit_range: tuple[float, float] | None = None,
):
    """Integrate the finite-range exponential background PDF over mass bins."""
    bin_edges = np.asarray(bin_edges, dtype=float)
    fit_range = fit_range or (float(bin_edges[0]), float(bin_edges[-1]))
    probabilities = np.diff(exponential_background_cdf(bin_edges, slope, fit_range))
    return np.clip(probabilities, a_min=0.0, a_max=None)


def linear_background_pdf(mass, slope: float, fit_range: tuple[float, float]):
    """Return a finite-range normalized linear background PDF in MeV^-1.

    The shape is proportional to ``1 + slope * x``, where ``x`` maps the fitted
    mass interval onto ``[-1, 1]``.  The dimensionless ``slope`` parameter is
    constrained by the fitter so the combinatorial-background density remains
    positive across the finite mass range.
    """
    if abs(slope) > 1.0:
        raise ValueError("Linear background slope must satisfy |slope| <= 1.")

    low, high = _validate_fit_range(fit_range)
    mass = np.asarray(mass, dtype=float)
    width = high - low
    pdf = np.zeros_like(mass, dtype=float)
    in_range = (mass >= low) & (mass <= high)
    x = 2.0 * (mass[in_range] - low) / width - 1.0
    pdf[in_range] = (1.0 + slope * x) / width
    return np.clip(pdf, a_min=0.0, a_max=None)


def linear_background_cdf(mass, slope: float, fit_range: tuple[float, float]):
    """Return the normalized linear-background CDF over ``fit_range``."""
    if abs(slope) > 1.0:
        raise ValueError("Linear background slope must satisfy |slope| <= 1.")

    low, high = _validate_fit_range(fit_range)
    mass = np.asarray(mass, dtype=float)
    width = high - low
    t = (np.clip(mass, low, high) - low) / width
    return t + slope * (t**2 - t)


def linear_background_bin_probabilities(
    bin_edges,
    slope: float,
    fit_range: tuple[float, float] | None = None,
):
    """Integrate the finite-range linear background PDF over mass bins."""
    bin_edges = np.asarray(bin_edges, dtype=float)
    fit_range = fit_range or (float(bin_edges[0]), float(bin_edges[-1]))
    probabilities = np.diff(linear_background_cdf(bin_edges, slope, fit_range))
    return np.clip(probabilities, a_min=0.0, a_max=None)


def background_bin_probabilities(
    bin_edges,
    slope: float,
    fit_range: tuple[float, float],
    background_model: str = "exponential",
):
    """Integrate the requested combinatorial-background PDF over mass bins."""
    background_model = _validate_background_model(background_model)
    if background_model == "exponential":
        return exponential_background_bin_probabilities(bin_edges, slope, fit_range)
    return linear_background_bin_probabilities(bin_edges, slope, fit_range)


def background_pdf(
    mass,
    slope: float,
    fit_range: tuple[float, float],
    background_model: str = "exponential",
):
    """Return the requested normalized combinatorial-background PDF."""
    background_model = _validate_background_model(background_model)
    if background_model == "exponential":
        return exponential_background_pdf(mass, slope, fit_range)
    return linear_background_pdf(mass, slope, fit_range)


def signal_background_binned_model(
    bin_edges,
    signal_yield: float,
    background_yield: float,
    mean: float,
    sigma: float,
    slope: float,
    fit_range: tuple[float, float] | None = None,
    signal_model: str = "gaussian",
    background_model: str = "exponential",
    sigma_wide: float | None = None,
    signal_fraction: float = 1.0,
):
    """Return expected binned counts for a signal plus background mass model.

    The returned array has one entry per mass bin.  ``signal_yield`` and
    ``background_yield`` are extended yields in the fitted mass interval.
    ``signal_model`` can be ``"gaussian"`` or ``"double_gaussian"``.  For the
    double-Gaussian model, ``sigma`` is the narrow width and ``sigma_wide`` is
    the broader tail width.  ``background_model`` can be ``"exponential"`` or
    ``"linear"``, both normalized over the finite fit range.
    """
    bin_edges = np.asarray(bin_edges, dtype=float)
    fit_range = fit_range or (float(bin_edges[0]), float(bin_edges[-1]))
    signal_model = _validate_signal_model(signal_model)
    background_model = _validate_background_model(background_model)

    if signal_model == "gaussian":
        signal_prob = gaussian_bin_probabilities(bin_edges, mean, sigma, fit_range)
    else:
        if sigma_wide is None:
            raise ValueError("sigma_wide is required for the double-Gaussian signal model.")
        signal_prob = double_gaussian_bin_probabilities(
            bin_edges,
            mean,
            sigma,
            sigma_wide,
            signal_fraction,
            fit_range,
        )

    background_prob = background_bin_probabilities(
        bin_edges,
        slope,
        fit_range,
        background_model=background_model,
    )
    return signal_yield * signal_prob + background_yield * background_prob


def signal_background_density(
    mass,
    signal_yield: float,
    background_yield: float,
    mean: float,
    sigma: float,
    slope: float,
    fit_range: tuple[float, float],
    signal_model: str = "gaussian",
    background_model: str = "exponential",
    sigma_wide: float | None = None,
    signal_fraction: float = 1.0,
):
    """Return the continuous fitted density in candidates per MeV."""
    low, high = _validate_fit_range(fit_range)
    signal_model = _validate_signal_model(signal_model)
    background_model = _validate_background_model(background_model)
    mass = np.asarray(mass, dtype=float)
    density = np.zeros_like(mass, dtype=float)
    in_range = (mass >= low) & (mass <= high)

    if signal_model == "gaussian":
        signal_norm = _gaussian_cdf(high, mean, sigma) - _gaussian_cdf(low, mean, sigma)
        signal_density = gaussian_pdf(mass[in_range], mean, sigma) / signal_norm
    else:
        if sigma_wide is None:
            raise ValueError("sigma_wide is required for the double-Gaussian signal model.")
        if not 0.0 <= signal_fraction <= 1.0:
            raise ValueError("signal_fraction must lie between 0 and 1.")
        # Each component is normalized inside the fit range on its own, exactly
        # as ``double_gaussian_bin_probabilities`` does, so that this density is
        # the same shape the binned fit uses and ``signal_fraction`` keeps its
        # meaning: the narrow-core fraction *within* the fitted interval.
        narrow_norm = _gaussian_cdf(high, mean, sigma) - _gaussian_cdf(low, mean, sigma)
        wide_norm = _gaussian_cdf(high, mean, sigma_wide) - _gaussian_cdf(low, mean, sigma_wide)
        if narrow_norm <= 0 or wide_norm <= 0:
            raise ValueError("Double-Gaussian normalization is not positive in fit_range.")
        signal_density = (
            signal_fraction * gaussian_pdf(mass[in_range], mean, sigma) / narrow_norm
            + (1.0 - signal_fraction)
            * gaussian_pdf(mass[in_range], mean, sigma_wide) / wide_norm
        )

    background_density = background_pdf(
        mass[in_range],
        slope,
        fit_range,
        background_model=background_model,
    )
    density[in_range] = signal_yield * signal_density + background_yield * background_density
    return density
