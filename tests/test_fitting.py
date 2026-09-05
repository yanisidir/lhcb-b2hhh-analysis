"""Checks of the mass models, the fit, and the asymmetry propagation.

The fit is tested by closure: pseudo-data are generated from known parameters
and the fit must recover them. That is the same logic as the toy-MC study under
``studies/toy_mc``, reduced to a single fast case.
"""

import numpy as np
import pytest

from src.fitting.acceptance import fit_quality_acceptance
from src.fitting.asymmetry import raw_asymmetry
from src.fitting.fit_mass import fit_mass_distribution
from src.fitting.mass_models import (
    double_gaussian_pdf,
    exponential_background_pdf,
    gaussian_pdf,
)


FIT_RANGE = (5200.0, 5600.0)


def _integral(pdf_values, grid):
    return np.trapezoid(pdf_values, grid) if hasattr(np, "trapezoid") else np.trapz(pdf_values, grid)


def test_gaussian_pdf_integrates_to_one():
    grid = np.linspace(5100.0, 5460.0, 20001)
    assert _integral(gaussian_pdf(grid, 5280.0, 20.0), grid) == pytest.approx(1.0, rel=1e-6)


def test_double_gaussian_reduces_to_a_single_gaussian():
    grid = np.linspace(5100.0, 5460.0, 2001)
    only_narrow = double_gaussian_pdf(grid, 5280.0, 15.0, 45.0, signal_fraction=1.0)
    assert only_narrow == pytest.approx(gaussian_pdf(grid, 5280.0, 15.0))


def test_exponential_background_is_normalised_inside_the_fit_range():
    grid = np.linspace(*FIT_RANGE, 20001)
    values = exponential_background_pdf(grid, -0.002, FIT_RANGE)
    assert _integral(values, grid) == pytest.approx(1.0, rel=1e-6)
    assert exponential_background_pdf(np.array([FIT_RANGE[0] - 1.0]), -0.002, FIT_RANGE) == 0.0


def _generate_pseudo_data(n_signal, n_background, mean, sigma, slope, seed):
    """Draw a signal peak plus an exponential background inside the fit range."""
    rng = np.random.default_rng(seed)
    low, high = FIT_RANGE

    signal = np.array([])
    while len(signal) < n_signal:
        draw = rng.normal(mean, sigma, size=2 * n_signal)
        signal = np.concatenate([signal, draw[(draw >= low) & (draw <= high)]])
    signal = signal[:n_signal]

    # Inverse-transform sampling of exp(slope * (m - low)) truncated to the range.
    uniform = rng.random(n_background)
    scale = np.expm1(slope * (high - low))
    background = low + np.log1p(uniform * scale) / slope

    return np.concatenate([signal, background])


def test_fit_recovers_the_generated_parameters():
    n_signal, n_background = 4000, 1000
    mean, sigma = 5280.0, 20.0
    masses = _generate_pseudo_data(n_signal, n_background, mean, sigma, slope=-0.004, seed=20240501)

    result = fit_mass_distribution(masses, fit_range=FIT_RANGE, bins=80, signal_model="gaussian")

    uncertainties = result["parameter_uncertainties"]
    assert result["fit_status"]["valid"]
    # Each parameter must sit within three of its own standard deviations of truth.
    assert abs(result["mass_mean"] - mean) < 3.0 * uncertainties["mass_mean"]
    assert abs(result["mass_resolution"] - sigma) < 3.0 * uncertainties["mass_resolution"]
    assert abs(result["signal_yield"] - n_signal) < 3.0 * uncertainties["signal_yield"]
    assert abs(result["background_yield"] - n_background) < 3.0 * uncertainties["background_yield"]


def _fit_result(signal_yield, uncertainty):
    return {
        "signal_yield": signal_yield,
        "parameter_uncertainties": {"signal_yield": uncertainty},
        "signal_model": "gaussian",
        "background_model": "exponential",
        "fit_status": {"fit_range": list(FIT_RANGE), "n_fitted_candidates": int(signal_yield)},
    }


def test_equal_yields_give_a_vanishing_asymmetry():
    result = raw_asymmetry(_fit_result(500.0, 25.0), _fit_result(500.0, 25.0), 500, 500)
    assert result["A_raw"] == pytest.approx(0.0)


def test_asymmetry_sign_follows_the_negative_charge_excess():
    more_minus = raw_asymmetry(_fit_result(450.0, 21.0), _fit_result(550.0, 23.0), 450, 550)
    assert more_minus["A_raw"] == pytest.approx(0.1)

    more_plus = raw_asymmetry(_fit_result(550.0, 23.0), _fit_result(450.0, 21.0), 550, 450)
    assert more_plus["A_raw"] == pytest.approx(-0.1)


def test_asymmetry_uncertainty_matches_the_propagation_formula():
    n_plus, n_minus, sigma_plus, sigma_minus = 689.0, 642.0, 28.6, 26.2
    total = n_plus + n_minus
    expected = np.sqrt(
        (2.0 * n_minus / total**2 * sigma_plus) ** 2 + (2.0 * n_plus / total**2 * sigma_minus) ** 2
    )

    result = raw_asymmetry(
        _fit_result(n_plus, sigma_plus), _fit_result(n_minus, sigma_minus), int(n_plus), int(n_minus)
    )

    assert result["sigma_A_raw"] == pytest.approx(expected)


def test_asymmetry_uncertainty_is_none_when_a_fit_gives_no_error():
    result = raw_asymmetry(_fit_result(500.0, None), _fit_result(500.0, 25.0), 500, 500)
    assert result["sigma_A_raw"] is None


def _acceptance_row(**overrides):
    row = {
        "fit_success_plus": True,
        "fit_valid_plus": True,
        "fit_success_minus": True,
        "fit_valid_minus": True,
        "chi2_ndof_plus": 1.2,
        "chi2_ndof_minus": 1.1,
        "N_plus": 690.0,
        "N_minus": 640.0,
        "sigma_N_plus": 28.0,
        "sigma_N_minus": 26.0,
        "variation_role": "candidate_systematic",
    }
    row.update(overrides)
    return row


def test_a_healthy_variation_is_accepted():
    accepted, reason = fit_quality_acceptance(_acceptance_row())
    assert accepted and reason == ""


@pytest.mark.parametrize(
    "overrides, expected",
    [
        ({"fit_valid_minus": False}, "B- fit did not converge"),
        ({"chi2_ndof_plus": 3.0}, "B+ chi2/ndof >= 2.5"),
        ({"sigma_N_plus": 0.0}, "B+ fitted yield uncertainty is not positive"),
        ({"variation_role": "stress_test"}, "stress-test model"),
    ],
)
def test_unusable_variations_are_rejected_with_a_reason(overrides, expected):
    accepted, reason = fit_quality_acceptance(_acceptance_row(**overrides))
    assert not accepted
    assert expected in reason
