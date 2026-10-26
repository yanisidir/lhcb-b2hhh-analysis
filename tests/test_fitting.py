"""Checks of the mass models, the fit, and the asymmetry propagation.

The fit is tested by closure: pseudo-data are generated from known parameters
and the fit must recover them. That is the same logic as the toy-MC study under
``studies/toy_mc``, reduced to a single fast case.
"""

import numpy as np
import pytest

from src.fitting.acceptance import barlow_shift, fit_quality_acceptance
from src.fitting.asymmetry import raw_asymmetry
from src.fitting.fit_mass import _fit_quality, _result_parameters, fit_mass_distribution
from src.fitting.mass_models import (
    double_gaussian_pdf,
    exponential_background_pdf,
    gaussian_pdf,
    signal_background_binned_model,
    signal_background_density,
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


DOUBLE_GAUSSIAN = {
    "signal_model": "double_gaussian",
    "sigma_wide": 100.0,
    "signal_fraction": 0.7,
}


def test_double_gaussian_density_integrates_to_the_binned_model():
    """The plotted density must be the PDF the binned fit actually uses.

    Both are normalised, so a mismatch does not show up in either one alone:
    the two have to be compared bin by bin.
    """
    edges = np.linspace(*FIT_RANGE, 21)
    binned = signal_background_binned_model(
        edges,
        1000.0,
        0.0,
        5285.0,
        19.5,
        0.0,
        fit_range=FIT_RANGE,
        **DOUBLE_GAUSSIAN,
    )
    assert binned.sum() == pytest.approx(1000.0, rel=1e-9)

    for index in range(len(edges) - 1):
        grid = np.linspace(edges[index], edges[index + 1], 801)
        density = signal_background_density(
            grid,
            1000.0,
            0.0,
            5285.0,
            19.5,
            0.0,
            FIT_RANGE,
            **DOUBLE_GAUSSIAN,
        )
        assert _integral(density, grid) == pytest.approx(binned[index], rel=1e-5)


def _double_gaussian_fit_values(sigma_narrow: float, sigma_wide: float, signal_fraction: float):
    return {
        "signal_yield": 1000.0,
        "background_yield": 200.0,
        "mean": 5285.0,
        "sigma_narrow": sigma_narrow,
        "sigma_wide": sigma_wide,
        "signal_fraction": signal_fraction,
        "slope": -0.002,
    }


def _double_gaussian_uncertainties():
    return {
        "signal_yield": 40.0,
        "background_yield": 20.0,
        "mass_mean": 0.8,
        "mass_resolution_narrow": 5.0,
        "mass_resolution_wide": 0.5,
        "signal_fraction": 0.05,
        "background_slope": 1.0e-4,
    }


def test_relabelled_double_gaussian_components_keep_their_uncertainties():
    """When the fitted widths come back in the wrong order they are swapped.

    The reported uncertainties have to be swapped with them, or each width is
    quoted with the error of the other component.
    """
    fit_values = _double_gaussian_fit_values(60.0, 18.0, 0.3)
    uncertainties = _double_gaussian_uncertainties()

    result, reported = _result_parameters(fit_values, uncertainties, "double_gaussian")

    assert result["mass_resolution_narrow"] == 18.0
    assert result["mass_resolution_wide"] == 60.0
    assert result["signal_fraction"] == pytest.approx(0.7)
    # 0.5 was the error on the 18 MeV width, which is now the narrow component.
    assert reported["mass_resolution_narrow"] == 0.5
    assert reported["mass_resolution_wide"] == 5.0
    # The caller's dictionary must not be mutated in place.
    assert uncertainties["mass_resolution_narrow"] == 5.0


def test_ordered_double_gaussian_components_are_left_alone():
    fit_values = _double_gaussian_fit_values(18.0, 60.0, 0.7)
    result, reported = _result_parameters(fit_values, _double_gaussian_uncertainties(), "double_gaussian")

    assert result["mass_resolution_narrow"] == 18.0
    assert result["mass_resolution_wide"] == 60.0
    assert result["signal_fraction"] == pytest.approx(0.7)
    assert reported["mass_resolution_narrow"] == 5.0
    assert reported["mass_resolution_wide"] == 0.5


def test_effective_double_gaussian_resolution_has_no_quoted_uncertainty():
    """It mixes three correlated parameters, so it needs the covariance matrix."""
    fit_values = _double_gaussian_fit_values(18.0, 60.0, 0.7)
    result, reported = _result_parameters(fit_values, _double_gaussian_uncertainties(), "double_gaussian")

    expected = np.sqrt(0.7 * 18.0**2 + 0.3 * 60.0**2)
    assert result["mass_resolution"] == pytest.approx(expected)
    assert "mass_resolution" in reported
    assert reported["mass_resolution"] is None


def test_barlow_removes_the_statistical_part_of_a_shift():
    # sigma_barlow = sqrt(0.05^2 - 0.03^2) = 0.04, so a 0.05 shift keeps 0.03.
    result = barlow_shift(0.05, 0.05, 0.03)

    assert result["sigma_barlow"] == pytest.approx(0.04)
    assert result["barlow_significance"] == pytest.approx(1.25)
    assert result["barlow_corrected_shift"] == pytest.approx(0.03)


def test_a_shift_within_its_statistical_spread_assigns_nothing():
    result = barlow_shift(-0.01, 0.05, 0.03)

    assert result["barlow_corrected_shift"] == 0.0


def test_barlow_is_symmetric_between_subset_and_superset():
    assert barlow_shift(0.05, 0.03, 0.05) == pytest.approx(barlow_shift(0.05, 0.05, 0.03))


def test_barlow_without_an_uncertainty_gives_nothing():
    assert barlow_shift(0.05, None, 0.03)["barlow_corrected_shift"] is None


def test_likelihood_ratio_chi2_is_zero_for_a_perfect_match_and_counts_empty_bins():
    expected = np.array([5.0, 3.0, 2.0, 1.0])
    counts = np.array([5, 3, 2, 0])

    quality = _fit_quality(counts, expected, n_parameters=1)

    # The empty bin contributes 2 * expected = 2; the others match exactly.
    assert quality["chi2_likelihood_ratio"] == pytest.approx(2.0)
    assert quality["ndof_likelihood_ratio"] == 3
    # The Neyman chi2 silently drops it.
    assert quality["n_fitted_bins"] == 3


def _asymptotic_p_values(mean_count: float, n_bins: int = 60, n_trials: int = 2000):
    rng = np.random.default_rng(11)
    expected = np.full(n_bins, mean_count)
    return np.array(
        [
            _fit_quality(rng.poisson(expected), expected, n_parameters=0)[
                "p_value_likelihood_ratio_asymptotic"
            ]
            for _ in range(n_trials)
        ]
    )


def test_asymptotic_likelihood_ratio_p_value_is_calibrated_at_high_counts():
    p_values = _asymptotic_p_values(mean_count=25.0)

    assert np.mean(p_values) == pytest.approx(0.5, abs=0.03)
    assert np.mean(p_values < 0.05) == pytest.approx(0.05, abs=0.015)


def test_asymptotic_likelihood_ratio_p_value_is_not_calibrated_at_low_counts():
    """Why the quoted goodness-of-fit p-value is taken from toys instead.

    With a few candidates per bin the asymptotic formula rejects a correct
    model well above its nominal rate; with mostly empty bins it errs the other
    way. Either way it cannot be quoted for these spectra.
    """
    p_values = _asymptotic_p_values(mean_count=2.0)

    assert np.mean(p_values < 0.05) > 0.10


def test_single_gaussian_fit_uses_one_starting_point():
    """The nominal model must not change behaviour with the multi-start search."""
    masses = _generate_pseudo_data(2000, 500, 5280.0, 20.0, slope=-0.004, seed=3)

    status = fit_mass_distribution(masses, fit_range=FIT_RANGE)["fit_status"]

    assert status["n_starting_points"] == 1


def test_double_gaussian_fit_tries_several_starting_points():
    masses = _generate_pseudo_data(2000, 500, 5280.0, 20.0, slope=-0.004, seed=3)

    status = fit_mass_distribution(masses, fit_range=FIT_RANGE, signal_model="double_gaussian")["fit_status"]

    assert status["n_starting_points"] > 1
    assert status["valid"]
