"""Checks of the simultaneous B+/B- fit and of the polarity combination."""

import numpy as np
import pytest

from src.fitting.simultaneous import combine_polarities, fit_charge_simultaneous
from studies.toy_mc.run_raw_asymmetry_toys import generate_toy_masses


FIT_RANGE = (5200.0, 5600.0)


def _truth(signal_yield, background_yield, slope=-0.006):
    return {
        "signal_yield": signal_yield,
        "background_yield": background_yield,
        "mass_mean": 5284.0,
        "mass_resolution": 19.5,
        "background_slope": slope,
        "signal_model": "gaussian",
        "background_model": "exponential",
    }


def _pseudo_data(seed, n_total=3000.0, asymmetry=-0.05, background=(150.0, 100.0)):
    rng = np.random.default_rng(seed)
    plus, _, _ = generate_toy_masses(rng, _truth(n_total * (1 - asymmetry) / 2, background[0]), FIT_RANGE)
    minus, _, _ = generate_toy_masses(rng, _truth(n_total * (1 + asymmetry) / 2, background[1]), FIT_RANGE)
    return plus, minus


def test_simultaneous_fit_recovers_the_injected_asymmetry():
    plus, minus = _pseudo_data(seed=5)

    result = fit_charge_simultaneous(plus, minus, fit_range=FIT_RANGE)

    assert result["fit_status"]["valid"]
    assert abs(result["A_raw"] - (-0.05)) < 3.0 * result["sigma_A_raw"]
    assert abs(result["mass_mean"] - 5284.0) < 3.0 * result["parameter_uncertainties"]["mass_mean"]
    assert abs(result["background_yield_plus"] - 150.0) < 3.0 * result["parameter_uncertainties"]["background_yield_plus"]


def test_charge_yields_are_consistent_with_the_total_and_the_asymmetry():
    plus, minus = _pseudo_data(seed=6)

    result = fit_charge_simultaneous(plus, minus, fit_range=FIT_RANGE)

    assert result["N_plus"] + result["N_minus"] == pytest.approx(result["signal_yield"])
    assert (result["N_minus"] - result["N_plus"]) / result["signal_yield"] == pytest.approx(result["A_raw"])
    assert result["sigma_N_plus"] > 0 and result["sigma_N_minus"] > 0


def test_asymmetry_pulls_have_unit_width():
    """The fitted uncertainty must match the spread of the fitted asymmetry."""
    pulls = []
    for seed in range(200):
        plus, minus = _pseudo_data(seed=1000 + seed, n_total=1300.0, background=(85.0, 30.0))
        result = fit_charge_simultaneous(plus, minus, fit_range=FIT_RANGE)
        if result["fit_status"]["valid"]:
            pulls.append((result["A_raw"] - (-0.05)) / result["sigma_A_raw"])
    pulls = np.array(pulls)

    assert len(pulls) >= 195
    assert np.mean(pulls) == pytest.approx(0.0, abs=0.2)
    assert np.std(pulls, ddof=1) == pytest.approx(1.0, abs=0.12)


def test_double_gaussian_simultaneous_fit_converges():
    plus, minus = _pseudo_data(seed=7)

    result = fit_charge_simultaneous(plus, minus, fit_range=FIT_RANGE, signal_model="double_gaussian")

    assert result["fit_status"]["valid"]
    assert result["fit_status"]["n_starting_points"] > 1
    assert result["mass_resolution_narrow"] <= result["mass_resolution_wide"]


def test_polarities_are_averaged_with_equal_weights():
    """Equal weights cancel a polarity-odd detection asymmetry; precision weights do not."""
    combined = combine_polarities(
        {
            "MagnetDown": {"A_raw": -0.02, "sigma_A_raw": 0.03},
            "MagnetUp": {"A_raw": -0.06, "sigma_A_raw": 0.04},
        }
    )

    assert combined["A_raw"] == pytest.approx(-0.04)
    assert combined["sigma_A_raw"] == pytest.approx(np.hypot(0.03, 0.04) / 2)
    assert combined["n_polarities"] == 2


def test_a_polarity_without_an_asymmetry_gives_no_combination():
    combined = combine_polarities(
        {"MagnetDown": {"A_raw": None, "sigma_A_raw": None}, "MagnetUp": {"A_raw": -0.06, "sigma_A_raw": 0.04}}
    )

    assert combined["A_raw"] is None
