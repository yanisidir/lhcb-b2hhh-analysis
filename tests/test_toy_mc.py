"""Checks of the toy-MC generator used by ``studies/toy_mc``.

The toys have to be generated from the same mass model the baseline fit used.
The two background shapes parameterise ``background_slope`` differently -- MeV^-1
for the exponential, dimensionless for the linear one -- so generating one with
the other's slope produces a silently wrong sample rather than an error.
"""

import numpy as np
import pytest

from src.fitting.mass_models import exponential_background_cdf, linear_background_cdf
from studies.toy_mc.run_raw_asymmetry_toys import sample_background_masses


FIT_RANGE = (5200.0, 5600.0)
N_SAMPLES = 200_000
PROBES = (5250.0, 5400.0, 5550.0)
# Binomial uncertainty on an empirical CDF at this sample size is ~0.0011.
CDF_TOLERANCE = 4.0e-3


def _empirical_cdf(sample, probe: float) -> float:
    return float(np.mean(sample <= probe))


@pytest.mark.parametrize("slope", [-0.009, 0.004])
def test_exponential_background_toys_follow_the_fitted_cdf(slope):
    rng = np.random.default_rng(20240502)
    fit_result = {"background_slope": slope, "background_model": "exponential"}

    sample = sample_background_masses(rng, fit_result, N_SAMPLES, FIT_RANGE)

    assert len(sample) == N_SAMPLES
    assert sample.min() >= FIT_RANGE[0]
    assert sample.max() <= FIT_RANGE[1]
    for probe in PROBES:
        expected = float(exponential_background_cdf(probe, slope, FIT_RANGE))
        assert _empirical_cdf(sample, probe) == pytest.approx(expected, abs=CDF_TOLERANCE)


@pytest.mark.parametrize("slope", [0.0, 0.6, -0.6, 0.95, -0.95])
def test_linear_background_toys_follow_the_fitted_cdf(slope):
    rng = np.random.default_rng(20240503)
    fit_result = {"background_slope": slope, "background_model": "linear"}

    sample = sample_background_masses(rng, fit_result, N_SAMPLES, FIT_RANGE)

    assert len(sample) == N_SAMPLES
    assert sample.min() >= FIT_RANGE[0]
    assert sample.max() <= FIT_RANGE[1]
    for probe in PROBES:
        expected = float(linear_background_cdf(probe, slope, FIT_RANGE))
        assert _empirical_cdf(sample, probe) == pytest.approx(expected, abs=CDF_TOLERANCE)


def test_the_background_model_defaults_to_the_exponential():
    """Baseline files written before the model was recorded stay reproducible."""
    rng = np.random.default_rng(7)
    without_model = sample_background_masses(
        rng,
        {"background_slope": -0.009},
        1000,
        FIT_RANGE,
    )
    rng = np.random.default_rng(7)
    with_model = sample_background_masses(
        rng,
        {"background_slope": -0.009, "background_model": "exponential"},
        1000,
        FIT_RANGE,
    )
    assert without_model == pytest.approx(with_model)


def test_an_unsupported_background_model_is_refused():
    rng = np.random.default_rng(1)
    fit_result = {"background_slope": 0.0, "background_model": "polynomial"}

    with pytest.raises(ValueError, match="polynomial"):
        sample_background_masses(rng, fit_result, 10, FIT_RANGE)


def test_an_out_of_range_linear_slope_is_refused():
    rng = np.random.default_rng(1)
    fit_result = {"background_slope": 1.5, "background_model": "linear"}

    with pytest.raises(ValueError, match="slope"):
        sample_background_masses(rng, fit_result, 10, FIT_RANGE)
