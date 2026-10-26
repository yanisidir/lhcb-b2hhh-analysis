"""Checks of how the systematic uncertainty on the raw asymmetry is assigned."""

import pytest

from studies.systematics.run_joint_variations import summarize
from studies.systematics.summarize_systematics import build_summary


def _row(model, fit_range, a_raw, accepted=True, nominal=False):
    return {
        "model_name": model,
        "range_name": fit_range,
        "A_raw": a_raw,
        "is_nominal": nominal,
        "accepted_for_systematic": accepted,
    }


def test_the_joint_systematic_is_the_largest_accepted_shift():
    rows = [
        _row("nominal", "5200_5600", -0.035, nominal=True),
        _row("nominal", "5230_5600", -0.004),
        _row("double", "5200_5550", -0.079),
        _row("linear", "5200_5600", -0.044),
    ]

    summary = summarize(rows, bins=80, max_chi2_ndof=2.5)

    assert summary["recommended_fit_systematic"] == pytest.approx(0.044)
    assert summary["largest_shift_configuration"] == {"model_name": "double", "range_name": "5200_5550"}
    assert (summary["accepted_A_raw_min"], summary["accepted_A_raw_max"]) == pytest.approx((-0.079, -0.004))


def test_rejected_configurations_do_not_enter_the_envelope():
    rows = [
        _row("nominal", "5200_5600", -0.035, nominal=True),
        _row("double", "5200_5600", -0.200, accepted=False),
        _row("linear", "5200_5600", -0.044),
    ]

    summary = summarize(rows, bins=80, max_chi2_ndof=2.5)

    assert summary["recommended_fit_systematic"] == pytest.approx(0.009)
    assert summary["n_accepted_configurations"] == 2


def test_the_summary_quotes_the_joint_envelope_not_the_quadrature_sum():
    summary = build_summary(
        {"A_raw": -0.035, "sigma_A_raw": 0.029},
        {"recommended_model_systematic": 0.030},
        {"recommended_fit_range_systematic": 0.027},
        {"recommended_fit_systematic": 0.044, "systematic_definition": "joint envelope"},
    )

    assert summary["total_systematic_uncertainty"] == pytest.approx(0.044)
    assert summary["quadrature_if_independent"] == pytest.approx((0.030**2 + 0.027**2) ** 0.5)
    assert summary["total_uncertainty"] == pytest.approx((0.029**2 + 0.044**2) ** 0.5)
