"""Raw charge asymmetry built from charge-separated fitted yields."""

from __future__ import annotations

from typing import Mapping


def signal_yield_uncertainty(fit_result: Mapping) -> float | None:
    """Return the fitted signal-yield uncertainty, or ``None`` if undetermined."""
    uncertainty = fit_result.get("parameter_uncertainties", {}).get("signal_yield")
    return None if uncertainty is None else float(uncertainty)


def raw_asymmetry(
    plus_result: Mapping,
    minus_result: Mapping,
    n_plus_candidates: int,
    n_minus_candidates: int,
) -> dict:
    """Build the raw asymmetry between fitted B+ and B- signal yields.

    The asymmetry is ``(N- - N+) / (N- + N+)``, and its uncertainty follows from
    propagating the two independent fitted-yield uncertainties. The result is
    raw: it carries no correction for detector, production, trigger,
    reconstruction or selection-induced charge asymmetries.
    """
    n_plus = float(plus_result["signal_yield"])
    n_minus = float(minus_result["signal_yield"])
    sigma_n_plus = signal_yield_uncertainty(plus_result)
    sigma_n_minus = signal_yield_uncertainty(minus_result)

    denominator = n_minus + n_plus
    if denominator <= 0:
        a_raw = None
        sigma_a_raw = None
    else:
        a_raw = (n_minus - n_plus) / denominator
        if sigma_n_plus is None or sigma_n_minus is None:
            sigma_a_raw = None
        else:
            d_a_d_n_plus = -2.0 * n_minus / denominator**2
            d_a_d_n_minus = 2.0 * n_plus / denominator**2
            sigma_a_raw = (
                (d_a_d_n_plus * sigma_n_plus) ** 2
                + (d_a_d_n_minus * sigma_n_minus) ** 2
            ) ** 0.5

    return {
        "N_plus": n_plus,
        "sigma_N_plus": sigma_n_plus,
        "N_minus": n_minus,
        "sigma_N_minus": sigma_n_minus,
        "A_raw": a_raw,
        "sigma_A_raw": sigma_a_raw,
        "fit_model_used": f"{plus_result['signal_model']} + {plus_result['background_model']}",
        "fit_range": plus_result["fit_status"]["fit_range"],
        "n_candidates_plus": int(n_plus_candidates),
        "n_candidates_minus": int(n_minus_candidates),
        "n_fitted_candidates_plus": int(plus_result["fit_status"]["n_fitted_candidates"]),
        "n_fitted_candidates_minus": int(minus_result["fit_status"]["n_fitted_candidates"]),
        "note": (
            "Raw fitted-yield asymmetry only; no corrections for detector, production, "
            "trigger, reconstruction, or selection-induced charge asymmetries are applied."
        ),
    }
