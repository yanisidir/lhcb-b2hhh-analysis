"""Acceptance criteria for systematic fit variations.

A variation is only allowed to define a systematic uncertainty if both
charge-separated fits describe the data acceptably. Keeping these criteria in
one place, applied identically by every systematic study, is what makes them
pre-registered rather than chosen after seeing the shifts.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np


DEFAULT_MAX_CHI2_NDOF = 2.5


def status_summary(fit_result: Mapping) -> dict:
    """Extract the convergence status of a fit result."""
    status = fit_result.get("fit_status", {})
    return {
        "success": bool(status.get("success")),
        "valid": bool(status.get("valid")),
        "backend": status.get("backend"),
        "message": status.get("message"),
    }


def _positive(value) -> bool:
    return value is not None and np.isfinite(value) and value > 0.0


def fit_quality_acceptance(row: Mapping, max_chi2_ndof: float = DEFAULT_MAX_CHI2_NDOF):
    """Decide whether a variation may be used for a systematic uncertainty.

    Returns ``(accepted, reason)``. A variation is accepted when both charge
    fits converged to a valid minimum, both yields and yield uncertainties are
    positive, and both chi2/ndof values stay below ``max_chi2_ndof``. Variations
    flagged as stress tests are never accepted.
    """
    reasons = []
    if not row["fit_success_plus"] or not row["fit_valid_plus"]:
        reasons.append("B+ fit did not converge")
    if not row["fit_success_minus"] or not row["fit_valid_minus"]:
        reasons.append("B- fit did not converge")
    if row["chi2_ndof_plus"] is None or row["chi2_ndof_plus"] >= max_chi2_ndof:
        reasons.append(f"B+ chi2/ndof >= {max_chi2_ndof:g}")
    if row["chi2_ndof_minus"] is None or row["chi2_ndof_minus"] >= max_chi2_ndof:
        reasons.append(f"B- chi2/ndof >= {max_chi2_ndof:g}")
    if not _positive(row["N_plus"]):
        reasons.append("B+ fitted yield is not positive")
    if not _positive(row["N_minus"]):
        reasons.append("B- fitted yield is not positive")
    if not _positive(row["sigma_N_plus"]):
        reasons.append("B+ fitted yield uncertainty is not positive")
    if not _positive(row["sigma_N_minus"]):
        reasons.append("B- fitted yield uncertainty is not positive")
    if row.get("variation_role") == "stress_test":
        reasons.append("stress-test model, not accepted for systematic assignment")

    return len(reasons) == 0, "; ".join(reasons) if reasons else ""


def barlow_shift(delta, sigma_variation, sigma_nominal) -> dict:
    """Separate a variation's shift from the statistical spread it must have.

    When a variation is fitted on a subset or a superset of the nominal sample,
    as for a change of fit range, the two results differ even with no
    systematic effect at all, because the candidates they do not share
    fluctuate. The expected size of that difference is
    ``sqrt(|sigma_variation^2 - sigma_nominal^2|)`` (R. Barlow, "Systematic
    errors: facts and fictions", hep-ex/0207026). Only the part of the shift
    beyond it is evidence of a systematic effect, so the contribution assigned
    is ``sqrt(max(delta^2 - sigma_barlow^2, 0))``.

    This does not apply to model variations fitted on the same candidates, whose
    difference has no such statistical component.
    """
    if delta is None or sigma_variation is None or sigma_nominal is None:
        return {"sigma_barlow": None, "barlow_significance": None, "barlow_corrected_shift": None}

    sigma_barlow = float(np.sqrt(abs(float(sigma_variation) ** 2 - float(sigma_nominal) ** 2)))
    delta = abs(float(delta))
    significance = delta / sigma_barlow if sigma_barlow > 0 else None
    corrected = float(np.sqrt(max(delta**2 - sigma_barlow**2, 0.0)))
    return {
        "sigma_barlow": sigma_barlow,
        "barlow_significance": significance,
        "barlow_corrected_shift": corrected,
    }


def simultaneous_fit_acceptance(results_by_polarity: Mapping, max_chi2_ndof: float = DEFAULT_MAX_CHI2_NDOF):
    """Apply the same pre-registered criteria to simultaneous B+/B- fits.

    Every polarity's fit must converge to a valid minimum, both charge yields
    and their uncertainties must be positive, and the chi2/ndof of each of the
    two charge spectra must stay below ``max_chi2_ndof``. Returns
    ``(accepted, reason)``.
    """
    reasons = []
    for polarity, result in results_by_polarity.items():
        status = result.get("fit_status", {})
        if not status.get("success") or not status.get("valid"):
            reasons.append(f"{polarity}: fit did not converge")
            continue
        for charge, label in (("plus", "B+"), ("minus", "B-")):
            chi2_ndof = result.get("fit_quality", {}).get(charge, {}).get("chi2_ndof")
            if chi2_ndof is None or chi2_ndof >= max_chi2_ndof:
                reasons.append(f"{polarity} {label} chi2/ndof >= {max_chi2_ndof:g}")
            if not _positive(result.get(f"N_{charge}")):
                reasons.append(f"{polarity} {label} fitted yield is not positive")
            if not _positive(result.get(f"sigma_N_{charge}")):
                reasons.append(f"{polarity} {label} fitted yield uncertainty is not positive")
    return len(reasons) == 0, "; ".join(reasons)
