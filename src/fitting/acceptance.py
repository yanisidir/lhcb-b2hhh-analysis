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
