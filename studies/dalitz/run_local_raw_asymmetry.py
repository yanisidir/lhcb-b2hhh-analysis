#!/usr/bin/env python
"""Compute a binned local raw asymmetry over the Dalitz plane."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
for path in (PROJECT_ROOT, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_mass_fit import DEFAULT_DATA_PATH, _display_path, _json_safe
from src.physics.dalitz import compute_dalitz_variables
from src.physics.selection import prepare_selected_candidates
from src.plotting.mass_plots import set_hep_style


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "dalitz" / "local_asymmetry"
DEFAULT_X_VAR = "m_K1K2_sq"
DEFAULT_Y_VAR = "m_K2K3_sq"
CAVEAT = (
    "Raw local asymmetry diagnostic only; no corrections are applied for detector "
    "efficiency, background contamination, production asymmetry, migration across "
    "Dalitz bins, or phase-space acceptance."
)


def _configure_matplotlib_cache() -> None:
    cache_root = Path(tempfile.gettempdir()) / "lhcb-b2hhh-analysis"
    matplotlib_cache = cache_root / "matplotlib"
    xdg_cache = cache_root / "xdg-cache"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    xdg_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache))


def _finite_range(values, requested_range: tuple[float, float] | None = None) -> tuple[float, float]:
    if requested_range is not None:
        low, high = requested_range
    else:
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            raise ValueError("Cannot build Dalitz binning: no finite values found.")
        low = float(np.min(finite))
        high = float(np.max(finite))

    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        raise ValueError(f"Invalid Dalitz range: ({low}, {high})")
    return low, high


def load_selected_with_dalitz(
    data_path: Path,
    x_var: str = DEFAULT_X_VAR,
    y_var: str = DEFAULT_Y_VAR,
):
    """Load selected B candidates and attach Dalitz coordinates if needed.

    The returned table uses the shared baseline selection and the project
    convention for kaon-hypothesis pair masses. Charge separation is defined by
    the sum of the three daughter charges.
    """
    selected = prepare_selected_candidates(data_path).copy()

    missing = {x_var, y_var}.difference(selected.columns)
    if missing:
        dalitz = compute_dalitz_variables(selected)
        for column in dalitz.columns:
            if column not in selected.columns:
                selected[column] = dalitz[column]

    missing = {x_var, y_var}.difference(selected.columns)
    if missing:
        available = ", ".join(sorted(selected.columns))
        raise KeyError(f"Missing Dalitz variable(s) {sorted(missing)}. Available columns: {available}")

    selected["total_charge"] = (
        selected["H1_Charge"] + selected["H2_Charge"] + selected["H3_Charge"]
    )

    finite_mask = (
        np.isfinite(selected[x_var].to_numpy(dtype=float))
        & np.isfinite(selected[y_var].to_numpy(dtype=float))
        & selected["total_charge"].isin([-1, 1]).to_numpy()
    )
    return selected.loc[finite_mask].copy()


def _raw_asymmetry_from_counts(n_plus, n_minus):
    n_plus = np.asarray(n_plus, dtype=float)
    n_minus = np.asarray(n_minus, dtype=float)
    n_total = n_plus + n_minus

    a_raw = np.full_like(n_total, np.nan, dtype=float)
    sigma_a_raw = np.full_like(n_total, np.nan, dtype=float)
    significance = np.full_like(n_total, np.nan, dtype=float)

    populated = n_total > 0
    a_raw[populated] = (n_minus[populated] - n_plus[populated]) / n_total[populated]
    sigma_a_raw[populated] = np.sqrt(
        np.maximum(0.0, 1.0 - a_raw[populated] ** 2) / n_total[populated]
    )

    finite_error = sigma_a_raw > 0
    significance[finite_error] = a_raw[finite_error] / sigma_a_raw[finite_error]
    return a_raw, sigma_a_raw, significance


def compute_local_asymmetry(
    candidates,
    x_var: str = DEFAULT_X_VAR,
    y_var: str = DEFAULT_Y_VAR,
    bins_x: int = 10,
    bins_y: int = 10,
    min_count: int = 20,
    x_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
):
    """Return binned Dalitz-plane raw asymmetries for B+ and B- candidates."""
    x_low, x_high = _finite_range(candidates[x_var].to_numpy(), x_range)
    y_low, y_high = _finite_range(candidates[y_var].to_numpy(), y_range)
    x_edges = np.linspace(x_low, x_high, bins_x + 1)
    y_edges = np.linspace(y_low, y_high, bins_y + 1)

    b_plus = candidates.loc[candidates["total_charge"] == 1]
    b_minus = candidates.loc[candidates["total_charge"] == -1]

    n_plus, _, _ = np.histogram2d(
        b_plus[x_var].to_numpy(),
        b_plus[y_var].to_numpy(),
        bins=(x_edges, y_edges),
    )
    n_minus, _, _ = np.histogram2d(
        b_minus[x_var].to_numpy(),
        b_minus[y_var].to_numpy(),
        bins=(x_edges, y_edges),
    )
    n_total = n_plus + n_minus
    a_raw, sigma_a_raw, significance = _raw_asymmetry_from_counts(n_plus, n_minus)
    passes_min_count = n_total >= min_count

    rows = []
    for ix in range(bins_x):
        for iy in range(bins_y):
            rows.append(
                {
                    "bin_x": ix,
                    "bin_y": iy,
                    "x_low": x_edges[ix],
                    "x_high": x_edges[ix + 1],
                    "x_center": 0.5 * (x_edges[ix] + x_edges[ix + 1]),
                    "y_low": y_edges[iy],
                    "y_high": y_edges[iy + 1],
                    "y_center": 0.5 * (y_edges[iy] + y_edges[iy + 1]),
                    "N_plus": int(n_plus[ix, iy]),
                    "N_minus": int(n_minus[ix, iy]),
                    "N_total": int(n_total[ix, iy]),
                    "A_raw": float(a_raw[ix, iy]) if np.isfinite(a_raw[ix, iy]) else np.nan,
                    "sigma_A_raw": (
                        float(sigma_a_raw[ix, iy])
                        if np.isfinite(sigma_a_raw[ix, iy])
                        else np.nan
                    ),
                    "significance": (
                        float(significance[ix, iy])
                        if np.isfinite(significance[ix, iy])
                        else np.nan
                    ),
                    "passes_min_count": bool(passes_min_count[ix, iy]),
                }
            )

    return {
        "x_edges": x_edges,
        "y_edges": y_edges,
        "N_plus": n_plus,
        "N_minus": n_minus,
        "N_total": n_total,
        "A_raw": a_raw,
        "sigma_A_raw": sigma_a_raw,
        "significance": significance,
        "passes_min_count": passes_min_count,
        "rows": rows,
    }


def _axis_label(variable: str) -> str:
    labels = {
        "m_K1K2_sq": r"$m^2(K_1K_2)$ [GeV$^2/c^4$]",
        "m_K2K3_sq": r"$m^2(K_2K_3)$ [GeV$^2/c^4$]",
        "m_K1K3_sq": r"$m^2(K_1K_3)$ [GeV$^2/c^4$]",
    }
    return labels.get(variable, variable)


def _plot_heatmap(
    values,
    x_edges,
    y_edges,
    output_path: Path,
    x_var: str,
    y_var: str,
    title: str,
    colorbar_label: str,
    cmap: str,
    vmin: float | None = None,
    vmax: float | None = None,
):
    _configure_matplotlib_cache()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    set_hep_style()
    fig, ax = plt.subplots(figsize=(7.0, 5.6))
    mesh = ax.pcolormesh(
        x_edges,
        y_edges,
        np.asarray(values, dtype=float).T,
        cmap=cmap,
        shading="auto",
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xlabel(_axis_label(x_var))
    ax.set_ylabel(_axis_label(y_var))
    ax.set_title(title)
    colorbar = fig.colorbar(mesh, ax=ax)
    colorbar.set_label(colorbar_label)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_plots(result, output_dir: Path, x_var: str, y_var: str, min_count: int):
    """Write Dalitz-plane maps for local raw asymmetry, significance, and counts."""
    passing = result["passes_min_count"]
    a_raw_plot = np.where(passing, result["A_raw"], np.nan)
    significance_plot = np.where(passing, result["significance"], np.nan)

    finite_significance = np.abs(significance_plot[np.isfinite(significance_plot)])
    significance_limit = max(1.0, float(np.max(finite_significance))) if finite_significance.size else 1.0

    _plot_heatmap(
        a_raw_plot,
        result["x_edges"],
        result["y_edges"],
        output_dir / "local_A_raw_map.png",
        x_var,
        y_var,
        title=f"Local raw asymmetry, bins with N >= {min_count}",
        colorbar_label=r"$A_{\mathrm{raw}}$",
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
    )
    _plot_heatmap(
        significance_plot,
        result["x_edges"],
        result["y_edges"],
        output_dir / "local_significance_map.png",
        x_var,
        y_var,
        title=f"Local raw-asymmetry significance, bins with N >= {min_count}",
        colorbar_label=r"$A_{\mathrm{raw}}/\sigma(A_{\mathrm{raw}})$",
        cmap="coolwarm",
        vmin=-significance_limit,
        vmax=significance_limit,
    )
    _plot_heatmap(
        result["N_total"],
        result["x_edges"],
        result["y_edges"],
        output_dir / "local_counts_map.png",
        x_var,
        y_var,
        title="Selected-candidate counts over the Dalitz plane",
        colorbar_label="Candidates",
        cmap="viridis",
    )


def build_summary(
    candidates,
    result,
    data_path: Path,
    x_var: str,
    y_var: str,
    bins_x: int,
    bins_y: int,
    min_count: int,
):
    n_plus = int((candidates["total_charge"] == 1).sum())
    n_minus = int((candidates["total_charge"] == -1).sum())
    global_a_raw, global_sigma, _ = _raw_asymmetry_from_counts(
        np.asarray([n_plus]),
        np.asarray([n_minus]),
    )

    passing = result["passes_min_count"]
    significance = np.where(passing, result["significance"], np.nan)
    finite_significance = np.abs(significance[np.isfinite(significance)])
    max_abs_significance = (
        float(np.max(finite_significance)) if finite_significance.size else None
    )

    return {
        "study": "local Dalitz-plane raw asymmetry",
        "data_path": str(_display_path(data_path)),
        "selection": "src.physics.selection.prepare_selected_candidates",
        "dalitz_variables": {
            "x": x_var,
            "y": y_var,
            "source": "src.physics.dalitz.compute_dalitz_variables",
        },
        "charge_definition": "total_charge = H1_Charge + H2_Charge + H3_Charge",
        "bins_x": bins_x,
        "bins_y": bins_y,
        "min_count": min_count,
        "x_edges": result["x_edges"].tolist(),
        "y_edges": result["y_edges"].tolist(),
        "n_candidates": int(len(candidates)),
        "n_candidates_plus": n_plus,
        "n_candidates_minus": n_minus,
        "global_counting_A_raw": (
            float(global_a_raw[0]) if np.isfinite(global_a_raw[0]) else None
        ),
        "global_counting_sigma_A_raw": (
            float(global_sigma[0]) if np.isfinite(global_sigma[0]) else None
        ),
        "n_bins_total": int(bins_x * bins_y),
        "n_bins_passing_min_count": int(np.sum(passing)),
        "n_bins_masked_low_count": int(np.sum(~passing)),
        "max_abs_significance_passing_bins": max_abs_significance,
        "caveat": CAVEAT,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help="Input ROOT ntuple.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for local Dalitz-asymmetry outputs.",
    )
    parser.add_argument("--bins-x", type=int, default=10, help="Number of Dalitz bins along x.")
    parser.add_argument("--bins-y", type=int, default=10, help="Number of Dalitz bins along y.")
    parser.add_argument(
        "--min-count",
        type=int,
        default=20,
        help="Minimum total bin count required for plotted asymmetry values.",
    )
    parser.add_argument(
        "--x-var",
        default=DEFAULT_X_VAR,
        help="Dalitz variable for the horizontal axis.",
    )
    parser.add_argument(
        "--y-var",
        default=DEFAULT_Y_VAR,
        help="Dalitz variable for the vertical axis.",
    )
    parser.add_argument(
        "--x-range",
        type=float,
        nargs=2,
        default=None,
        metavar=("LOW", "HIGH"),
        help="Optional x-axis Dalitz range in GeV^2/c^4.",
    )
    parser.add_argument(
        "--y-range",
        type=float,
        nargs=2,
        default=None,
        metavar=("LOW", "HIGH"),
        help="Optional y-axis Dalitz range in GeV^2/c^4.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_path = args.data.resolve()
    output_dir = args.output_dir.resolve()

    if args.bins_x <= 0 or args.bins_y <= 0:
        raise ValueError("--bins-x and --bins-y must be positive.")
    if args.min_count < 0:
        raise ValueError("--min-count must be non-negative.")
    if not data_path.exists():
        raise FileNotFoundError(f"Input ROOT file not found: {data_path}")

    candidates = load_selected_with_dalitz(data_path, args.x_var, args.y_var)
    result = compute_local_asymmetry(
        candidates,
        x_var=args.x_var,
        y_var=args.y_var,
        bins_x=args.bins_x,
        bins_y=args.bins_y,
        min_count=args.min_count,
        x_range=tuple(args.x_range) if args.x_range is not None else None,
        y_range=tuple(args.y_range) if args.y_range is not None else None,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    bins_path = output_dir / "local_asymmetry_bins.csv"
    summary_path = output_dir / "local_asymmetry_summary.json"

    import pandas as pd

    pd.DataFrame(result["rows"]).to_csv(bins_path, index=False)
    summary = build_summary(
        candidates,
        result,
        data_path=data_path,
        x_var=args.x_var,
        y_var=args.y_var,
        bins_x=args.bins_x,
        bins_y=args.bins_y,
        min_count=args.min_count,
    )
    summary_path.write_text(
        json.dumps(_json_safe(summary), indent=2) + "\n",
        encoding="utf-8",
    )
    write_plots(result, output_dir, args.x_var, args.y_var, args.min_count)

    print(f"Selected finite Dalitz candidates: {len(candidates):,}")
    print(f"B+ candidates:                   {summary['n_candidates_plus']:,}")
    print(f"B- candidates:                   {summary['n_candidates_minus']:,}")
    print(
        "Global counting A_raw:           "
        f"{summary['global_counting_A_raw']:.5f} +/- {summary['global_counting_sigma_A_raw']:.5f}"
    )
    print(
        "Bins passing min-count:          "
        f"{summary['n_bins_passing_min_count']} / {summary['n_bins_total']}"
    )
    print(f"Wrote {_display_path(bins_path)}")
    print(f"Wrote {_display_path(summary_path)}")
    print(f"Wrote {_display_path(output_dir / 'local_A_raw_map.png')}")
    print(f"Wrote {_display_path(output_dir / 'local_significance_map.png')}")
    print(f"Wrote {_display_path(output_dir / 'local_counts_map.png')}")


if __name__ == "__main__":
    main()
