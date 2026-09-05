#!/usr/bin/env python
"""Run charge-separated nominal B -> hhh invariant-mass fits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from run_mass_fit import DEFAULT_DATA_PATH, _display_path, _json_safe, _plot_fit
from src.fitting.fit_mass import fit_mass_distribution
from src.physics.selection import prepare_selected_candidates


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "fits" / "charge_separated"


def _fit_charge_sample(
    masses,
    fit_range,
    bins: int,
    signal_model: str,
    background_model: str,
):
    return fit_mass_distribution(
        masses,
        fit_range=fit_range,
        bins=bins,
        signal_model=signal_model,
        background_model=background_model,
    )


def _signal_yield_uncertainty(fit_result) -> float | None:
    uncertainty = fit_result.get("parameter_uncertainties", {}).get("signal_yield")
    return None if uncertainty is None else float(uncertainty)


def _raw_asymmetry(plus_result, minus_result, n_plus_candidates: int, n_minus_candidates: int):
    n_plus = float(plus_result["signal_yield"])
    n_minus = float(minus_result["signal_yield"])
    sigma_n_plus = _signal_yield_uncertainty(plus_result)
    sigma_n_minus = _signal_yield_uncertainty(minus_result)

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


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help="Input ROOT ntuple.")
    parser.add_argument("--bins", type=int, default=80, help="Number of fitted mass bins.")
    parser.add_argument(
        "--signal-model",
        choices=("gaussian", "double_gaussian"),
        default="gaussian",
        help="Signal mass model used in the nominal charge-separated fits.",
    )
    parser.add_argument(
        "--background-model",
        choices=("exponential", "linear"),
        default="exponential",
        help="Background mass model used in the nominal charge-separated fits.",
    )
    parser.add_argument(
        "--fit-range",
        type=float,
        nargs=2,
        default=(5200.0, 5600.0),
        metavar=("LOW", "HIGH"),
        help="Fitted invariant-mass range in MeV/c^2.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for charge-separated mass-fit outputs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_path = args.data.resolve()
    output_dir = args.output_dir.resolve()
    fit_range = tuple(args.fit_range)

    if not data_path.exists():
        raise FileNotFoundError(f"Input ROOT file not found: {data_path}")

    selected = prepare_selected_candidates(data_path)
    selected = selected.copy()
    selected["total_charge"] = (
        selected["H1_Charge"] + selected["H2_Charge"] + selected["H3_Charge"]
    )
    b_plus = selected.loc[selected["total_charge"] == 1].copy()
    b_minus = selected.loc[selected["total_charge"] == -1].copy()

    plus_masses = b_plus["B_mass"].to_numpy()
    minus_masses = b_minus["B_mass"].to_numpy()

    plus_result = _fit_charge_sample(
        plus_masses,
        fit_range,
        args.bins,
        args.signal_model,
        args.background_model,
    )
    minus_result = _fit_charge_sample(
        minus_masses,
        fit_range,
        args.bins,
        args.signal_model,
        args.background_model,
    )

    shared_metadata = {
        "note": "Charge-separated first nominal mass fit; not yet a full systematic study.",
        "data_path": str(_display_path(data_path)),
        "mass_column": "B_mass",
        "charge_definition": "total_charge = H1_Charge + H2_Charge + H3_Charge",
        "selection": "src.physics.selection.prepare_selected_candidates",
    }
    plus_result.update({**shared_metadata, "charge_sample": "B_plus", "total_charge": 1})
    minus_result.update({**shared_metadata, "charge_sample": "B_minus", "total_charge": -1})

    asymmetry = _raw_asymmetry(
        plus_result,
        minus_result,
        n_plus_candidates=len(b_plus),
        n_minus_candidates=len(b_minus),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    plus_results_path = output_dir / "B_plus_fit_results.json"
    minus_results_path = output_dir / "B_minus_fit_results.json"
    plus_plot_path = output_dir / "B_plus_fit_plot.png"
    minus_plot_path = output_dir / "B_minus_fit_plot.png"
    asymmetry_path = output_dir / "raw_asymmetry.json"

    plus_results_path.write_text(
        json.dumps(_json_safe(plus_result), indent=2) + "\n",
        encoding="utf-8",
    )
    minus_results_path.write_text(
        json.dumps(_json_safe(minus_result), indent=2) + "\n",
        encoding="utf-8",
    )
    asymmetry_path.write_text(
        json.dumps(_json_safe(asymmetry), indent=2) + "\n",
        encoding="utf-8",
    )
    _plot_fit(
        plus_masses,
        plus_result,
        plus_plot_path,
        fit_range=fit_range,
        bins=args.bins,
        title=r"$B^+$ candidates: invariant-mass fit",
    )
    _plot_fit(
        minus_masses,
        minus_result,
        minus_plot_path,
        fit_range=fit_range,
        bins=args.bins,
        title=r"$B^-$ candidates: invariant-mass fit",
    )

    print(f"B+ candidates:            {len(b_plus):,}")
    print(f"B- candidates:            {len(b_minus):,}")
    print(f"Signal model:             {args.signal_model}")
    print(f"Background model:         {args.background_model}")
    print(f"N_plus:                   {asymmetry['N_plus']:.1f}")
    print(f"N_minus:                  {asymmetry['N_minus']:.1f}")
    print(f"A_raw:                    {asymmetry['A_raw']:.5f}")
    if asymmetry["sigma_A_raw"] is None:
        print("sigma_A_raw:              n/a")
    else:
        print(f"sigma_A_raw:              {asymmetry['sigma_A_raw']:.5f}")
    print(f"Wrote {_display_path(plus_results_path)}")
    print(f"Wrote {_display_path(minus_results_path)}")
    print(f"Wrote {_display_path(plus_plot_path)}")
    print(f"Wrote {_display_path(minus_plot_path)}")
    print(f"Wrote {_display_path(asymmetry_path)}")


if __name__ == "__main__":
    main()
