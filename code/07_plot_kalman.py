#!/usr/bin/env python3
"""Regenerate the canonical Kalman manuscript figures (Fig 6, 7, B1, B2).

It uses the shared manuscript plotting module under
``plotting/generate_main_figures.py`` and the model package in
``models/three_state_kalman/``.

Public model outputs already follow the manuscript display order
(exp 1 = dynamic/ramped, exp 2 = fixed/constant), so no experiment remap is
applied here. Figures are rendered to a temporary staging directory and only the
canonical stems are copied into ``figures/``.
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PLOTTING = ROOT / "plotting"
FIG_DIR = ROOT / "figures"
FITS = ROOT / "results" / "kalman_model_fits" / "model_fits_deduplicated.csv"
KALMAN_INPUT = ROOT / "results" / "kalman_model_fits" / "kalman_input_data.csv"

# (staged stem -> canonical manuscript stem)
DUPLICATE_MAP = [
    ("figC1_model_comparison", "figB1_model_comparison"),
    ("fig7_parameter_estimates", "fig6_parameter_estimates"),
    ("figC2_cti_sdi_recovery", "figB2_cti_sdi_recovery"),
    ("fig8_trial_level_serial_dependence", "fig7_trial_level_serial_dependence"),
]
CANONICAL_STEMS = [dst for _, dst in DUPLICATE_MAP]


def parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(description=__doc__).parse_args()


def import_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_model_data():
    """Load canonical fit/input tables (already in manuscript display order)."""
    sys.path.insert(0, str(ROOT / "models"))
    from three_state_kalman import (  # noqa: E402
        C_AXIS,
        S_AXIS,
        B_AXIS,
        compare_axes,
        generate_ppc_single_subject,
        get_best_model,
    )

    results_df = pd.read_csv(FITS)

    df_model = pd.read_csv(KALMAN_INPUT)
    df_model["Structure"] = df_model["Structure"].replace({"Repeat": "Same"})
    df_model = df_model.sort_values(["Sub", "exp", "trial_num"]).copy()
    df_model["preDur"] = df_model.groupby(["Sub", "exp"])["Duration"].shift(1)

    best_exp1, _ = get_best_model(results_df, exp_num=1, criterion="AIC")
    best_exp2, _ = get_best_model(results_df, exp_num=2, criterion="AIC")
    return (
        results_df,
        df_model,
        best_exp1,
        best_exp2,
        C_AXIS,
        S_AXIS,
        B_AXIS,
        compare_axes,
        generate_ppc_single_subject,
        get_best_model,
    )


def duplicate(staging: Path, stem_from: str, stem_to: str) -> None:
    for ext in ("png", "pdf"):
        src = staging / f"{stem_from}.{ext}"
        dst = staging / f"{stem_to}.{ext}"
        if src.exists():
            shutil.copyfile(src, dst)


def main() -> None:
    staging = Path(tempfile.mkdtemp(prefix="kalman_figs_"))

    main_figs = import_from_path(
        "current_main_figures", PLOTTING / "generate_main_figures.py"
    )
    main_figs.FIG_DIR = staging

    (
        results_df,
        df_model,
        best_exp1,
        best_exp2,
        C_AXIS,
        S_AXIS,
        B_AXIS,
        compare_axes,
        generate_ppc_single_subject,
        get_best_model,
    ) = load_model_data()

    main_figs.plot_figC1_model_comparison(results_df, C_AXIS, S_AXIS, B_AXIS, compare_axes)
    main_figs.plot_fig7_parameters(results_df, get_best_model)
    main_figs.plot_figC2_cti_sdi_recovery(df_model, results_df, best_exp1, best_exp2, generate_ppc_single_subject)
    main_figs.plot_fig8_trial_level_sd(df_model, results_df, best_exp1, best_exp2, generate_ppc_single_subject)

    for stem_from, stem_to in DUPLICATE_MAP:
        duplicate(staging, stem_from, stem_to)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for stem in CANONICAL_STEMS:
        for ext in ("png", "pdf"):
            src = staging / f"{stem}.{ext}"
            if src.exists():
                shutil.copyfile(src, FIG_DIR / f"{stem}.{ext}")
    shutil.rmtree(staging, ignore_errors=True)
    print(f"Saved canonical Kalman figures to {FIG_DIR}: {', '.join(CANONICAL_STEMS)}")


if __name__ == "__main__":
    parse_args()
    main()
