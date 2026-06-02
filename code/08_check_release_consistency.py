#!/usr/bin/env python3
"""Release consistency checks for the public repository.

This script verifies that the public release contains the final manuscript/appendix
figure assets, that notebooks explicitly cover those final figures while retaining
broader saved-output records, and that the included data/model outputs reproduce
key headline checks.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

FINAL_FIGURE_STEMS = [
    "fig1_experimental_procedure",
    "fig2_behavioral_results_combined",
    "fig3_cross_experiment_serial_dependence",
    "fig4_temporal_window",
    "fig5_three_state_kalman_filter",
    "fig6_parameter_estimates",
    "fig7_trial_level_serial_dependence",
    "figA1_response_error_lag",
    "figB1_model_comparison",
    "figB2_cti_sdi_recovery",
]

MAIN_NOTEBOOK_STEMS = FINAL_FIGURE_STEMS[:7]
APPENDIX_NOTEBOOK_STEMS = FINAL_FIGURE_STEMS[7:]

EXPECTED_VALID_ROWS = {1: 4693, 2: 4654}
EXPECTED_SUBJECTS = {1: 22, 2: 22}
EXPECTED_WINNERS = {1: "C1_S0_B2", 2: "C1_S1_B2"}


def fail(message: str) -> None:
    raise AssertionError(message)


def check_figures() -> None:
    figures = ROOT / "figures"
    for stem in FINAL_FIGURE_STEMS:
        png = figures / f"{stem}.png"
        pdf = figures / f"{stem}.pdf"
        if not png.exists():
            fail(f"Missing final PNG figure: {png.relative_to(ROOT)}")
        if stem != "fig5_three_state_kalman_filter" and not pdf.exists():
            fail(f"Missing final PDF figure: {pdf.relative_to(ROOT)}")
    svg = figures / "fig5_three_state_kalman_filter.svg"
    if not svg.exists():
        fail(f"Missing Kalman schematic SVG: {svg.relative_to(ROOT)}")


def check_notebooks() -> None:
    notebooks = ROOT / "notebooks"
    for path in notebooks.glob("*.ipynb"):
        json.loads(path.read_text())

    main_text = (notebooks / "figures_main_analysis_record.ipynb").read_text()
    appendix_text = (notebooks / "figures_appendix_analysis_record.ipynb").read_text()

    for stem in MAIN_NOTEBOOK_STEMS:
        if stem not in main_text:
            fail(f"Main figures notebook does not explicitly cover {stem}")
    for stem in APPENDIX_NOTEBOOK_STEMS:
        if stem not in appendix_text:
            fail(f"Appendix figures notebook does not explicitly cover {stem}")

    helper = notebooks / "shared" / "plot_config.py"
    if not helper.exists():
        fail("Notebook-local shared/plot_config.py is missing")
    if (ROOT / "shared").exists():
        fail("Top-level shared/ should not exist; shared helpers belong under notebooks/")


def check_data() -> None:
    for exp, expected_n in EXPECTED_VALID_ROWS.items():
        path = ROOT / "data" / f"experiment{exp}" / f"E{exp}.pkl"
        df = pd.read_pickle(path)
        if "is_outlier" not in df.columns:
            fail(f"Experiment {exp} data missing is_outlier column")
        valid_rows = (~pd.Series(df["is_outlier"]).astype(bool)).sum()
        if valid_rows != expected_n:
            fail(f"Experiment {exp} valid row count mismatch: {valid_rows} != {expected_n}")
        subject_col = "subID" if "subID" in df.columns else ("Sub" if "Sub" in df.columns else "subject")
        n_subjects = pd.Series(df[subject_col]).nunique()
        if n_subjects != EXPECTED_SUBJECTS[exp]:
            fail(f"Experiment {exp} subject count mismatch: {n_subjects} != {EXPECTED_SUBJECTS[exp]}")


def check_model_outputs() -> None:
    ranking = pd.read_csv(ROOT / "results" / "kalman_model_fits" / "model_ranking.csv")
    for exp, expected_model in EXPECTED_WINNERS.items():
        top = ranking.loc[ranking["exp"].eq(exp)].sort_values("delta_AIC").iloc[0]
        if top["model_id"] != expected_model:
            fail(
                f"Experiment {exp} winning model mismatch: "
                f"{top['model_id']} != {expected_model}"
            )
        if int(top["n_subjects"]) != EXPECTED_SUBJECTS[exp]:
            fail(f"Experiment {exp} model subject count mismatch")

    fit_counts = {
        1: len(pd.read_csv(ROOT / "results" / "kalman_model_fits" / "model_fits_experiment1.csv")),
        2: len(pd.read_csv(ROOT / "results" / "kalman_model_fits" / "model_fits_experiment2.csv")),
    }
    for exp, n in fit_counts.items():
        expected = 135 * EXPECTED_SUBJECTS[exp]
        if n != expected:
            fail(f"Experiment {exp} fit count mismatch: {n} != {expected}")

    summary = (ROOT / "results" / "figure_source_data" / "kalman_results_summary.md").read_text()
    for exp, expected_model in EXPECTED_WINNERS.items():
        expected_line = f"Experiment {exp}: winner `{expected_model}`"
        if expected_line not in summary:
            fail(f"Kalman source-data summary missing: {expected_line}")

    parameter_summary = pd.read_csv(
        ROOT / "results" / "figure_source_data" / "kalman_winner_parameter_summary.csv",
        header=[0, 1],
        index_col=0,
    )
    for experiment in ["Experiment 1", "Experiment 2"]:
        if experiment not in parameter_summary.index:
            fail(f"Kalman parameter summary missing {experiment}")


def check_readme() -> None:
    text = (ROOT / "README.md").read_text()
    required_phrases = [
        "notebooks/shared/",
        "final manuscript and appendix figure coverage",
        "canonical exported figures remain in `figures/`",
    ]
    for phrase in required_phrases:
        if phrase not in text:
            fail(f"README missing expected phrase: {phrase}")


def main() -> int:
    checks = [
        check_figures,
        check_notebooks,
        check_data,
        check_model_outputs,
        check_readme,
    ]
    for check in checks:
        check()
        print(f"OK {check.__name__}")
    print("All public-release consistency checks passed.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001 - CLI should print concise failure
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
