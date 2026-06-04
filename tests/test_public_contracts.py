from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_cleaned_data_uses_sd_outlier_field() -> None:
    data_paths = [
        ROOT / "data" / "experiment1" / "E1.pkl",
        ROOT / "data" / "experiment2" / "E2.pkl",
    ]
    for path in data_paths:
        data = pd.read_pickle(path)

        assert "sd_outlier_final_sample" in data.columns
        assert "iqr_outlier_final_sample" not in data.columns
        sd_outlier = data["sd_outlier_final_sample"]
        assert not (sd_outlier & ~data["is_outlier"]).any()


def test_behavior_figure_source_data_has_correct_subject_counts() -> None:
    source_data = ROOT / "results" / "figure_source_data"
    path = source_data / "response_error_lag_lmm_parameters.csv"
    params = pd.read_csv(path)
    subject_counts = params.groupby("dataset")["n_subjects"].first().to_dict()

    assert subject_counts == {"experiment1": 22, "experiment2": 22}
    assert params[["beta", "se", "z", "p"]].notna().all().all()
    assert params["fit_method"].eq("powell").all()
    assert not (ROOT / "plotting" / "manuscript_ready_outputs").exists()


def test_preprocessing_threshold_filename_matches_sd_rule() -> None:
    preprocessing = ROOT / "results" / "preprocessing"
    thresholds = pd.read_csv(preprocessing / "final_sample_sd_thresholds.csv")

    assert "sd_error" in thresholds.columns
    assert not (preprocessing / "final_sample_iqr_thresholds.csv").exists()
