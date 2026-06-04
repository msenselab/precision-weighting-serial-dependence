import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def load_behavior_results_module():
    path = ROOT / "code" / "02_behavior_results.py"
    spec = importlib.util.spec_from_file_location("behavior_results", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_result(*, llf: float, random_variance: float, intercept_se: float):
    return SimpleNamespace(
        converged=True,
        llf=llf,
        fe_params=pd.Series({"Intercept": 0.0}),
        bse_fe=pd.Series({"Intercept": intercept_se}),
        cov_re=pd.DataFrame([[random_variance]]),
    )


def test_fit_mixedlm_skips_degenerate_optimizer_result(monkeypatch) -> None:
    module = load_behavior_results_module()
    bad_kwargs = {
        "llf": np.inf,
        "random_variance": 0.0,
        "intercept_se": np.nan,
    }
    bad_result = fake_result(**bad_kwargs)
    results = {
        "lbfgs": bad_result,
        "powell": fake_result(llf=1.0, random_variance=0.1, intercept_se=0.2),
    }

    class FakeModel:
        def fit(self, *, method, **kwargs):
            return results[method]

    def fake_mixedlm(*args, **kwargs):
        return FakeModel()

    monkeypatch.setattr(module.smf, "mixedlm", fake_mixedlm)
    result, method = module.fit_mixedlm(
        pd.DataFrame({"subID": ["1"]}),
        "response ~ 1",
        methods=("lbfgs", "powell"),
    )

    assert method == "powell"
    assert result is results["powell"]
