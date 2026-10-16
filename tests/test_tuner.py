import copy

import mlflow
import pytest

from src.config import load_params
from src.models.logreg_model import LogRegModel
from src.models.tuner import OptunaHPOTuner, make_study_name
from tests.conftest import features

CV_KWARGS = dict(n_splits=2, seed=42, early_stopping_rounds=10, es_fraction=0.1)


def test_study_name_changes_with_data_and_config(synth):
    X, y = features(synth), synth["TARGET"]
    p = load_params()
    n1 = make_study_name("lgbm", X, y, p)
    assert make_study_name("lgbm", X.iloc[:-1], y.iloc[:-1], p) != n1
    p2 = copy.deepcopy(p)
    p2["features"]["top_n"] = 200
    assert make_study_name("lgbm", X, y, p2) != n1
    assert make_study_name("lgbm", X, y, p) == n1


def test_all_trials_failed_raises_with_cause(tmp_path, synth, monkeypatch):
    X, y = features(synth), synth["TARGET"]
    monkeypatch.setattr(
        LogRegModel,
        "cross_validate",
        lambda *a, **k: (_ for _ in ()).throw(MemoryError("boom-oom")),
    )
    t = OptunaHPOTuner(LogRegModel, f"sqlite:///{tmp_path}/o.db", "s", 42, {})
    with pytest.raises(RuntimeError, match="boom-oom"):
        t.optimize(X, y, n_trials=2, timeout=60, cv_kwargs=CV_KWARGS)


def test_optimize_returns_best_with_base_params(tmp_path, synth):
    X, y = features(synth), synth["TARGET"]
    t = OptunaHPOTuner(LogRegModel, f"sqlite:///{tmp_path}/o.db", "s", 42, {"device": "cpu"})
    result = t.optimize(X, y, n_trials=2, timeout=120, cv_kwargs=CV_KWARGS)
    assert "C" in result
    assert result["device"] == "cpu"
    assert mlflow.active_run() is None
