import matplotlib
import mlflow
import numpy as np
import pytest
import shap
import xgboost as xgb
from mlflow.tracking import MlflowClient

matplotlib.use("Agg")

from src.config import load_params
from src.evaluation.explainer import ModelExplainer
from src.models.catboost_model import CatBoostModel
from src.models.lgbm_model import LGBMModel
from src.models.logreg_model import LogRegModel
from src.models.train import log_explanations
from src.models.xgb_model import XGBModel
from tests.conftest import features

N_TREES = 50
N_EXPLAINED = 50


def _params(cls, seed: int) -> dict:
    if cls is LogRegModel:
        return {"C": 1.0}
    if cls is LGBMModel:
        return {"n_estimators": N_TREES, "random_state": seed, "verbose": -1}
    if cls is XGBModel:
        return {"n_estimators": N_TREES, "random_state": seed}
    return {"iterations": N_TREES, "random_seed": seed, "thread_count": 1}


@pytest.mark.parametrize("cls", [LogRegModel, LGBMModel, XGBModel, CatBoostModel], ids=lambda c: c.__name__)
def test_global_shap_shape(cls, synth, tmp_path):
    seed = load_params()["seed"]
    X, y = features(synth), synth["TARGET"]
    model = cls(_params(cls, seed)).fit(X, y)

    explainer = ModelExplainer(model, X, seed=seed, output_dir=tmp_path)
    values = explainer.compute_global_shap(X.head(N_EXPLAINED))

    assert values.values.shape == (N_EXPLAINED, X.shape[1])


def test_log_explanations_xgb_logs_artifacts(synth, tmp_path):
    seed = load_params()["seed"]
    X, y = features(synth), synth["TARGET"]
    model = XGBModel(_params(XGBModel, seed)).fit(X, y)

    with mlflow.start_run() as run:
        log_explanations(model, X, tmp_path / "xai", seed)

    client = MlflowClient()
    global_files = {a.path for a in client.list_artifacts(run.info.run_id, "xai/global")}
    local_files = {a.path for a in client.list_artifacts(run.info.run_id, "xai/local")}
    assert {"xai/global/shap_summary_beeswarm.png", "xai/global/shap_summary_bar.png",
            "xai/global/shap_feature_importance.csv"} <= global_files
    assert len(local_files) == 3


def _fit_xgb(synth):
    seed = load_params()["seed"]
    X, y = features(synth), synth["TARGET"]
    return X, XGBModel(_params(XGBModel, seed)).fit(X, y), seed


def test_xgb_shap_is_exact_in_log_odds(synth, tmp_path):
    X, model, seed = _fit_xgb(synth)
    explainer = ModelExplainer(model, X, seed=seed, output_dir=tmp_path)
    values = explainer.compute_global_shap(X.head(N_EXPLAINED))

    margin = model.model.get_booster().predict(xgb.DMatrix(model.transform(X.head(N_EXPLAINED))), output_margin=True)
    np.testing.assert_allclose(values.values.sum(axis=1) + values.base_values, margin, atol=1e-3)
    assert not explainer.get_feature_importance().empty


def test_xgb_native_path_when_tree_explainer_fails(synth, tmp_path, monkeypatch, caplog):
    def broken(*args, **kwargs):
        raise ValueError("cannot parse booster")

    monkeypatch.setattr(shap, "TreeExplainer", broken)
    monkeypatch.setattr(shap, "Explainer", broken)  # the permutation fallback must not be reached
    X, model, seed = _fit_xgb(synth)
    with caplog.at_level("WARNING", logger="src.evaluation.explainer"):
        explainer = ModelExplainer(model, X, seed=seed, output_dir=tmp_path)
    assert "using native booster TreeSHAP" in caplog.text

    values = explainer.compute_global_shap(X.head(N_EXPLAINED))
    assert values.values.shape == (N_EXPLAINED, X.shape[1])
    assert (tmp_path / "shap_summary_beeswarm.png").exists()
    assert explainer.explain_local_shap(X.iloc[0], save_path=tmp_path / "local.png").exists()


def test_xgb_permutation_fallback_is_seeded_last_resort(synth, tmp_path, monkeypatch, caplog):
    import src.evaluation.explainer as module

    def broken(*args, **kwargs):
        raise ValueError("boom")

    monkeypatch.setattr(shap, "TreeExplainer", broken)
    monkeypatch.setattr(module, "_native_xgb_explainer", broken)
    X, model, seed = _fit_xgb(synth)
    with caplog.at_level("WARNING", logger="src.evaluation.explainer"):
        explainer = ModelExplainer(model, X, seed=seed, output_dir=tmp_path)
    assert "Native XGBoost TreeSHAP failed" in caplog.text
    assert explainer.explain_local_shap(X.iloc[0], save_path=tmp_path / "local.png").exists()
