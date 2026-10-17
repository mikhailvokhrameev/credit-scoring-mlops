import matplotlib
import mlflow
import numpy as np
import pandas as pd
import pytest
import shap
import xgboost as xgb
from mlflow.tracking import MlflowClient

matplotlib.use("Agg")

from src.config import load_params
from src.evaluation.explainer import ModelExplainer, compare_local_contributions
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


def test_global_shap_plotting_error_propagates(synth, tmp_path, monkeypatch):
    seed = load_params()["seed"]
    X, y = features(synth), synth["TARGET"]
    model = LGBMModel(_params(LGBMModel, seed)).fit(X, y)
    explainer = ModelExplainer(model, X, seed=seed, output_dir=tmp_path)

    def boom(*args, **kwargs):
        raise RuntimeError("plot failed")

    monkeypatch.setattr(shap.plots, "beeswarm", boom)
    with pytest.raises(RuntimeError, match="plot failed"):
        explainer.compute_global_shap(X.head(N_EXPLAINED))


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
    assert explainer.output_space == "probability"
    assert explainer.explain_local_shap(X.iloc[0], save_path=tmp_path / "local.png").exists()


@pytest.mark.parametrize("cls", [LogRegModel, LGBMModel, XGBModel, CatBoostModel], ids=lambda c: c.__name__)
def test_local_explanation_is_additive_in_log_odds(cls, synth, tmp_path):
    seed = load_params()["seed"]
    X, y = features(synth), synth["TARGET"]
    model = cls(_params(cls, seed)).fit(X, y)
    explainer = ModelExplainer(model, X, seed=seed, output_dir=tmp_path)
    row = X.iloc[0]
    exp = explainer.local_explanation(row)
    p = model.predict_proba(pd.DataFrame([row]))[0]
    assert exp.values.shape == (X.shape[1],)
    assert explainer.output_space == "log_odds"
    assert exp.values.sum() + float(exp.base_values) == pytest.approx(np.log(p / (1 - p)), abs=1e-3)


def test_compare_local_contributions_aligns_by_feature_name(synth, tmp_path):
    seed = load_params()["seed"]
    X, y = features(synth), synth["TARGET"]
    assert "NOISE_0" in X.columns
    models = {
        "lgbm": LGBMModel(_params(LGBMModel, seed)).fit(X, y),
        "logreg": LogRegModel(_params(LogRegModel, seed)).fit(X.drop(columns="NOISE_0"), y),
    }
    top_n = X.shape[1]  # all features, so NOISE_0 is guaranteed to be in the result

    result = compare_local_contributions(models, X, X.iloc[0], seed=seed, top_n=top_n)

    assert list(result.columns) == ["lgbm", "logreg"]
    assert len(result) == top_n
    assert set(result.index) == set(X.columns)
    assert np.isnan(result.loc["NOISE_0", "logreg"])
    assert not np.isnan(result.loc["NOISE_0", "lgbm"])
    assert result.drop(index="NOISE_0").notna().all().all()
    strength = result.abs().max(axis=1).to_numpy()
    assert (np.diff(strength) <= 0).all()


def test_compare_local_contributions_truncates_to_top_n(synth):
    seed = load_params()["seed"]
    X, y = features(synth), synth["TARGET"]
    models = {"lgbm": LGBMModel(_params(LGBMModel, seed)).fit(X, y)}
    result = compare_local_contributions(models, X, X.iloc[0], seed=seed, top_n=5)
    assert len(result) == 5


def test_compare_local_contributions_rejects_probability_space(synth, monkeypatch):
    seed = load_params()["seed"]
    X, y = features(synth), synth["TARGET"]
    model = XGBModel(_params(XGBModel, seed)).fit(X, y)

    def probability_explainer(self, booster_model):
        self.output_space = "probability"
        background = self.X_train.head(20)
        return shap.Explainer(
            lambda d: booster_model.predict_proba(d)[:, 1], shap.maskers.Independent(background), seed=self.seed
        )

    monkeypatch.setattr(ModelExplainer, "_create_xgb_explainer", probability_explainer)
    with pytest.raises(ValueError, match="xgb"):
        compare_local_contributions({"xgb": model}, X, X.iloc[0], seed=seed, top_n=5)
