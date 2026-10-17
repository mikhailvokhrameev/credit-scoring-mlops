"""The run's seed (params["seed"]) must reach the fitted estimators, with and without HPO."""
import copy

import joblib
import pytest
import yaml

from src.config import ROOT, load_params
from src.models.base import CVResult
from src.models.catboost_model import CatBoostModel
from src.models.lgbm_model import LGBMModel
from src.models.logreg_model import LogRegModel
from src.models.train import MODEL_REGISTRY, run_training
from src.models.tuner import OptunaHPOTuner
from tests.conftest import features

SEEDS = (7, 13)


def run_params(tmp_path, seed: int) -> dict:
    p = copy.deepcopy(load_params())
    p["seed"] = seed
    p["mlflow"]["tracking_uri"] = f"sqlite:///{tmp_path}/mlflow.db"
    p["cv"]["n_splits"] = 3
    return p


def final_estimator_seed(artifact_dir, model_name: str):
    wrapper = joblib.load(artifact_dir / f"{model_name}_model" / "model.joblib")
    estimator = wrapper.model.named_steps["clf"] if model_name == "logreg" else wrapper.model
    key = MODEL_REGISTRY[model_name].seed_key
    return estimator.get_params()[key]


def test_seed_keys():
    assert LGBMModel.seed_key == "random_state"
    assert MODEL_REGISTRY["xgb"].seed_key == "random_state"
    assert LogRegModel.seed_key == "random_state"
    assert CatBoostModel.seed_key == "random_seed"


@pytest.mark.parametrize("model_name", ["logreg", "lgbm", "xgb", "catboost"])
@pytest.mark.parametrize("seed", SEEDS)
def test_run_seed_reaches_final_model_without_hpo(model_name, seed, synthetic_data_dir, tmp_path):
    run_training(
        model_name, run_params(tmp_path, seed), data_dir=synthetic_data_dir,
        artifact_dir=tmp_path / "art", trials=0, register=False,
    )
    assert final_estimator_seed(tmp_path / "art", model_name) == seed


def test_alt_params_file_seed_reaches_model(synthetic_data_dir, tmp_path):
    alt = yaml.safe_load((ROOT / "params.yaml").read_text())
    alt["seed"] = 99
    alt["mlflow"]["tracking_uri"] = f"sqlite:///{tmp_path}/mlflow.db"
    alt["cv"]["n_splits"] = 3
    f = tmp_path / "alt.yaml"
    f.write_text(yaml.safe_dump(alt))
    run_training(
        "lgbm", load_params(f), data_dir=synthetic_data_dir,
        artifact_dir=tmp_path / "art", trials=0, register=False,
    )
    assert final_estimator_seed(tmp_path / "art", "lgbm") == 99


def test_tuner_keeps_base_seed_over_search_space(tmp_path, synth, monkeypatch):
    X, y = features(synth), synth["TARGET"]

    def fake_cv(self, X, y, **kwargs):
        return CVResult(oof=None, models=[], fold_auc=[0.5], best_iterations=[])

    monkeypatch.setattr(LGBMModel, "cross_validate", fake_cv)
    t = OptunaHPOTuner(LGBMModel, f"sqlite:///{tmp_path}/o.db", "s", 7, {"random_state": 7})
    cv_kwargs = dict(n_splits=2, seed=7, early_stopping_rounds=10, es_fraction=0.1)
    assert t.optimize(X, y, n_trials=1, timeout=60, cv_kwargs=cv_kwargs)["random_state"] == 7


def test_logreg_without_seed_uses_library_default(synth):
    X, y = features(synth), synth["TARGET"]
    m = LogRegModel({"C": 1.0}).fit(X, y)
    assert m.model.named_steps["clf"].random_state is None
