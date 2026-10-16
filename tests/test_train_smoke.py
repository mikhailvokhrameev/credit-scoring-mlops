import json

import numpy as np
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

from src.config import load_params
from src.models.logreg_model import LogRegModel
from src.models.train import run_training

pytestmark = pytest.mark.smoke


def smoke_params(tmp_path) -> dict:
    p = load_params()
    p["mlflow"]["tracking_uri"] = f"sqlite:///{tmp_path}/mlflow.db"
    p["cv"]["n_splits"] = 3
    return p


def test_run_training_logreg_smoke(synthetic_data_dir, tmp_path):
    p = smoke_params(tmp_path)
    m = run_training("logreg", p, data_dir=synthetic_data_dir, artifact_dir=tmp_path / "art", trials=0)

    assert m["holdout_roc_auc"] >= p["gate"]["fixture_auc_min"]
    assert {"final_oof_roc_auc", "holdout_business_loss"} <= m.keys()
    assert "holdout_brier" in m
    assert 0.0 <= m["holdout_fairness_min_di"] <= 1.0
    assert {"holdout_fairness_gender_di_ratio", "holdout_fairness_gender_eo_gap",
            "holdout_fairness_age_di_ratio", "holdout_fairness_age_eo_gap"} <= m.keys()
    saved = json.loads((tmp_path / "art/logreg_model/metrics.json").read_text())
    assert saved["holdout_roc_auc"] == m["holdout_roc_auc"]
    assert (tmp_path / "art/logreg_model/model.joblib").exists()

    client = MlflowClient(p["mlflow"]["tracking_uri"])
    v = client.get_model_version_by_alias("credit_scoring_logreg", "challenger")
    assert v is not None
    run = client.get_run(v.run_id)
    assert run.data.tags["guardrails"] == "brier,fairness"
    artifacts = {a.path for a in client.list_artifacts(v.run_id, "guardrails")}
    assert {"guardrails/fairness_gender.csv", "guardrails/fairness_age.csv",
            "guardrails/reliability.csv", "guardrails/reliability.png"} <= artifacts


def test_run_training_base_flag(synthetic_data_dir, tmp_path):
    (synthetic_data_dir / "processed").rename(synthetic_data_dir / "processed_base")
    p = smoke_params(tmp_path)
    m = run_training(
        "logreg", p, data_dir=synthetic_data_dir, artifact_dir=tmp_path / "art", trials=0, base=True
    )
    assert "holdout_roc_auc" in m
    assert (tmp_path / "art/logreg_base_model/metrics.json").exists()
    client = MlflowClient(p["mlflow"]["tracking_uri"])
    assert client.get_model_version_by_alias("credit_scoring_logreg_base", "challenger") is not None


def test_run_training_requires_holdout(synthetic_data_dir, tmp_path):
    (synthetic_data_dir / "processed/holdout_features.parquet").unlink()
    with pytest.raises(FileNotFoundError, match="dvc repro"):
        run_training("logreg", load_params(), data_dir=synthetic_data_dir, artifact_dir=tmp_path, trials=0)


def test_run_training_rejects_overlapping_holdout(synthetic_data_dir, tmp_path):
    processed = synthetic_data_dir / "processed"
    train = pd.read_parquet(processed / "train_features.parquet")
    holdout = pd.read_parquet(processed / "holdout_features.parquet")
    holdout.loc[holdout.index[0], "SK_ID_CURR"] = train["SK_ID_CURR"].iloc[0]
    holdout.to_parquet(processed / "holdout_features.parquet", index=False)
    (processed / "holdout_ids.json").write_text(json.dumps([int(i) for i in holdout["SK_ID_CURR"]]))
    with pytest.raises(ValueError, match="overlap"):
        run_training(
            "logreg", smoke_params(tmp_path), data_dir=synthetic_data_dir,
            artifact_dir=tmp_path / "art", trials=0,
        )


def test_holdout_predicted_once(synthetic_data_dir, tmp_path, monkeypatch):
    processed = synthetic_data_dir / "processed"
    holdout = pd.read_parquet(processed / "holdout_features.parquet")
    selected = json.loads((processed / "selected_features.json").read_text())
    holdout_values = holdout[selected].to_numpy(dtype=np.float64)

    original = LogRegModel.predict_proba
    holdout_calls = []

    def counting(self, X):
        if list(X.columns)[: len(selected)] == selected and np.array_equal(
            X[selected].to_numpy(dtype=np.float64), holdout_values, equal_nan=True
        ):
            holdout_calls.append(len(X))
        return original(self, X)

    monkeypatch.setattr(LogRegModel, "predict_proba", counting)
    run_training(
        "logreg", smoke_params(tmp_path), data_dir=synthetic_data_dir,
        artifact_dir=tmp_path / "art", trials=0,
    )
    assert len(holdout_calls) == 1
