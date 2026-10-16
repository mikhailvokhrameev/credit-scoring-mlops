import json

import numpy as np
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

from src.config import load_params
from src.models.train import run_training
from src.monitoring.drift import drift_report, psi


def test_psi_identical_is_zero():
    x = np.random.default_rng(0).normal(size=5000)
    assert psi(x, x) == pytest.approx(0.0, abs=1e-9)


def test_psi_shift_detected():
    r = np.random.default_rng(0)
    assert psi(r.normal(size=5000), r.normal(1.0, size=5000)) > 0.25


def test_psi_nan_and_constant():
    e = np.array([1.0] * 100 + [np.nan] * 10)
    a = np.array([1.0] * 50 + [np.nan] * 60)
    v = psi(e, a)
    assert np.isfinite(v) and v >= 0


def test_psi_nan_share_shift_is_detected():
    e = np.array([1.0] * 100 + [np.nan] * 10)
    a = np.array([1.0] * 10 + [np.nan] * 100)
    assert psi(e, a) > 0.25


def test_psi_empty_expected_does_not_crash():
    v = psi(np.full(20, np.nan), np.array([1.0, 2.0, np.nan]))
    assert np.isfinite(v) and v >= 0


def test_psi_actual_outside_expected_range_is_finite():
    e = np.linspace(0, 1, 1000)
    v = psi(e, e + 100)
    assert np.isfinite(v) and v > 0.25


def test_drift_report_sorted_desc():
    r = np.random.default_rng(1)
    ref = pd.DataFrame({"a": r.normal(size=2000), "b": r.normal(size=2000), "c": r.normal(size=2000)})
    cur = pd.DataFrame({"a": r.normal(size=2000), "b": r.normal(2.0, size=2000), "c": r.normal(0.5, size=2000)})
    rep = drift_report(ref, cur, ["a", "b", "c"])
    assert list(rep.columns) == ["feature", "psi"]
    assert rep["feature"].tolist() == ["b", "c", "a"]
    assert rep["psi"].is_monotonic_decreasing


def test_drift_report_missing_column_raises():
    ref = pd.DataFrame({"a": [1.0, 2.0]})
    with pytest.raises(ValueError, match="b"):
        drift_report(ref, ref, ["a", "b"])


def _params(tmp_path) -> dict:
    p = load_params()
    p["mlflow"]["tracking_uri"] = f"sqlite:///{tmp_path}/mlflow.db"
    p["cv"]["n_splits"] = 3
    return p


def test_training_logs_psi_and_report(synthetic_data_dir, tmp_path):
    p = _params(tmp_path)
    m = run_training("logreg", p, data_dir=synthetic_data_dir, artifact_dir=tmp_path / "art", trials=0)
    assert m["psi_score"] >= 0 and m["psi_max_feature"] >= 0
    assert json.loads((tmp_path / "art/logreg_model/metrics.json").read_text())["psi_score"] == m["psi_score"]

    client = MlflowClient(p["mlflow"]["tracking_uri"])
    v = client.get_model_version_by_alias("credit_scoring_logreg", "challenger")
    run = client.get_run(v.run_id)
    assert run.data.metrics["psi_score"] == pytest.approx(m["psi_score"])
    assert run.data.tags["drift_alert"] == str(m["psi_score"] > p["monitoring"]["psi_alert"]).lower()
    assert "monitoring/drift_report.csv" in {a.path for a in client.list_artifacts(v.run_id, "monitoring")}


def test_training_flags_drift_alert_without_failing(synthetic_data_dir, tmp_path, caplog):
    path = synthetic_data_dir / "processed/test_features.parquet"
    test = pd.read_parquet(path)
    selected = json.loads((synthetic_data_dir / "processed/selected_features.json").read_text())
    test[selected] = test[selected] + 100.0
    test.to_parquet(path, index=False)

    p = _params(tmp_path)
    with caplog.at_level("WARNING"):
        m = run_training("logreg", p, data_dir=synthetic_data_dir, artifact_dir=tmp_path / "art", trials=0)
    assert m["psi_max_feature"] > p["monitoring"]["psi_alert"]
    assert m["psi_score"] > p["monitoring"]["psi_alert"]
    client = MlflowClient(p["mlflow"]["tracking_uri"])
    run = client.get_run(client.get_model_version_by_alias("credit_scoring_logreg", "challenger").run_id)
    assert run.data.tags["drift_alert"] == "true"
    assert any(r.levelname == "WARNING" and "drift" in r.getMessage().lower() for r in caplog.records)


def test_training_requires_test_features(synthetic_data_dir, tmp_path):
    (synthetic_data_dir / "processed/test_features.parquet").unlink()
    with pytest.raises(FileNotFoundError, match="dvc repro features"):
        run_training("logreg", _params(tmp_path), data_dir=synthetic_data_dir,
                     artifact_dir=tmp_path / "art", trials=0)


def test_training_rejects_test_missing_selected_feature(synthetic_data_dir, tmp_path):
    path = synthetic_data_dir / "processed/test_features.parquet"
    test = pd.read_parquet(path)
    selected = json.loads((synthetic_data_dir / "processed/selected_features.json").read_text())
    test.drop(columns=selected[0]).to_parquet(path, index=False)
    with pytest.raises(ValueError, match=selected[0]):
        run_training("logreg", _params(tmp_path), data_dir=synthetic_data_dir,
                     artifact_dir=tmp_path / "art", trials=0)


def test_training_base_pipeline_uses_selected_order(synthetic_data_dir, tmp_path, caplog):
    processed = synthetic_data_dir / "processed"
    (processed / "feature_importance.csv").unlink()
    processed.rename(synthetic_data_dir / "processed_base")
    with caplog.at_level("INFO"):
        m = run_training("logreg", _params(tmp_path), data_dir=synthetic_data_dir,
                         artifact_dir=tmp_path / "art", trials=0, base=True)
    assert "psi_score" in m and "psi_max_feature" in m
    assert "feature_importance.csv" in caplog.text
