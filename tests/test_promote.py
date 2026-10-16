import logging
import sys

import mlflow
import pytest
from mlflow.tracking import MlflowClient

from src.models import promote as promote_module
from src.models.promote import promote

NAME = "credit_scoring_xgb"
PARAMS = {
    "mlflow": {},  # set per test from the isolated tracking URI
    "fairness": {"di_min": 0.8},
    "gate": {"auc_epsilon": 0.005},
}


@pytest.fixture
def params():
    return {**PARAMS, "mlflow": {"tracking_uri": mlflow.get_tracking_uri()}}


@pytest.fixture
def client(params):
    return MlflowClient(params["mlflow"]["tracking_uri"])


def _register(client: MlflowClient, auc: float, min_di: float | None, alias: str | None = None) -> str:
    """Create a run with holdout metrics, register it as a new model version, optionally alias it."""
    metrics = {}
    if auc is not None:
        metrics["holdout_roc_auc"] = auc
    if min_di is not None:
        metrics["holdout_fairness_min_di"] = min_di
    with mlflow.start_run() as run:
        mlflow.log_metrics(metrics)
    try:
        client.create_registered_model(NAME)
    except mlflow.exceptions.MlflowException:
        pass  # already registered by an earlier version
    version = client.create_model_version(NAME, source=f"runs:/{run.info.run_id}/model", run_id=run.info.run_id)
    if alias:
        client.set_registered_model_alias(NAME, alias, version.version)
    return str(version.version)


def _champion(client: MlflowClient) -> str | None:
    try:
        return str(client.get_model_version_by_alias(NAME, "champion").version)
    except mlflow.exceptions.MlflowException:
        return None


def test_first_challenger_becomes_champion(client, params):
    v = _register(client, auc=0.75, min_di=0.9, alias="challenger")
    assert promote(NAME, params, client) is True
    assert _champion(client) == v


def test_worse_challenger_rejected(client, params, caplog):
    champion = _register(client, auc=0.78, min_di=0.9, alias="champion")
    _register(client, auc=0.77, min_di=0.9, alias="challenger")
    with caplog.at_level(logging.WARNING):
        assert promote(NAME, params, client) is False
    assert _champion(client) == champion
    assert "AUC" in caplog.text


def test_unfair_challenger_rejected(client, params):
    champion = _register(client, auc=0.75, min_di=0.9, alias="champion")
    _register(client, auc=0.80, min_di=0.6, alias="challenger")
    assert promote(NAME, params, client) is False
    assert _champion(client) == champion


def test_challenger_within_epsilon_promoted(client, params):
    _register(client, auc=0.780, min_di=0.9, alias="champion")
    v = _register(client, auc=0.776, min_di=0.9, alias="challenger")
    assert promote(NAME, params, client) is True
    assert _champion(client) == v


def test_nan_min_di_fails_closed(client, params, caplog):
    _register(client, auc=0.80, min_di=float("nan"), alias="challenger")
    with caplog.at_level(logging.WARNING):
        assert promote(NAME, params, client) is False
    assert _champion(client) is None
    assert "holdout_fairness_min_di" in caplog.text


def test_missing_auc_fails_closed(client, params, caplog):
    _register(client, auc=None, min_di=0.9, alias="challenger")
    with caplog.at_level(logging.WARNING):
        assert promote(NAME, params, client) is False
    assert "holdout_roc_auc" in caplog.text


def test_non_finite_champion_auc_fails_closed(client, params):
    champion = _register(client, auc=float("nan"), min_di=0.9, alias="champion")
    _register(client, auc=0.80, min_di=0.9, alias="challenger")
    assert promote(NAME, params, client) is False
    assert _champion(client) == champion


def test_no_challenger_returns_false(client, params, caplog):
    _register(client, auc=0.8, min_di=0.9)
    with caplog.at_level(logging.WARNING):
        assert promote(NAME, params, client) is False
    assert "challenger" in caplog.text


def test_challenger_already_champion_is_noop(client, params):
    v = _register(client, auc=0.8, min_di=0.9, alias="challenger")
    client.set_registered_model_alias(NAME, "champion", v)
    assert promote(NAME, params, client) is True
    assert _champion(client) == v


def test_default_client_uses_tracking_uri_from_params(client, params):
    v = _register(client, auc=0.75, min_di=0.9, alias="challenger")
    assert promote(NAME, params) is True
    assert _champion(client) == v


@pytest.mark.parametrize("min_di, expected_code", [(0.9, None), (0.5, 1)])
def test_cli_exit_code(client, params, monkeypatch, min_di, expected_code):
    _register(client, auc=0.75, min_di=min_di, alias="challenger")
    monkeypatch.setattr(promote_module, "load_params", lambda path=None: params)
    monkeypatch.setattr(sys, "argv", ["promote", "--model", "xgb"])
    if expected_code is None:
        promote_module.main()  # exit code 0: returns normally
        assert _champion(client) == "1"
    else:
        with pytest.raises(SystemExit) as exc:
            promote_module.main()
        assert exc.value.code == expected_code
        assert _champion(client) is None


def test_cli_base_uses_base_registered_name(client, params, monkeypatch):
    seen = {}
    monkeypatch.setattr(promote_module, "load_params", lambda path=None: params)
    monkeypatch.setattr(promote_module, "promote", lambda name, p, c=None: seen.setdefault("name", name) and True)
    monkeypatch.setattr(sys, "argv", ["promote", "--model", "lgbm", "--base"])
    promote_module.main()
    assert seen["name"] == "credit_scoring_lgbm_base"
