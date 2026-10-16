import hashlib
import logging
from typing import Type

import mlflow
import optuna
import pandas as pd
from optuna.samplers import TPESampler
from optuna.trial import TrialState

from src.config import stable_hash
from src.models.base import BaseModel

logger = logging.getLogger(__name__)

CONFIG_SECTIONS = ("seed", "features", "hpo", "train")


def data_fingerprint(X: pd.DataFrame, y: pd.Series) -> str:
    """sha256 over the values of X, its column names and the values of y."""
    h = hashlib.sha256()
    h.update(pd.util.hash_pandas_object(X, index=False).values.tobytes())
    h.update(repr(list(X.columns)).encode("utf-8"))
    h.update(pd.util.hash_pandas_object(y, index=False).values.tobytes())
    return h.hexdigest()


def make_study_name(model_name: str, X: pd.DataFrame, y: pd.Series, params: dict) -> str:
    """Study name unique per model, training data and the config sections that shape HPO."""
    config_hash = stable_hash({k: params[k] for k in CONFIG_SECTIONS})
    return f"{model_name}_{data_fingerprint(X, y)[:8]}_{config_hash[:8]}"


class OptunaHPOTuner:
    """
    Unified Optuna Hyperparameter Optimization Tuner for all BaseModel instances.
    Integrates with SQLite for resume capability and optuna-dashboard.
    The study name is expected to be unique per data and config (see make_study_name),
    so an existing study with the same name is a resumable run of the same experiment.
    """
    def __init__(self, model_class: Type[BaseModel], storage: str, study_name: str, seed: int, base_params: dict):
        self.model_class = model_class
        self.storage = storage
        self.study_name = study_name
        self.base_params = base_params
        self.study = optuna.create_study(
            direction="maximize",
            study_name=self.study_name,
            storage=self.storage,
            load_if_exists=True,
            sampler=TPESampler(seed=seed),
        )

    def optimize(self, X: pd.DataFrame, y: pd.Series, n_trials: int, timeout: int, cv_kwargs: dict) -> dict:
        """
        Executes the optimization loop and returns the best params (usable directly as model params).

        Args:
            X (pd.DataFrame): Training features.
            y (pd.Series): Target variable.
            n_trials (int): Number of trials to run.
            timeout (int): Timeout in seconds.
            cv_kwargs (dict): Keyword args of BaseModel.cross_validate other than X, y.

        Raises:
            RuntimeError: if no trial completed.
        """
        logger.info(f"Starting HPO for {self.model_class.__name__}. Study: {self.study_name}")

        def objective(trial):
            with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}"):
                try:
                    params = self.model_class(dict(self.base_params)).get_optuna_space(trial) | self.base_params
                    trial.set_user_attr("params", params)
                    mlflow.log_params(params)
                    model = self.model_class(params)
                    return model.cross_validate(X, y, **cv_kwargs).mean_auc
                except Exception as e:
                    logger.exception(f"Trial {trial.number} failed")
                    trial.set_user_attr("error", f"{type(e).__name__}: {e}")
                    mlflow.set_tag("status", "failed")
                    raise

        self.study.optimize(objective, n_trials=n_trials, timeout=timeout, catch=(Exception,))

        if not any(t.state == TrialState.COMPLETE for t in self.study.trials):
            msg = next((t.user_attrs["error"] for t in self.study.trials if "error" in t.user_attrs), "unknown")
            raise RuntimeError(f"All {n_trials} HPO trials failed; first error: {msg}")

        if mlflow.active_run() is not None:
            self._log_best_results()

        best = self.study.best_trial.user_attrs.get("params", self.study.best_params)
        return best | self.base_params

    def _log_best_results(self):
        """Logs the best parameters and hyperparameter importance to the active MLflow run"""
        logger.info(f"Best Trial: {self.study.best_trial.number} | AUC: {self.study.best_value:.4f}")

        mlflow.log_params({f"best_{k}": v for k, v in self.study.best_params.items()})
        mlflow.log_metric("best_cv_auc", self.study.best_value)

        try:
            importances = optuna.importance.get_param_importances(self.study)
            mlflow.log_dict(importances, "param_importances.json")
        except Exception as e:
            logger.warning(f"Could not calculate param importances: {str(e)}")
