import lightgbm as lgb
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from src.config import load_params
from src.models.base import BaseModel

class LGBMModel(BaseModel):
    """
    LightGBM implementation for credit default risk prediction.
    Features modern Early Stopping callback to avoid deprecated arguments.
    """
    n_estimators_key = "n_estimators"

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set=None,
        early_stopping_rounds: Optional[int] = None,
    ) -> 'LGBMModel':
        X = self._sanitize_columns(X)
        self.features_ = list(X.columns)

        fit_kwargs = {}
        callbacks = []
        use_early_stopping = eval_set is not None and early_stopping_rounds is not None
        if eval_set is not None:
            fit_kwargs = {
                "eval_X": self._sanitize_columns(eval_set[0])[self.features_],
                "eval_y": eval_set[1],
            }
        if use_early_stopping:
            callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False))
            callbacks.append(lgb.log_evaluation(period=50))

        self.model = lgb.LGBMClassifier(**dict(self.params))
        self.model.fit(X, y, eval_metric="auc", callbacks=callbacks, **fit_kwargs)
        self.best_iteration_ = int(self.model.best_iteration_) if use_early_stopping else None

        return self


    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X = self.transform(X)
        return self.model.predict_proba(X)[:, 1]


    def get_optuna_space(self, trial) -> Dict[str, Any]:
        
        device = self.params.get('device', 'cpu')
        
        space = {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "device_type": "gpu" if device == "gpu" else "cpu",
            "n_estimators": trial.suggest_int("n_estimators", 200, 2000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 31, 255),
            "random_state": load_params()["seed"],
        }
        
        if device == "gpu":
            space.update({
                "gpu_platform_id": 0,
                "gpu_device_id": 0,
        })
        return space