from typing import Any, Dict, Optional

import catboost as cb
import numpy as np
import pandas as pd

from src.config import load_params
from src.models.base import BaseModel


class CatBoostModel(BaseModel):
    """
    CatBoost implementation optimized for native categorical feature handling.
    Employs 'Balanced' auto class weights for imbalanced datasets.
    """
    n_estimators_key = "iterations"

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set=None,
        early_stopping_rounds: Optional[int] = None,
    ) -> 'CatBoostModel':
        X = self._sanitize_columns(X)
        self.features_ = X.columns.tolist()

        eval_data = None
        if eval_set is not None:
            eval_data = (self._sanitize_columns(eval_set[0])[self.features_], eval_set[1])
        use_early_stopping = eval_data is not None and early_stopping_rounds is not None

        cat_features = [col for col in X.columns if X[col].dtype.name in ['object', 'category']]

        fit_params = dict(self.params)

        device = fit_params.pop('device', 'cpu').upper()
        if device == 'GPU':
            fit_params['task_type'] = 'GPU'
            fit_params.setdefault('devices', '0')
        else:
            fit_params['task_type'] = 'CPU'

        fit_params.setdefault('auto_class_weights', 'Balanced')
        fit_params.setdefault('verbose', False)
        fit_params.setdefault('allow_writing_files', False)

        self.model = cb.CatBoostClassifier(**fit_params)
        self.model.fit(
            X, y,
            eval_set=eval_data,
            cat_features=cat_features,
            early_stopping_rounds=early_stopping_rounds if use_early_stopping else None,
        )
        self.best_iteration_ = int(self.model.get_best_iteration()) if use_early_stopping else None
        return self


    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X = self.transform(X)
        return self.model.predict_proba(X)[:, 1]


    def get_optuna_space(self, trial) -> Dict[str, Any]:
        return {
            "iterations": trial.suggest_int("iterations", 200, 2000),
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
            "border_count": trial.suggest_int("border_count", 32, 255),
            "eval_metric": "AUC",
            "random_seed": load_params()["seed"]
        }