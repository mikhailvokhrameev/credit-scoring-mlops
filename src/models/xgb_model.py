from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import xgboost as xgb

from src.models.base import BaseModel


class XGBModel(BaseModel):
    """
    XGBoost implementation. Automatically utilizes CUDA device if available.
    Adjusts scale_pos_weight for imbalanced learning contexts.
    """
    n_estimators_key = "n_estimators"
    seed_key = "random_state"

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set=None,
        early_stopping_rounds: Optional[int] = None,
    ) -> 'XGBModel':
        X = self._sanitize_columns(X)
        self.features_ = list(X.columns)

        params = dict(self.params)

        # Calculate optimal class weight if not explicitly provided
        if 'scale_pos_weight' not in params:
            params['scale_pos_weight'] = (len(y) - y.sum()) / y.sum()

        params['tree_method'] = 'hist'
        params['device'] = 'cuda' if params.get('device', 'cpu') == 'gpu' else 'cpu'

        use_early_stopping = eval_set is not None and early_stopping_rounds is not None
        eval_data = None
        if eval_set is not None:
            eval_data = [(self._sanitize_columns(eval_set[0])[self.features_], eval_set[1])]

        self.model = xgb.XGBClassifier(
            **params,
            early_stopping_rounds=early_stopping_rounds if use_early_stopping else None,
        )
        self.model.fit(X, y, eval_set=eval_data, verbose=False)
        self.best_iteration_ = int(self.model.best_iteration) if use_early_stopping else None

        return self


    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X = self.transform(X)
        return self.model.predict_proba(X)[:, 1]


    def get_optuna_space(self, trial) -> Dict[str, Any]:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 2000),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "eta": trial.suggest_float("eta", 0.01, 0.3, log=True),
            "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 300),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "eval_metric": "auc",
        }