import catboost as cb
import pandas as pd
import numpy as np
import re
from typing import Dict, Any
from src.models.base import BaseModel


class CatBoostModel(BaseModel):
    """
    CatBoost implementation optimized for native categorical feature handling.
    Employs 'Balanced' auto class weights for imbalanced datasets.
    """
    def _sanitize_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        """Removes special characters from column names"""
        X = X.copy()
        X.columns = [
            re.sub(r"[^0-9a-zA-Z_]+", "_", str(c))
            for c in X.columns
        ]
        return X


    def _align_features(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        if not hasattr(self, "features_"):
            raise ValueError("Model is not fitted yet")

        # Add missing columns
        missing = set(self.features_) - set(X.columns)
        for col in missing:
            X[col] = 0

        # Enforce exact order
        return X.reindex(columns=self.features_)
    
    
    def transform(self, X):
        X = self._sanitize_columns(X)
        X = self._align_features(X)
        return X


    def fit(self, X: pd.DataFrame, y: pd.Series, eval_set=None) -> 'CatBoostModel':
        X = self._sanitize_columns(X)

        if eval_set is not None:
            eval_set = (
                self._sanitize_columns(eval_set[0]),
                eval_set[1]
            )

        self.features_ = X.columns.tolist()

        cat_features = [col for col in X.columns if X[col].dtype.name in ['object', 'category']]

        fit_params = self.params.copy()

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

        eval_data = (eval_set[0], eval_set[1]) if eval_set else None

        self.model.fit(
            X, y,
            eval_set=eval_data,
            cat_features=cat_features,
            early_stopping_rounds=100 if eval_set else None
        )
        return self


    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X = self._sanitize_columns(X)
        X = self._align_features(X)
        return self.model.predict_proba(X)[:, 1]


    def get_optuna_space(self, trial) -> Dict[str, Any]:
        return {
            "iterations": trial.suggest_int("iterations", 200, 2000),
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
            "border_count": trial.suggest_int("border_count", 32, 255),
            "eval_metric": "AUC",
            "random_seed": 42
        }