import xgboost as xgb
import pandas as pd
import numpy as np
import mlflow.xgboost
import re
from typing import Dict, Any
from src.models.base import BaseModel


class XGBModel(BaseModel):
    """
    XGBoost implementation. Automatically utilizes CUDA device if available.
    Adjusts scale_pos_weight for imbalanced learning contexts.
    """
    def _sanitize_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        """Removes special characters from column names to comply with XGBoost constraints"""
        X = X.copy()
        X.columns = [
            re.sub(r"[^0-9a-zA-Z_]+", "_", str(c))
            for c in X.columns
        ]
        return X


    def _align_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Ensures inference data has the exact same columns as training data"""
        X = X.copy()

        if not hasattr(self, "features_"):
            raise ValueError("Model is not fitted yet (features_ not found)")

        missing = set(self.features_) - set(X.columns)
        for col in missing:
            X[col] = 0

        return X[self.features_]


    def fit(self, X: pd.DataFrame, y: pd.Series, eval_set = None) -> 'XGBModel':
        # Sanitize features
        X = self._sanitize_columns(X)
        
        if eval_set is not None:
            eval_set = (
                self._sanitize_columns(eval_set[0]),
                eval_set[1]
            )

        self.features_ = list(X.columns)
        
        # Calculate optimal class weight if not explicitly provided
        if 'scale_pos_weight' not in self.params:
            pos_ratio = (len(y) - y.sum()) / y.sum()
            self.params['scale_pos_weight'] = pos_ratio
            
        device = self.params.get('device', 'cpu')
        
        if device == 'gpu':
            self.params['tree_method'] = 'hist'
            self.params['device'] = 'cuda'
        else:
            self.params['tree_method'] = 'hist'
            self.params['device'] = 'cpu'

        self.model = xgb.XGBClassifier(**self.params, early_stopping_rounds=100 if eval_set else None)
        
        eval_data = [(eval_set[0], eval_set[1])] if eval_set else None
        
        mlflow.xgboost.autolog()
        self.model.fit(X, y, eval_set=eval_data, verbose=False)
        
        return self
    

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        # Sanitize and align internally before predicting
        X = self._sanitize_columns(X)
        X = self._align_features(X)
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
            "random_state": 42
        }