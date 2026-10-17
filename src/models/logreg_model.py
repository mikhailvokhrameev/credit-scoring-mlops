import logging
from typing import Any, Dict, Optional

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.base import BaseModel

logger = logging.getLogger(__name__)


class LogRegModel(BaseModel):
    """
    Fast Baseline Logistic Regression.
    Pipeline: Imputer -> Scaler -> LogReg.
    """
    n_estimators_key = None
    seed_key = "random_state"  # optional: without a seed in params the library default applies

    @staticmethod
    def _cast_ints(X: pd.DataFrame) -> pd.DataFrame:
        """Casts integer columns to float64 so fit and predict see identical dtypes."""
        int_cols = X.select_dtypes(include='integer').columns
        X[int_cols] = X[int_cols].astype('float64')
        return X

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self._cast_ints(super().transform(X).copy())

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set=None,
        early_stopping_rounds: Optional[int] = None,
    ) -> 'LogRegModel':
        self.best_iteration_ = None
        X = self._sanitize_columns(X)
        
        self.features_ = list(X.columns) # Save sanitized features list
        X = self._cast_ints(X)
        
        # Choose solver based on penalty
        penalty = self.params.get('penalty', 'l2')
        solver = 'saga' if penalty == 'l1' else 'lbfgs'
        
        pipeline_steps =[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(
                class_weight='balanced',
                solver=solver,
                penalty=penalty,
                C=self.params.get('C', 1.0),
                max_iter=1000,
                n_jobs=-1,
                random_state=self.params.get(self.seed_key)
            ))
        ]
        
        self.model = Pipeline(pipeline_steps)
        
        self.model.fit(X, y)
        
        return self
    
    
    def predict_proba(self, X):
        X = self.transform(X)
        return self.model.predict_proba(X)[:, 1]


    def get_optuna_space(self, trial) -> Dict[str, Any]:
        return {
            "C": trial.suggest_float("C", 1e-4, 10.0, log=True),
            "penalty": trial.suggest_categorical("penalty", ["l2"]),
            "device": "cpu"
        }