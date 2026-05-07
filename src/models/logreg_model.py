import pandas as pd
import numpy as np
import mlflow.sklearn
from typing import Dict, Any
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from src.models.base import BaseModel
from sklearn.impute import SimpleImputer
import logging

logger = logging.getLogger(__name__)


class LogRegModel(BaseModel):
    """
    Fast Baseline Logistic Regression.
    Pipeline: Imputer -> Scaler -> LogReg.
    """     
    def fit(self, X: pd.DataFrame, y: pd.Series, eval_set=None) -> 'LogRegModel':
        X = X.copy()
        numeric_cols = X.select_dtypes(include=['int64', 'int32', 'integer']).columns
        X[numeric_cols] = X[numeric_cols].astype('float64')
        
        self.features_ = list(X.columns) # Save features list
        
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
                random_state=42
            ))
        ]
        
        self.model = Pipeline(pipeline_steps)
        
        mlflow.sklearn.autolog(log_models=False) # Log params, metrics, learning time
        self.model.fit(X, y)
        
        return self
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]


    def get_optuna_space(self, trial) -> Dict[str, Any]:
        return {
            "C": trial.suggest_float("C", 1e-4, 10.0, log=True),
            "penalty": trial.suggest_categorical("penalty", ["l2"]),
            "device": "cpu"
        }
        
        