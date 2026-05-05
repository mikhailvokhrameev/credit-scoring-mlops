import pandas as pd
import numpy as np
import mlflow.sklearn
from typing import Dict, Any
from optbinning import BinningProcess
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import VarianceThreshold
from src.models.base import BaseModel


class LogRegModel(BaseModel):
    """
    Baseline interpretable model serving as a standard scorecard.
    Pipeline: OptBinning (WoE + IV Selection) -> Variance Filter -> Scaler -> LogReg.
    """
    def fit(self, X: pd.DataFrame, y: pd.Series, eval_set=None) -> 'LogRegModel':
        self.features_ = list(X.columns) # Save features list
        
        selection_criteria = {
            "iv": {"min": 0.02, "max": 1.0}
        }
        
        binning_process = BinningProcess(
            variable_names=self.features_,
            max_n_prebins=50,
            min_prebin_size=0.01,
            min_bin_size=0.05,
            selection_criteria=selection_criteria
        )
        
        pipeline_steps =[
            ('binning', binning_process),
            ('variance_filter', VarianceThreshold(threshold=0.0)), # Filter out constant features after WoE transformation
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(
                class_weight=None,
                solver='saga',
                penalty=self.params.get('penalty', 'l2'),
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

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Extracts feature importance based on absolute values of Logistic Regression coefficients.
        Properly maps feature names through OptBinning and VarianceThreshold.
        """
        clf = self.model.named_steps['clf']
        selected_features = self.model[:-1].get_feature_names_out(self.features_)

        importance = np.abs(clf.coef_[0])

        fi_df = pd.DataFrame({"feature": selected_features, "importance": importance})
        return fi_df.sort_values("importance", ascending=False)

    def get_optuna_space(self, trial) -> Dict[str, Any]:
        return {
            "C": trial.suggest_float("C", 1e-4, 10.0, log=True),
            "penalty": trial.suggest_categorical("penalty", ["l1", "l2"])
        }