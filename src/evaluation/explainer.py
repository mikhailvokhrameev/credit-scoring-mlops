import logging
import warnings

import shap

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline 
from pathlib import Path


logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


class ModelExplainer:
    """
    Universal SHAP explainer.

    Supports:
    - XGBoost
    - CatBoost
    - LightGBM
    - LogisticRegression
    - RandomForest
    - Any sklearn-compatible model
    """
    def __init__(self, model, X_train: pd.DataFrame, feature_names: list = None, max_background_samples: int = 1000, output_dir: Path = None):
        self.model = model
        self.output_dir = output_dir
        
        if self.output_dir is None:
            self.output_dir = Path("artifacts/xai")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if len(X_train) > max_background_samples:
            self.X_train = X_train.sample(max_background_samples, random_state=42)
        else:
            self.X_train = X_train.copy()

        self.feature_names = (feature_names or self.X_train.columns.tolist())

        self.shap_values = None

        logger.info(f"Initializing explainer for "f"{type(model).__name__}")

        self.explainer = self._create_explainer()
        
        if self.explainer is None:
            raise RuntimeError("SHAP explainer initialization failed.")
    
    
    def _unwrap_model(self, model):
        """Extract real estimator from wrappers or pipelines"""

        # Pipeline
        if isinstance(model, Pipeline):
            model = model.steps[-1][1]

        # Custom wrapper
        while hasattr(model, "model"):
            model = model.model

        return model


    def compute_global_shap(self, X_sample: pd.DataFrame):
        logger.info("Computing SHAP values...")
        try:
            vals = self.explainer(X_sample, check_additivity=False)
        except Exception:
            vals = self.explainer(X_sample)

        logger.info(f"Raw SHAP values shape: {vals.shape}")

        if len(vals.shape) == 3:
            if vals.shape[2] == 2:
                logger.info("Selecting class 1 from SHAP values.")
                self.shap_values = vals[:, :, 1]
            else:
                self.shap_values = vals[:, :, 0]
        else:
            self.shap_values = vals

        if hasattr(self.shap_values, "feature_names"):
             self.shap_values.feature_names = self.feature_names

        logger.info(f"Final SHAP values shape for plotting: {self.shap_values.shape}")
        return self.shap_values


    def _create_explainer(self):
        model = self.model

        logger.info(f"Raw model type: {type(model)}")

        while hasattr(model, "model"):
            model = model.model
            logger.info(f"Unwrapped wrapper -> {type(model).__name__}")

        if isinstance(model, Pipeline):
            logger.info("Detected Pipeline - using model.predict_proba directly")
            masker = shap.maskers.Independent(self.X_train)
            return shap.Explainer(model.predict_proba, masker)

        if isinstance(model, (XGBClassifier, LGBMClassifier, CatBoostClassifier)):
            logger.info(f"Using TreeExplainer for {type(model).__name__}")
            return shap.TreeExplainer(model)

        if isinstance(model, LogisticRegression):
            logger.info("Using Linear-style SHAP for LogisticRegression")
            masker = shap.maskers.Independent(self.X_train)
            return shap.Explainer(model.predict_proba, masker)

        raise ValueError(f"Unsupported model type: {type(model)}")
            

    def explain_local_shap(self, instance: pd.Series, save_path: str = None, max_display: int = 15) -> str:
        logger.info("Generating local SHAP explanation...")
        
        if save_path is None:
            save_path = self.output_dir / "local_shap_waterfall.png"
    
        instance_df = pd.DataFrame([instance])

        local_shap = self.explainer(instance_df)
        
        if len(local_shap.shape) == 3 and local_shap.shape[2] == 2:
            local_shap = local_shap[:, :, 1]
        elif len(local_shap.shape) == 3:
            local_shap = local_shap[:, :, 0]

        plt.figure(figsize=(12, 6))
        shap.plots.waterfall(local_shap[0], max_display=max_display, show=False)
        
        plt.title("Local SHAP Explanation")
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()
        return save_path


    def get_feature_importance(self):
        if self.shap_values is None:
            raise ValueError("SHAP values are not computed.")

        values = self.shap_values.values

        # Multiclass support
        if len(values.shape) == 3:
            values = np.mean(np.abs(values), axis=2)

        importance = np.abs(values).mean(axis=0)
        importance_df = pd.DataFrame({"feature": self.feature_names, "importance": importance})
        importance_df = (importance_df.sort_values(by="importance", ascending=False).reset_index(drop=True))

        return importance_df