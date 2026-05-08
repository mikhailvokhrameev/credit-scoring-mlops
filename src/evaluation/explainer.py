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
    def __init__(self, model, X_train: pd.DataFrame, feature_names: list = None, max_background_samples: int = 1000, output_dir: Path = None):
        self.model = model
        self.output_dir = output_dir
        self.is_linear_pipeline = False
        self.preprocessor = None
        
        if self.output_dir is None:
            self.output_dir = Path("artifacts/xai")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.feature_names = (feature_names or X_train.columns.tolist())

        if hasattr(self.model, "transform"):
            X_train_transformed = self.model.transform(X_train)
        else:
            X_train_transformed = X_train

        if len(X_train_transformed) > max_background_samples:
            self.X_train = X_train_transformed.sample(max_background_samples, random_state=42)
        else:
            self.X_train = X_train_transformed.copy()

        self.shap_values = None

        logger.info(f"Initializing explainer for {type(model).__name__}")

        self.explainer = self._create_explainer()
        
        if self.explainer is None:
            raise RuntimeError("SHAP explainer initialization failed.")
    

    def compute_global_shap(self, X_sample: pd.DataFrame):
        logger.info("Computing SHAP values...")
        
        if hasattr(self.model, "transform"):
            X_sample_transformed = self.model.transform(X_sample)
        else:
            X_sample_transformed = X_sample

        if getattr(self, "is_linear_pipeline", False):
            X_for_shap = pd.DataFrame(
                self.preprocessor.transform(X_sample_transformed), 
                columns=X_sample_transformed.columns
            )
        else:
            X_for_shap = X_sample_transformed

        max_evals = X_sample_transformed.shape[1] * 2 + 100

        try:
            vals = self.explainer(X_for_shap, check_additivity=False)
        except TypeError:
            try:
                vals = self.explainer(X_for_shap, max_evals=max_evals)
            except Exception:
                vals = self.explainer(X_for_shap)
        except Exception:
            try:
                vals = self.explainer(X_for_shap, max_evals=max_evals)
            except Exception:
                vals = self.explainer(X_for_shap)

        if isinstance(vals, shap.Explanation):
            if len(vals.shape) == 3:
                self.shap_values = vals[:, :, 1] if vals.shape[2] == 2 else vals[:, :, 0]
            else:
                self.shap_values = vals
        else:
            sv = vals[1] if isinstance(vals, list) and len(vals) > 1 else (vals[0] if isinstance(vals, list) else vals)
            if len(sv.shape) == 3:
                sv = sv[:, :, 1] if sv.shape[2] == 2 else sv[:, :, 0]
            
            self.shap_values = shap.Explanation(
                values=sv,
                data=X_for_shap.values,
                feature_names=self.feature_names
            )

        if getattr(self, "is_linear_pipeline", False) and isinstance(self.shap_values, shap.Explanation):
            self.shap_values.data = X_sample_transformed.values
            
        if hasattr(self.shap_values, "feature_names"):
             self.shap_values.feature_names = self.feature_names
        
        try:
            shap_beeswarm_path = self.output_dir / "shap_summary_beeswarm.png"
            plt.figure(figsize=(10, 6))
            shap.plots.beeswarm(self.shap_values, max_display=15, show=False)
            plt.tight_layout()
            plt.savefig(shap_beeswarm_path, dpi=200, bbox_inches="tight")
            plt.close()

            shap_bar_path = self.output_dir / "shap_summary_bar.png"
            plt.figure(figsize=(10, 6))
            shap.plots.bar(self.shap_values, max_display=15, show=False)
            plt.tight_layout()
            plt.savefig(shap_bar_path, dpi=200, bbox_inches="tight")
            plt.close()
        except Exception as e:
            logger.error(f"Error during global SHAP plotting: {e}")

        return self.shap_values


    def _create_explainer(self):
        model = self.model

        while hasattr(model, "model"):
            model = model.model
            logger.info(f"Unwrapped wrapper -> {type(model).__name__}")

        if isinstance(model, Pipeline) or type(model).__name__ == "Pipeline":
            clf = model.steps[-1][1]
            clf_name = type(clf).__name__
            
            if "LogisticRegression" in clf_name or "Linear" in clf_name:
                logger.info("Pipeline ends with LogisticRegression. Extracting LinearExplainer for ultra-fast processing.")
                self.is_linear_pipeline = True
                self.preprocessor = Pipeline(model.steps[:-1])
                
                X_train_preprocessed = pd.DataFrame(
                    self.preprocessor.transform(self.X_train), 
                    columns=self.X_train.columns
                )
                masker = shap.maskers.Independent(X_train_preprocessed)
                return shap.LinearExplainer(clf, masker)
            else:
                logger.info("Using default PermutationExplainer for Pipeline")
                masker = shap.maskers.Independent(self.X_train)
                return shap.Explainer(model.predict_proba, masker)

        if isinstance(model, XGBClassifier):
            logger.info(f"Using TreeExplainer for {type(model).__name__}")
            return shap.TreeExplainer(model)

        if isinstance(model, (LGBMClassifier, CatBoostClassifier)):
            logger.info(f"Using TreeExplainer for {type(model).__name__}")
            return shap.TreeExplainer(model)

        if isinstance(model, LogisticRegression):
            masker = shap.maskers.Independent(self.X_train)
            return shap.LinearExplainer(model, masker)

        return shap.TreeExplainer(model)
            

    def explain_local_shap(self, instance: pd.Series, save_path: str = None, max_display: int = 15) -> str:
        if save_path is None:
            save_path = self.output_dir / "local_shap_waterfall.png"
    
        instance_df = pd.DataFrame([instance])

        if hasattr(self.model, "transform"):
            instance_df_transformed = self.model.transform(instance_df)
        else:
            instance_df_transformed = instance_df

        if getattr(self, "is_linear_pipeline", False):
            X_for_shap = pd.DataFrame(
                self.preprocessor.transform(instance_df_transformed), 
                columns=instance_df_transformed.columns
            )
        else:
            X_for_shap = instance_df_transformed

        max_evals = instance_df_transformed.shape[1] * 2 + 100
        
        try:
            local_shap = self.explainer(X_for_shap)
        except TypeError:
            try:
                local_shap = self.explainer(X_for_shap, max_evals=max_evals)
            except Exception:
                local_shap = self.explainer(X_for_shap)
        except Exception:
            try:
                local_shap = self.explainer(X_for_shap, max_evals=max_evals)
            except Exception:
                local_shap = self.explainer(X_for_shap)

        if isinstance(local_shap, shap.Explanation):
            if len(local_shap.shape) == 3:
                local_shap = local_shap[:, :, 1] if local_shap.shape[2] == 2 else local_shap[:, :, 0]
        else:
            ls = local_shap[1] if isinstance(local_shap, list) and len(local_shap) > 1 else (local_shap[0] if isinstance(local_shap, list) else local_shap)
            if len(ls.shape) == 3:
                ls = ls[:, :, 1] if ls.shape[2] == 2 else ls[:, :, 0]
            
            local_shap = shap.Explanation(
                values=ls,
                data=X_for_shap.values,
                feature_names=self.feature_names
            )

        if getattr(self, "is_linear_pipeline", False) and isinstance(local_shap, shap.Explanation):
            local_shap.data = instance_df_transformed.values

        if hasattr(local_shap, "feature_names"):
            local_shap.feature_names = self.feature_names

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

        if len(values.shape) == 3:
            values = np.mean(np.abs(values), axis=2)

        importance = np.abs(values).mean(axis=0)
        importance_df = pd.DataFrame({"feature": self.feature_names, "importance": importance})
        importance_df = (importance_df.sort_values(by="importance", ascending=False).reset_index(drop=True))

        return importance_df