import logging
import warnings
import json

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
    - LogisticRegression (Fast LinearExplainer)
    - Any sklearn-compatible model
    """
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
            X_for_shap = self.preprocessor.transform(X_sample_transformed)
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

        logger.info(f"Raw SHAP values type: {type(vals)}")

        if isinstance(vals, shap.Explanation):
            logger.info(f"Raw SHAP values shape: {vals.shape}")
            if len(vals.shape) == 3:
                if vals.shape[2] == 2:
                    logger.info("Selecting class 1 from SHAP values.")
                    self.shap_values = vals[:, :, 1]
                else:
                    self.shap_values = vals[:, :, 0]
            else:
                self.shap_values = vals
        else:
            logger.info("SHAP returned raw arrays instead of Explanation object.")
            if isinstance(vals, list):
                sv = vals[1] if len(vals) > 1 else vals[0]
            else:
                if len(vals.shape) == 3:
                    sv = vals[:, :, 1] if vals.shape[2] == 2 else vals[:, :, 0]
                else:
                    sv = vals
            
            self.shap_values = shap.Explanation(
                values=sv,
                data=X_sample_transformed.values,
                feature_names=self.feature_names
            )

        if getattr(self, "is_linear_pipeline", False) and isinstance(self.shap_values, shap.Explanation):
            self.shap_values.data = X_sample_transformed.values
            
        if hasattr(self.shap_values, "feature_names"):
             self.shap_values.feature_names = self.feature_names

        logger.info(f"Final SHAP values shape for plotting: {self.shap_values.shape}")
        
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

        logger.info(f"Raw model type: {type(model)}")

        while hasattr(model, "model"):
            model = model.model
            logger.info(f"Unwrapped wrapper -> {type(model).__name__}")

        if isinstance(model, Pipeline):
            logger.info("Detected Pipeline")
            clf = model.steps[-1][1]
            if isinstance(clf, LogisticRegression):
                logger.info("Pipeline ends with LogisticRegression. Using ultra-fast LinearExplainer.")
                self.is_linear_pipeline = True
                self.preprocessor = Pipeline(model.steps[:-1])
                
                X_train_preprocessed = self.preprocessor.transform(self.X_train)
                masker = shap.maskers.Independent(X_train_preprocessed)
                return shap.LinearExplainer(clf, masker)
            else:
                logger.info("Using default Explainer for Pipeline (PermutationExplainer)")
                masker = shap.maskers.Independent(self.X_train)
                return shap.Explainer(model.predict_proba, masker)

        if isinstance(model, XGBClassifier):
            logger.info(f"Using TreeExplainer for {type(model).__name__}")
            booster = model.get_booster()
            
            old_save_config = booster.save_config
            def custom_save_config(*args, **kwargs):
                conf = json.loads(old_save_config(*args, **kwargs))
                if "learner" in conf and "learner_model_param" in conf["learner"]:
                    bs = conf["learner"]["learner_model_param"].get("base_score", "")
                    if isinstance(bs, str) and bs.startswith("["):
                        conf["learner"]["learner_model_param"]["base_score"] = bs.strip("[]")
                return json.dumps(conf)
                
            booster.save_config = custom_save_config
            return shap.TreeExplainer(booster)

        if isinstance(model, (LGBMClassifier, CatBoostClassifier)):
            logger.info(f"Using TreeExplainer for {type(model).__name__}")
            return shap.TreeExplainer(model)

        if isinstance(model, LogisticRegression):
            logger.info("Using LinearExplainer for LogisticRegression")
            masker = shap.maskers.Independent(self.X_train)
            return shap.LinearExplainer(model, masker)

        raise ValueError(f"Unsupported model type: {type(model)}")
            

    def explain_local_shap(self, instance: pd.Series, save_path: str = None, max_display: int = 15) -> str:
        logger.info("Generating local SHAP explanation...")
        
        if save_path is None:
            save_path = self.output_dir / "local_shap_waterfall.png"
    
        instance_df = pd.DataFrame([instance])

        if hasattr(self.model, "transform"):
            instance_df_transformed = self.model.transform(instance_df)
        else:
            instance_df_transformed = instance_df

        if getattr(self, "is_linear_pipeline", False):
            X_for_shap = self.preprocessor.transform(instance_df_transformed)
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
            if len(local_shap.shape) == 3 and local_shap.shape[2] == 2:
                local_shap = local_shap[:, :, 1]
            elif len(local_shap.shape) == 3:
                local_shap = local_shap[:, :, 0]
        else:
            if isinstance(local_shap, list):
                ls = local_shap[1] if len(local_shap) > 1 else local_shap[0]
            else:
                if len(local_shap.shape) == 3:
                    ls = local_shap[:, :, 1] if local_shap.shape[2] == 2 else local_shap[:, :, 0]
                else:
                    ls = local_shap
            
            local_shap = shap.Explanation(
                values=ls,
                data=instance_df_transformed.values,
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