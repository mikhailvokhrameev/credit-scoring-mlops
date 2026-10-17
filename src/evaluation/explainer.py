import logging
import tempfile
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.models.base import BaseModel

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


def _native_xgb_explainer(booster: xgb.Booster):
    """Callable with the `shap.TreeExplainer` call interface, backed by XGBoost's exact `pred_contribs`.

    Values are in log-odds, like TreeExplainer on the other boosters. Extra keyword arguments
    (`check_additivity`, `max_evals`) are accepted and ignored.
    """
    def explain(X: pd.DataFrame, **_ignored) -> shap.Explanation:
        contribs = booster.predict(xgb.DMatrix(X), pred_contribs=True)
        return shap.Explanation(
            values=contribs[:, :-1],
            base_values=contribs[:, -1],
            data=X.values,
            feature_names=X.columns.tolist(),
        )
    return explain


class ModelExplainer:
    def __init__(
        self,
        model,
        X_train: pd.DataFrame,
        feature_names: list = None,
        max_background_samples: int = 1000,
        output_dir: Path = None,
        *,
        seed: int,
    ):
        self.model = model
        self.seed = seed
        self.output_dir = output_dir
        self.is_linear_pipeline = False
        self.preprocessor = None
        
        if self.output_dir is None:
            self.output_dir = Path("artifacts/xai")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if hasattr(self.model, "transform"):
            X_train_transformed = self.model.transform(X_train)
        else:
            X_train_transformed = X_train

        # Names of the columns the model actually sees (it may drop or rename some of X_train's).
        self.feature_names = feature_names or X_train_transformed.columns.tolist()
        # Unit of the explainer's values: "log_odds" or "probability" (permutation fallbacks).
        self.output_space = "log_odds"

        if len(X_train_transformed) > max_background_samples:
            self.X_train = X_train_transformed.sample(max_background_samples, random_state=seed)
        else:
            self.X_train = X_train_transformed.copy()

        self.shap_values = None

        logger.info(f"Initializing explainer for {type(model).__name__}")

        self.explainer = self._create_explainer()
        
        if self.explainer is None:
            raise RuntimeError("SHAP explainer initialization failed.")
    

    def compute_global_shap(self, X_sample: pd.DataFrame):
        logger.info("Computing SHAP values...")
        
        self.shap_values = self._explain(X_sample)

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


    def _explain(self, X_raw: pd.DataFrame) -> shap.Explanation:
        """2-D explanation (rows x features) of raw, not yet preprocessed rows."""
        if hasattr(self.model, "transform"):
            X_transformed = self.model.transform(X_raw)
        else:
            X_transformed = X_raw

        if getattr(self, "is_linear_pipeline", False):
            X_for_shap = pd.DataFrame(
                self.preprocessor.transform(X_transformed),
                columns=X_transformed.columns
            )
        else:
            X_for_shap = X_transformed

        max_evals = X_transformed.shape[1] * 2 + 100

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
                explanation = vals[:, :, 1] if vals.shape[2] == 2 else vals[:, :, 0]
            else:
                explanation = vals
        else:
            sv = vals[1] if isinstance(vals, list) and len(vals) > 1 else (vals[0] if isinstance(vals, list) else vals)
            if len(sv.shape) == 3:
                sv = sv[:, :, 1] if sv.shape[2] == 2 else sv[:, :, 0]

            explanation = shap.Explanation(
                values=sv,
                data=X_for_shap.values,
                feature_names=self.feature_names
            )

        if getattr(self, "is_linear_pipeline", False) and isinstance(explanation, shap.Explanation):
            explanation.data = X_transformed.values

        if hasattr(explanation, "feature_names"):
            explanation.feature_names = self.feature_names

        return explanation

    def local_explanation(self, instance: pd.Series) -> shap.Explanation:
        """1-D explanation of one raw row: `values` (n_features,), scalar `base_values`."""
        return self._explain(pd.DataFrame([instance]))[0]

    def _create_explainer(self):
        model = self.model

        while hasattr(model, "model"):
            model = model.model
            logger.info(f"Unwrapped wrapper -> {type(model).__name__}")

        if isinstance(model, Pipeline) or type(model).__name__ == "Pipeline":
            clf = model.steps[-1][1]
            clf_name = type(clf).__name__
            
            if "LogisticRegression" in clf_name or "Linear" in clf_name:
                logger.info(
                    "Pipeline ends with LogisticRegression. Extracting LinearExplainer for ultra-fast processing."
                )
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
                self.output_space = "probability"
                masker = shap.maskers.Independent(self.X_train)
                return shap.Explainer(model.predict_proba, masker)

        if isinstance(model, XGBClassifier):
            return self._create_xgb_explainer(model)

        if isinstance(model, (LGBMClassifier, CatBoostClassifier)):
            logger.info(f"Using TreeExplainer for {type(model).__name__}")
            return shap.TreeExplainer(model)

        if isinstance(model, LogisticRegression):
            masker = shap.maskers.Independent(self.X_train)
            return shap.LinearExplainer(model, masker)

        return shap.TreeExplainer(model)
            

    def _create_xgb_explainer(self, model: XGBClassifier):
        """Exact TreeSHAP for XGBoost, in three steps, each logged when it fails.

        1. `shap.TreeExplainer` on the booster;
        2. native `pred_contribs` of the booster (exact, log-odds), used when shap cannot parse the
           booster: shap 0.49 fails on boosters saved by xgboost 3.x (`base_score` is '[5E-1]');
        3. a model-agnostic permutation explainer on probabilities, the last resort (slow, approximate).
        """
        booster = model.get_booster()
        try:
            logger.info("Using TreeExplainer on the XGBoost booster")
            return shap.TreeExplainer(booster)
        except Exception as e:
            logger.warning(f"TreeExplainer failed for XGBoost ({e}); using native booster TreeSHAP")

        try:
            explain = _native_xgb_explainer(booster)
            explain(self.X_train.head(1))  # fail here, not in the middle of the training run
            return explain
        except Exception as e:
            logger.warning(f"Native XGBoost TreeSHAP failed ({e}); falling back to a model-agnostic explainer")

        self.output_space = "probability"
        background = self.X_train.sample(min(200, len(self.X_train)), random_state=self.seed)
        return shap.Explainer(
            lambda d: model.predict_proba(d)[:, 1], shap.maskers.Independent(background), seed=self.seed
        )

    def explain_local_shap(self, instance: pd.Series, save_path: str = None, max_display: int = 15) -> str:
        if save_path is None:
            save_path = self.output_dir / "local_shap_waterfall.png"
    
        local_shap = self._explain(pd.DataFrame([instance]))

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


def compare_local_contributions(
    models: dict[str, BaseModel],
    X_background: pd.DataFrame,
    instance: pd.Series,
    *,
    seed: int,
    top_n: int,
) -> pd.DataFrame:
    """SHAP contributions (log-odds) of one raw row for several models, aligned by feature name.

    Rows are features (union over the models), columns are the `models` keys in order; a feature the
    model does not use is NaN. Rows are sorted by the largest |contribution| over the models and cut
    to `top_n`. The index holds the models' sanitized feature names (`BaseModel._sanitize_columns`),
    which may differ from the raw column names.

    Raises:
        ValueError: if a model's explainer does not work in log-odds (values would not be comparable).
    """
    columns = {}
    with tempfile.TemporaryDirectory() as tmp:
        for name, model in models.items():
            explainer = ModelExplainer(model, X_background, seed=seed, output_dir=Path(tmp))
            if explainer.output_space != "log_odds":
                raise ValueError(
                    f"Model '{name}': explainer works in {explainer.output_space} space, "
                    "contributions are comparable only in log_odds."
                )
            exp = explainer.local_explanation(instance)
            columns[name] = pd.Series(exp.values, index=exp.feature_names)

    table = pd.DataFrame(columns)[list(models)]
    order = table.abs().max(axis=1).sort_values(ascending=False, kind="stable").index
    return table.loc[order].head(top_n)
