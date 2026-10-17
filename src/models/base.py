import abc
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class CVResult:
    """Outcome of `BaseModel.cross_validate`."""
    oof: np.ndarray
    models: List['BaseModel']
    fold_auc: List[float]
    best_iterations: List[int]  # 0-based best iteration per fold; empty for non-boosters

    @property
    def mean_auc(self) -> float:
        return float(np.mean(self.fold_auc))


class BaseModel(abc.ABC):
    """
    Abstract base class for all credit scoring machine learning models.
    
    Enforces a standard API for fitting, predicting, and cross-validating models.
    Integrates MLflow for experiment tracking and provides threshold optimization
    utilities specifically tailored for business metrics (e.g., cost of default).
    """
    n_estimators_key: Optional[str] = None  # name of the boosting-rounds param, None if not applicable

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Args:
            params (Optional[Dict[str, Any]]): Model hyperparameters.
        """
        self.params = params or {}
        self.model = None
        self.features_: List[str] = []
        # 0-based index of the best boosting iteration, identical for every booster
        # (tree count = best_iteration_ + 1). None without active early stopping / for non-boosters.
        self.best_iteration_: Optional[int] = None

    @abc.abstractmethod
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set=None,
        early_stopping_rounds: Optional[int] = None,
    ) -> 'BaseModel':
        """Fits the model to the training data.

        Early stopping is active only when both `eval_set` and `early_stopping_rounds`
        are given. Must not modify `self.params`.
        """
        pass

    def _sanitize_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        """Replaces special characters in column names (LightGBM/XGBoost constraints)."""
        X = X.copy()
        X.columns = [re.sub(r"[^0-9a-zA-Z_]+", "_", str(c)) for c in X.columns]
        return X

    def _align_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Selects the training columns in training order; extra columns are dropped.

        Raises:
            ValueError: if the model is not fitted or any training feature is missing.
        """
        if not self.features_:
            raise ValueError("Model is not fitted yet (features_ not found)")
        missing = set(self.features_) - set(X.columns)
        if missing:
            raise ValueError(f"Missing features at inference: {sorted(missing)}")
        return X[self.features_]

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Sanitizes column names and aligns to the training feature set."""
        return self._align_features(self._sanitize_columns(X))

    @abc.abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Returns the probability of the positive class (default)"""
        pass


    @abc.abstractmethod
    def get_optuna_space(self, trial) -> Dict[str, Any]:
        """Defines the hyperparameter search space for Optuna"""
        pass


    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int,
        seed: int,
        early_stopping_rounds: int,
        es_fraction: float,
    ) -> 'CVResult':
        """
        Executes Stratified K-Fold cross-validation and generates Out-Of-Fold (OOF)
        predictions for stacking.

        For boosters, the train part of every fold is split into `fit` and `es` parts;
        early stopping watches only `es`, so the validation fold stays untouched and
        OOF predictions are not optimistically biased. Models without a boosting-rounds
        parameter (`n_estimators_key is None`) are fitted on the whole train part.
        Metrics are logged to MLflow only if there is an active run.

        Args:
            X (pd.DataFrame): Feature matrix.
            y (pd.Series): Target variable.
            n_splits (int): Number of CV folds.
            seed (int): Seed for fold and early-stopping splits.
            early_stopping_rounds (int): Patience of early stopping.
            es_fraction (float): Share of each fold's train part held out for early stopping.

        Returns:
            CVResult: OOF probabilities, fitted fold models, fold AUCs and best iterations.
        """
        logger.info(f"Starting {n_splits}-fold Stratified CV for {self.__class__.__name__}")

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        oof_preds = np.zeros(len(X))
        models: List['BaseModel'] = []
        fold_auc, fold_pr_auc, best_iterations = [], [], []
        log_to_mlflow = mlflow.active_run() is not None

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

            # Clone instance for the fold
            fold_model = self.__class__(self.params)
            if self.n_estimators_key is None:
                fold_model.fit(X.iloc[train_idx], y.iloc[train_idx])
            else:
                y_tr = y.iloc[train_idx]
                fit_idx, es_idx = train_test_split(
                    train_idx, test_size=es_fraction, stratify=y_tr, random_state=seed
                )
                fold_model.fit(
                    X.iloc[fit_idx], y.iloc[fit_idx],
                    eval_set=(X.iloc[es_idx], y.iloc[es_idx]),
                    early_stopping_rounds=early_stopping_rounds,
                )
                if fold_model.best_iteration_ is not None:
                    best_iterations.append(fold_model.best_iteration_)

            val_preds = fold_model.predict_proba(X_val)
            oof_preds[val_idx] = val_preds
            models.append(fold_model)

            # Fold Metrics
            auc = roc_auc_score(y_val, val_preds)
            pr_auc = average_precision_score(y_val, val_preds)
            fold_auc.append(auc)
            fold_pr_auc.append(pr_auc)

            logger.info(f"Fold {fold + 1} | ROC AUC: {auc:.4f} | PR AUC: {pr_auc:.4f}")
            if log_to_mlflow:
                mlflow.log_metrics({f"fold_{fold+1}_auc": auc, f"fold_{fold+1}_pr_auc": pr_auc})

        result = CVResult(oof=oof_preds, models=models, fold_auc=fold_auc, best_iterations=best_iterations)
        std_auc = float(np.std(fold_auc))
        logger.info(f"CV Complete | mean_auc: {result.mean_auc:.4f}, std_auc: {std_auc:.4f}")

        if log_to_mlflow:
            mlflow.log_metrics({
                "cv_mean_roc_auc": result.mean_auc,
                "cv_std_roc_auc": std_auc,
                "cv_mean_pr_auc": float(np.mean(fold_pr_auc)),
            })

        return result

    def refit_params(self, cv: 'CVResult') -> Dict[str, Any]:
        """Params for the final refit: boosting rounds = median best iteration over folds + 1.

        `best_iteration_` is a 0-based index, hence the +1 to get the tree count.
        Returns an unchanged copy of `self.params` when the model has no rounds parameter.
        """
        params = dict(self.params)
        if self.n_estimators_key is not None:
            params[self.n_estimators_key] = int(np.median(cv.best_iterations)) + 1
        return params

    def optimize_threshold(
        self, y_true: np.ndarray, y_proba: np.ndarray, fn_cost: float, fp_cost: float
    ) -> Dict[str, float]:
        """
        Finds the optimal probability threshold based on F1-Score and Business Cost.
        
        Args:
            y_true (np.ndarray): Ground truth labels.
            y_proba (np.ndarray): Predicted probabilities.
            fn_cost (float): Cost of False Negative (missing a default).
            fp_cost (float): Cost of False Positive (rejecting a good client).

        Returns:
            Dict[str, float]: Best thresholds for different optimization targets.
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
        
        # Optimize for F1
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
        # The last PR-curve point (precision=1, recall=0) has no threshold; exclude it.
        f1_scores = f1_scores[:-1]
        best_f1_idx = np.argmax(f1_scores)
        best_f1_threshold = thresholds[best_f1_idx]
        
        # Optimize for Business Loss
        best_loss_threshold, min_loss = 0.5, float('inf')
        for thresh in np.arange(0.01, 1.0, 0.01):
            y_pred = (y_proba >= thresh).astype(int)
            fn = np.sum((y_true == 1) & (y_pred == 0))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            loss = (fn * fn_cost) + (fp * fp_cost)
            
            if loss < min_loss:
                min_loss = loss
                best_loss_threshold = thresh

        metrics = {
            "best_f1_threshold": best_f1_threshold,
            "max_f1_score": f1_scores[best_f1_idx],
            "best_business_threshold": best_loss_threshold,
            "min_business_loss": min_loss
        }
        if mlflow.active_run() is not None:
            mlflow.log_metrics(metrics)
        return metrics
