import abc
import numpy as np
import pandas as pd
import mlflow
import logging
from typing import Tuple, Dict, Any, Optional, List
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)


class BaseModel(abc.ABC):
    """
    Abstract base class for all credit scoring machine learning models.
    
    Enforces a standard API for fitting, predicting, and cross-validating models.
    Integrates MLflow for experiment tracking and provides threshold optimization
    utilities specifically tailored for business metrics (e.g., cost of default).
    """
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Args:
            params (Optional[Dict[str, Any]]): Model hyperparameters.
        """
        self.params = params or {}
        self.model = None
        self.features_: List[str] =[]

    @abc.abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, eval_set = None) -> 'BaseModel':
        """Fits the model to the training data"""
        pass

    @abc.abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Returns the probability of the positive class (default)"""
        pass


    @abc.abstractmethod
    def get_optuna_space(self, trial) -> Dict[str, Any]:
        """Defines the hyperparameter search space for Optuna"""
        pass


    def cross_validate(self, X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> Tuple[np.ndarray, List['BaseModel']]:
        """
        Executes Stratified K-Fold cross-validation, logs metrics to MLflow,
        and generates Out-Of-Fold (OOF) predictions for stacking.

        Args:
            X (pd.DataFrame): Feature matrix.
            y (pd.Series): Target variable.
            n_splits (int): Number of CV folds.

        Returns:
            Tuple[np.ndarray, List['BaseModel']]: 1D array of OOF probabilities and a list of fitted models.
        """
        logger.info(f"Starting {n_splits}-fold Stratified CV for {self.__class__.__name__}")
        
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        oof_preds = np.zeros(len(X))
        models =[]
        fold_auc, fold_pr_auc = [],[]

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

            # Clone instance for the fold
            fold_model = self.__class__(self.params)
            fold_model.fit(X_train, y_train, eval_set=(X_val, y_val))
            
            val_preds = fold_model.predict_proba(X_val)
            oof_preds[val_idx] = val_preds
            models.append(fold_model)

            # Fold Metrics
            auc = roc_auc_score(y_val, val_preds)
            pr_auc = average_precision_score(y_val, val_preds)
            fold_auc.append(auc)
            fold_pr_auc.append(pr_auc)

            logger.info(f"Fold {fold + 1} | ROC AUC: {auc:.4f} | PR AUC: {pr_auc:.4f}")
            mlflow.log_metrics({f"fold_{fold+1}_auc": auc, f"fold_{fold+1}_pr_auc": pr_auc})

        mean_auc, std_auc = np.mean(fold_auc), np.std(fold_auc)
        logger.info(f"CV Complete | mean_auc: {mean_auc:.4f}, std_auc: {std_auc:.4f}")
        
        mlflow.log_metrics({
            "cv_mean_roc_auc": mean_auc,
            "cv_std_roc_auc": std_auc,
            "cv_mean_pr_auc": np.mean(fold_pr_auc)
        })

        return oof_preds, models

    def optimize_threshold(self, y_true: np.ndarray, y_proba: np.ndarray, fn_cost: float = 10.0, fp_cost: float = 1.0) -> Dict[str, float]:
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
        best_f1_idx = np.argmax(f1_scores)
        best_f1_threshold = thresholds[best_f1_idx] if best_f1_idx < len(thresholds) else 0.5
        
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
        mlflow.log_metrics(metrics)
        return metrics
    
    
    def transform(self, X):
        raise NotImplementedError