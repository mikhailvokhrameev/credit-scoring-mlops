import logging
from typing import Dict, Tuple
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import mlflow

logger = logging.getLogger(__name__)

def compute_ks_statistic(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Proper KS statistic for credit scoring:
    max difference between cumulative distributions of positive and negative classes.
    """
    if len(y_true) == 0:
        return 0.0

    # Sort by predicted probability
    order = np.argsort(y_prob)
    y_true_sorted = y_true[order]

    pos = (y_true_sorted == 1).astype(int)
    neg = (y_true_sorted == 0).astype(int)

    if pos.sum() == 0 or neg.sum() == 0:
        return 0.0

    cum_pos = np.cumsum(pos) / pos.sum()
    cum_neg = np.cumsum(neg) / neg.sum()

    return float(np.max(np.abs(cum_pos - cum_neg)))


def compute_all_metrics(y_true: np.ndarray, y_prob: np.ndarray, prefix: str = "") -> Dict[str, float]:
    """
    Computes credit scoring metrics with ROC-AUC as primary metric:
    - ROC-AUC
    - Gini
    - KS statistic
    - Average Precision (PR-AUC)

    Automatically logs to MLflow if active run exists.
    """
    metrics = {}

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    # ROC-AUC
    try:
        if len(np.unique(y_true)) < 2:
            auc = 0.5
        else:
            auc = roc_auc_score(y_true, y_prob)
    except Exception as e:
        logger.warning(f"ROC-AUC computation failed: {e}")
        auc = 0.5

    metrics["roc_auc"] = float(auc)
    metrics["gini"] = float(2 * auc - 1)

    # KS statistic
    try:
        metrics["ks_statistic"] = compute_ks_statistic(y_true, y_prob)
    except Exception as e:
        logger.warning(f"KS computation failed: {e}")
        metrics["ks_statistic"] = 0.0

    # PR-AUC
    try:
        metrics["average_precision"] = float(
            average_precision_score(y_true, y_prob)
        )
    except Exception as e:
        logger.warning(f"Average precision failed: {e}")
        metrics["average_precision"] = 0.0
        
    # Apply prefix to returned metrics
    metrics_with_prefix = {
        f"{prefix}{k}": v for k, v in metrics.items()
    }

    # MLflow logging
    if mlflow.active_run():
        try:
            mlflow_metrics = {
                f"{prefix}{k}": v for k, v in metrics.items()
            }
            mlflow.log_metrics(mlflow_metrics)
            logger.info(f"Metrics logged to MLflow with prefix '{prefix}'")
        except Exception as e:
            logger.warning(f"MLflow logging failed: {e}")
    else:
        logger.warning("No active MLflow run found")

    return metrics_with_prefix


def compare_models(results_dict: Dict[str, list]) -> Tuple[str, float]:
    """
    Compares models using ONLY mean ROC-AUC across CV folds.

    Args:
        results_dict: model_name -> list of ROC-AUC scores per fold

    Returns:
        (best_model_name, best_mean_auc)
    """

    if not results_dict:
        raise ValueError("results_dict must not be empty")

    mean_scores = {
        name: float(np.mean(scores))
        for name, scores in results_dict.items()
    }

    best_model_name = max(mean_scores, key=mean_scores.get)
    best_mean_auc = mean_scores[best_model_name]

    logger.info(
        f"Best model: {best_model_name} | "
        f"Mean ROC-AUC: {best_mean_auc:.6f}"
    )

    return best_model_name, best_mean_auc