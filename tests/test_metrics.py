import numpy as np

from src.evaluation.metrics import compute_all_metrics, compute_ks_statistic, evaluate_threshold
from src.models.logreg_model import LogRegModel


def test_ks_perfect_separation():
    assert compute_ks_statistic(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])) == 1.0


def test_gini_from_auc():
    m = compute_all_metrics(np.array([0, 1, 0, 1]), np.array([0.1, 0.9, 0.2, 0.8]))
    assert m["roc_auc"] == 1.0 and m["gini"] == 1.0


def test_f1_threshold_is_real_threshold():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.4, 0.35, 0.8])
    t = LogRegModel().optimize_threshold(y, p, 10.0, 1.0)["best_f1_threshold"]
    assert t in p


def test_evaluate_threshold_counts():
    r = evaluate_threshold(np.array([1, 0, 1, 0]), np.array([0.9, 0.6, 0.2, 0.1]), 0.5, 10.0, 1.0)
    assert (r["fn"], r["fp"], r["business_loss"], r["approval_rate"]) == (1, 1, 11.0, 0.5)
