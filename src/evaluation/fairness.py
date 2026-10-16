import logging

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)

DAYS_PER_YEAR = 365
AGE_BINS = [0, 30, 40, 50, 60, 200]
AGE_LABELS = ["<30", "30-39", "40-49", "50-59", "60+"]


def age_groups(days_birth: pd.Series) -> pd.Series:
    """Age bands from DAYS_BIRTH given as positive days; values outside [0, 200) years become NaN."""
    years = days_birth / DAYS_PER_YEAR
    return pd.cut(years, AGE_BINS, right=False, labels=AGE_LABELS)


def fairness_report(
    y_true: np.ndarray, y_proba: np.ndarray, groups: pd.Series, threshold: float
) -> pd.DataFrame:
    """Per-group approval statistics at a fixed threshold (approve when y_proba < threshold).

    Returns a frame indexed by group with columns n, approval_rate, good_approval_rate
    (P(approved | y=0), the equal opportunity rate of good borrowers) and auc.
    A group with a single class has auc = NaN and a warning; rows with a missing group are dropped.
    """
    frame = pd.DataFrame({
        "y": np.asarray(y_true),
        "p": np.asarray(y_proba, dtype=float),
        "group": np.asarray(groups),
    })
    frame["group"] = frame["group"].where(pd.notna(frame["group"]))
    frame["approved"] = frame["p"] < threshold

    rows = {}
    for name, part in frame.dropna(subset=["group"]).groupby("group", observed=True, sort=True):
        good = part.loc[part["y"] == 0, "approved"]
        if part["y"].nunique() < 2:
            logger.warning(f"Fairness group '{name}' has a single class in y_true (n={len(part)}): auc is NaN")
            auc = np.nan
        else:
            auc = float(roc_auc_score(part["y"], part["p"]))
        rows[name] = {
            "n": len(part),
            "approval_rate": float(part["approved"].mean()),
            "good_approval_rate": float(good.mean()) if len(good) else np.nan,
            "auc": auc,
        }
    report = pd.DataFrame.from_dict(rows, orient="index", columns=["n", "approval_rate", "good_approval_rate", "auc"])
    report.index.name = "group"
    return report


def fairness_summary(report: pd.DataFrame) -> dict[str, float]:
    """Disparate impact ratio (min/max approval rate) and equal opportunity gap; NaN groups are ignored."""
    approval = report["approval_rate"].dropna()
    good = report["good_approval_rate"].dropna()
    di_ratio = float(approval.min() / approval.max()) if len(approval) and approval.max() > 0 else float("nan")
    eo_gap = float(good.max() - good.min()) if len(good) else float("nan")
    return {"di_ratio": di_ratio, "eo_gap": eo_gap}
