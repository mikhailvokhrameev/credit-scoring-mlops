"""Population Stability Index (PSI) drift monitoring."""
import numpy as np
import pandas as pd


def psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10, eps: float = 1e-6) -> float:
    """PSI of `actual` against `expected`: sum((a - e) * ln(a / e)) over bin shares.

    Bin edges are the unique quantiles of the non-NaN `expected`; the outer bins are open
    (-inf, +inf), so values outside the reference range are still counted. NaN is its own bin.
    Shares are clipped from below at `eps`, which keeps the result finite and non-negative for
    empty bins, constant features and an all-NaN `expected`.

    Args:
        expected: reference sample (train).
        actual: monitored sample (production surrogate).
        n_bins: number of quantile bins of the non-NaN reference values.
        eps: lower clip of the bin shares.

    Returns:
        The PSI; 0 means identical distributions, > 0.25 is conventionally a major shift.
    """
    expected = np.asarray(expected, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    reference = expected[~np.isnan(expected)]
    edges = np.empty(0)
    if reference.size:
        edges = np.unique(np.quantile(reference, np.linspace(0, 1, n_bins + 1)[1:-1]))

    def shares(x: np.ndarray) -> np.ndarray:
        nan = np.isnan(x)
        counts = np.bincount(np.searchsorted(edges, x[~nan], side="right"), minlength=edges.size + 1)
        counts = np.append(counts, nan.sum())
        return np.clip(counts / max(x.size, 1), eps, None)

    e, a = shares(expected), shares(actual)
    return float(np.sum((a - e) * np.log(a / e)))


def drift_report(reference: pd.DataFrame, current: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """PSI per feature, sorted by descending PSI.

    Raises:
        ValueError: if a feature is missing from either frame.
    """
    for name, frame in (("reference", reference), ("current", current)):
        absent = [c for c in features if c not in frame.columns]
        if absent:
            raise ValueError(f"{name} frame lacks features: {absent}")
    report = pd.DataFrame({
        "feature": features,
        "psi": [psi(reference[f].to_numpy(), current[f].to_numpy()) for f in features],
    })
    return report.sort_values("psi", ascending=False, kind="stable").reset_index(drop=True)
