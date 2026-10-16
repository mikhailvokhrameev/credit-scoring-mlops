import logging

import numpy as np
import pandas as pd
import pytest

from src.evaluation.fairness import age_groups, fairness_report, fairness_summary
from src.evaluation.metrics import plot_reliability, reliability_table


def test_disparate_impact_four_fifths():
    y = np.zeros(8)
    p = np.array([.1, .1, .1, .9, .1, .9, .9, .9])
    g = pd.Series(list("aaaabbbb"))
    s = fairness_summary(fairness_report(y, p, g, .5), min_group_size=1)
    assert s["di_ratio"] == pytest.approx(0.25 / 0.75)
    assert s["eo_gap"] == pytest.approx(0.5)


def test_fairness_degenerate_group(caplog):
    y = np.array([0, 1, 0, 0])
    p = np.array([.1, .8, .2, .3])
    g = pd.Series(["a", "a", "b", "b"])
    r = fairness_report(y, p, g, .5)
    assert np.isnan(r.loc["b", "auc"]) and np.isfinite(r.loc["a", "auc"])
    warnings = [rec.getMessage() for rec in caplog.records if rec.levelno == logging.WARNING]
    assert any("group 'b'" in m for m in warnings)
    assert list(r.columns) == ["n", "approval_rate", "good_approval_rate", "auc"]


def test_fairness_summary_ignores_nan_groups():
    report = pd.DataFrame(
        {"n": [10, 10, 10], "approval_rate": [0.5, 1.0, np.nan], "good_approval_rate": [0.4, 0.9, np.nan]},
        index=["a", "b", "c"],
    )
    s = fairness_summary(report, min_group_size=1)
    assert s["di_ratio"] == pytest.approx(0.5) and s["eo_gap"] == pytest.approx(0.5)


def test_fairness_summary_all_rejected_is_nan_not_crash():
    y = np.array([0, 1, 0, 1])
    p = np.array([.9, .9, .9, .9])
    s = fairness_summary(fairness_report(y, p, pd.Series(list("aabb")), .5), min_group_size=1)
    assert np.isnan(s["di_ratio"]) and s["eo_gap"] == 0.0


def test_fairness_summary_ignores_tiny_groups(caplog):
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 201)
    p = rng.random(201)
    g = pd.Series(["a"] * 100 + ["b"] * 100 + ["tiny"])
    p[-1], y[-1] = 0.99, 1  # a lone rejected default would pin di_ratio to 0
    report = fairness_report(y, p, g, .5)
    full = fairness_summary(report, min_group_size=1)
    with caplog.at_level(logging.WARNING):
        s = fairness_summary(report, min_group_size=30)
    assert s == fairness_summary(report.drop(index="tiny"), min_group_size=1)
    assert s["di_ratio"] > full["di_ratio"] == 0.0
    assert "tiny" in caplog.text and "tiny" in report.index


def test_age_groups_edges():
    assert list(age_groups(pd.Series([29 * 365, 30 * 365, 61 * 365]))) == ["<30", "30-39", "60+"]


def test_reliability_table_and_plot(tmp_path):
    rng = np.random.default_rng(0)
    p = rng.random(200)
    y = (rng.random(200) < p).astype(int)
    table = reliability_table(y, p, n_bins=5)
    assert list(table.columns) == ["bin", "mean_pred", "frac_pos", "count"]
    assert table["count"].sum() == 200
    out = plot_reliability(table, tmp_path / "rel.png")
    assert out.exists() and out.stat().st_size > 0
