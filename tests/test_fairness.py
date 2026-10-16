import numpy as np
import pandas as pd
import pytest

from src.evaluation.fairness import age_groups, fairness_report, fairness_summary
from src.evaluation.metrics import plot_reliability, reliability_table


def test_disparate_impact_four_fifths():
    y = np.zeros(8)
    p = np.array([.1, .1, .1, .9, .1, .9, .9, .9])
    g = pd.Series(list("aaaabbbb"))
    s = fairness_summary(fairness_report(y, p, g, .5))
    assert s["di_ratio"] == pytest.approx(0.25 / 0.75)
    assert s["eo_gap"] == pytest.approx(0.5)


def test_fairness_degenerate_group(caplog):
    y = np.array([0, 1, 0, 0])
    p = np.array([.1, .8, .2, .3])
    g = pd.Series(["a", "a", "b", "b"])
    r = fairness_report(y, p, g, .5)
    assert np.isnan(r.loc["b", "auc"]) and "b" in caplog.text
    assert list(r.columns) == ["n", "approval_rate", "good_approval_rate", "auc"]


def test_fairness_summary_ignores_nan_groups():
    report = pd.DataFrame(
        {"approval_rate": [0.5, 1.0, np.nan], "good_approval_rate": [0.4, 0.9, np.nan]},
        index=["a", "b", "c"],
    )
    s = fairness_summary(report)
    assert s["di_ratio"] == pytest.approx(0.5) and s["eo_gap"] == pytest.approx(0.5)


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
