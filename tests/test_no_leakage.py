import json

import mlflow
import numpy as np
import pandas as pd
import pytest

from src.data.preprocessor import ApplicationPreprocessor, PassThroughPreprocessor, process_side_table
from src.data.split import load_ids, make_holdout_ids, save_ids
from src.config import load_params
from src.features.pipeline import FeatureEngineeringPipeline, select_features_ridge, write_feature_sets
from tests.conftest import features


def test_holdout_stratified_and_deterministic(synth):
    a = make_holdout_ids(synth.SK_ID_CURR, synth.TARGET, 0.2, 42)
    assert len(a) == 100 and np.array_equal(a, make_holdout_ids(synth.SK_ID_CURR, synth.TARGET, 0.2, 42))
    assert np.array_equal(a, np.sort(a))
    assert abs(synth.set_index("SK_ID_CURR").loc[a, "TARGET"].mean() - synth.TARGET.mean()) < 0.03


def test_ids_roundtrip(tmp_path, synth):
    ids = make_holdout_ids(synth.SK_ID_CURR, synth.TARGET, 0.2, 42)
    path = tmp_path / "nested" / "holdout_ids.json"
    save_ids(ids, path)
    assert np.array_equal(load_ids(path), ids)
    assert all(isinstance(i, int) for i in json.loads(path.read_text()))


def test_process_side_table_encoders_ignore_holdout():
    side = pd.DataFrame({"SK_ID_CURR": [1, 1, 2, 3], "CAT": ["a", "a", "b", "ONLY_IN_HOLDOUT"]})
    out, cat_cols = process_side_table(side, fit_ids=np.array([1, 2]), preprocessor=PassThroughPreprocessor(),
                                       table_name="t", cardinality_threshold=10)
    assert not any("ONLY_IN_HOLDOUT" in c for c in cat_cols)
    assert len(out) == 4


def test_label_encoder_fit_without_holdout():
    fit = pd.DataFrame({"C": ["x", "y"]})
    rest = pd.DataFrame({"C": ["z"]})
    _, r = ApplicationPreprocessor().fit_transform_categorical(fit, rest)
    # "z" was not seen in fit, so it is encoded as UNKNOWN (the last class)
    assert r["C"].iloc[0] == 2


def test_ridge_selection_independent_of_holdout_rows(synth):
    hold = make_holdout_ids(synth.SK_ID_CURR, synth.TARGET, 0.2, 42)
    fit = synth[~synth.SK_ID_CURR.isin(hold)]
    cols1, _ = select_features_ridge(features(fit), fit.TARGET, 5)
    poisoned = synth.copy()
    poisoned.loc[poisoned.SK_ID_CURR.isin(hold), "NOISE_0"] = poisoned.TARGET * 100
    fit2 = poisoned[~poisoned.SK_ID_CURR.isin(hold)]
    assert select_features_ridge(features(fit2), fit2.TARGET, 5)[0] == cols1


def test_ridge_selection_returns_sorted_importance(synth):
    cols, importance = select_features_ridge(features(synth), synth.TARGET, 5)
    assert len(cols) == 5
    assert list(importance.columns) == ["feature", "importance"]
    assert importance["importance"].is_monotonic_decreasing
    assert set(cols) == set(importance["feature"].head(5))


def test_excluded_features_not_selected(tmp_path, synth):
    hold = make_holdout_ids(synth.SK_ID_CURR, synth.TARGET, 0.2, 42)
    is_hold = synth.SK_ID_CURR.isin(hold)
    train, holdout = synth[~is_hold], synth[is_hold]
    test = synth.drop(columns="TARGET").head(20)
    selected = write_feature_sets(
        tmp_path, train, holdout, test,
        selected=["EXT_SOURCE_1", "CODE_GENDER", "AMT_CREDIT"],
        exclude=["CODE_GENDER"], protected=["CODE_GENDER", "DAYS_BIRTH"],
    )
    assert selected == ["EXT_SOURCE_1", "AMT_CREDIT"]
    assert json.loads((tmp_path / "selected_features.json").read_text()) == selected
    for name in ("train", "holdout", "test"):
        cols = pd.read_parquet(tmp_path / f"{name}_features.parquet").columns
        assert "CODE_GENDER" in cols and "DAYS_BIRTH" in cols
        assert set(selected) <= set(cols)
    assert "TARGET" not in pd.read_parquet(tmp_path / "test_features.parquet").columns
    assert "TARGET" in pd.read_parquet(tmp_path / "holdout_features.parquet").columns


def test_label_encoder_missing_is_consistent_between_fit_and_rest():
    fit = pd.DataFrame({"C": ["x", np.nan, "y", np.nan]})
    rest = pd.DataFrame({"C": [np.nan, "x", "z"]})
    f, r = ApplicationPreprocessor().fit_transform_categorical(fit, rest)
    missing_code = f["C"].iloc[1]
    assert f["C"].iloc[3] == missing_code
    assert r["C"].iloc[0] == missing_code  # NaN in rest = NaN class from fit, not UNKNOWN
    assert r["C"].iloc[1] == f["C"].iloc[0]
    assert r["C"].iloc[2] == f["C"].nunique()  # genuinely unseen -> UNKNOWN


def test_label_encoder_missing_only_in_rest_is_unknown():
    fit = pd.DataFrame({"C": ["x", "y"]})
    rest = pd.DataFrame({"C": [np.nan]})
    _, r = ApplicationPreprocessor().fit_transform_categorical(fit, rest)
    assert r["C"].iloc[0] == 2


def _pipeline_without_data(tmp_path) -> FeatureEngineeringPipeline:
    pipe = FeatureEngineeringPipeline.__new__(FeatureEngineeringPipeline)
    pipe.params = load_params()
    pipe.output_dir = tmp_path
    return pipe


def test_pipeline_ridge_selection_uses_only_fit_ids(tmp_path, synth):
    pipe = _pipeline_without_data(tmp_path)
    hold = make_holdout_ids(synth.SK_ID_CURR, synth.TARGET, 0.2, 42)
    fit_ids = np.setdiff1d(synth.SK_ID_CURR.to_numpy(), hold)
    poisoned = synth.copy()
    poisoned.loc[poisoned.SK_ID_CURR.isin(hold), "NOISE_0"] = poisoned.TARGET * 100
    with mlflow.start_run():
        clean = pipe._select_features_ridge(synth, fit_ids, top_n=5)
        dirty = pipe._select_features_ridge(poisoned, fit_ids, top_n=5)
    assert dirty == clean
    assert "CODE_GENDER" not in clean  # features.exclude never competes for a slot


def test_write_feature_sets_requires_protected_columns(tmp_path, synth):
    train = synth.drop(columns="DAYS_BIRTH")
    with pytest.raises(ValueError, match="DAYS_BIRTH"):
        write_feature_sets(tmp_path, train, synth, synth.drop(columns="TARGET"),
                           selected=["EXT_SOURCE_1"], exclude=[], protected=["DAYS_BIRTH"])
