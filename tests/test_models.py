import copy

import numpy as np
import pytest

from src.config import load_params
from src.models.catboost_model import CatBoostModel
from src.models.lgbm_model import LGBMModel
from src.models.logreg_model import LogRegModel
from src.models.xgb_model import XGBModel
from tests.conftest import features

N_ESTIMATORS = 50
ES_ROUNDS = 10
MODELS = [LogRegModel, LGBMModel, XGBModel, CatBoostModel]


def small_params(cls) -> dict:
    seed = load_params()["seed"]
    if cls is LogRegModel:
        return {"C": 1.0}
    if cls is LGBMModel:
        return {"n_estimators": N_ESTIMATORS, "random_state": seed, "verbose": -1}
    if cls is XGBModel:
        return {"n_estimators": N_ESTIMATORS, "random_state": seed}
    return {"iterations": N_ESTIMATORS, "random_seed": seed, "thread_count": 1}


@pytest.fixture(params=MODELS, ids=lambda c: c.__name__)
def cls(request):
    return request.param


def test_missing_feature_raises(cls, synth):
    X, y = features(synth), synth["TARGET"]
    m = cls(small_params(cls)).fit(X, y)
    with pytest.raises(ValueError, match="EXT_SOURCE_2"):
        m.predict_proba(X.drop(columns=["EXT_SOURCE_2"]))


def test_align_extra_and_reordered_columns(cls, synth):
    X, y = features(synth), synth["TARGET"]
    m = cls(small_params(cls)).fit(X, y)
    shuffled = X[X.columns[::-1]].assign(EXTRA_COL=1.0)
    np.testing.assert_allclose(m.predict_proba(shuffled), m.predict_proba(X))


def test_unfitted_model_raises(cls, synth):
    with pytest.raises(ValueError, match="not fitted"):
        cls(small_params(cls)).predict_proba(features(synth))


def test_fit_does_not_mutate_params(cls, synth):
    p = small_params(cls) | {"device": "cpu"}
    before = copy.deepcopy(p)
    m = cls(p)
    m.fit(features(synth), synth["TARGET"])
    assert m.params == before


def test_best_iteration_set_with_eval_set(cls, synth):
    X, y = features(synth), synth["TARGET"]
    X_tr, y_tr, X_val, y_val = X.iloc[:400], y.iloc[:400], X.iloc[400:], y.iloc[400:]
    m = cls(small_params(cls)).fit(
        X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=ES_ROUNDS
    )
    if cls.n_estimators_key is None:
        assert m.best_iteration_ is None
    else:
        assert isinstance(m.best_iteration_, int)
        it = m.best_iteration_
        assert 0 <= it < N_ESTIMATORS
        # best_iteration_ is the 0-based index for every booster
        if cls is LGBMModel:
            assert it + 1 == m.model.best_iteration_
        elif cls is CatBoostModel:
            assert it + 1 == m.model.tree_count_
        else:
            assert it == m.model.best_iteration


def test_best_iteration_none_without_eval_set(cls, synth):
    m = cls(small_params(cls)).fit(features(synth), synth["TARGET"])
    assert m.best_iteration_ is None


def test_n_estimators_key():
    assert LGBMModel.n_estimators_key == "n_estimators"
    assert XGBModel.n_estimators_key == "n_estimators"
    assert CatBoostModel.n_estimators_key == "iterations"
    assert LogRegModel.n_estimators_key is None


def test_lgbm_best_iteration_when_early_stopping_never_triggers(synth):
    X, y = features(synth), synth["TARGET"]
    params = small_params(LGBMModel) | {"n_estimators": 20}
    m = LGBMModel(params).fit(
        X.iloc[:400], y.iloc[:400], eval_set=(X.iloc[400:], y.iloc[400:]), early_stopping_rounds=1000
    )
    assert 0 <= m.best_iteration_ < 20
    assert m.best_iteration_ + 1 == m.model.best_iteration_
