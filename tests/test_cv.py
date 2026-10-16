import numpy as np
from sklearn.model_selection import StratifiedKFold

from src.config import load_params
from src.models.base import CVResult
from src.models.lgbm_model import LGBMModel
from src.models.logreg_model import LogRegModel
from tests.conftest import features

N_SPLITS = 3
ES_ROUNDS = 20
ES_FRACTION = 0.1


def lgbm_params() -> dict:
    return {"n_estimators": 50, "random_state": load_params()["seed"], "verbose": -1}


def test_cv_reproducible(synth):
    X, y = features(synth), synth["TARGET"]
    seed = load_params()["seed"]
    a = LGBMModel(lgbm_params()).cross_validate(X, y, N_SPLITS, seed, ES_ROUNDS, ES_FRACTION)
    b = LGBMModel(lgbm_params()).cross_validate(X, y, N_SPLITS, seed, ES_ROUNDS, ES_FRACTION)
    np.testing.assert_array_equal(a.oof, b.oof)


def test_cv_val_fold_not_used_for_early_stopping(synth, monkeypatch):
    X, y = features(synth), synth["TARGET"]
    seed = load_params()["seed"]
    calls = []
    original_fit = LGBMModel.fit

    def recording_fit(self, X_fit, y_fit, eval_set=None, early_stopping_rounds=None):
        calls.append(
            {
                "fit": set(X_fit.index),
                "es": None if eval_set is None else set(eval_set[0].index),
                "rounds": early_stopping_rounds,
            }
        )
        return original_fit(self, X_fit, y_fit, eval_set=eval_set, early_stopping_rounds=early_stopping_rounds)

    monkeypatch.setattr(LGBMModel, "fit", recording_fit)
    cv = LGBMModel(lgbm_params()).cross_validate(X, y, N_SPLITS, seed, ES_ROUNDS, ES_FRACTION)

    folds = list(StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed).split(X, y))
    assert len(calls) == N_SPLITS
    for call, (train_idx, val_idx) in zip(calls, folds):
        val_index = set(X.index[val_idx])
        assert call["es"] is not None and call["rounds"] == ES_ROUNDS
        assert call["es"].isdisjoint(val_index)
        assert call["fit"].isdisjoint(val_index)
        assert call["fit"].isdisjoint(call["es"])
        assert call["fit"] | call["es"] == set(X.index[train_idx])

    assert len(cv.oof) == len(X)
    assert not np.any(np.isnan(cv.oof)) and np.all((cv.oof > 0) & (cv.oof < 1))
    assert len(cv.models) == len(cv.fold_auc) == len(cv.best_iterations) == N_SPLITS


def test_cv_logreg_has_no_inner_split(synth, monkeypatch):
    X, y = features(synth), synth["TARGET"]
    seed = load_params()["seed"]
    calls = []
    original_fit = LogRegModel.fit

    def recording_fit(self, X_fit, y_fit, eval_set=None, early_stopping_rounds=None):
        calls.append((len(X_fit), eval_set))
        return original_fit(self, X_fit, y_fit, eval_set=eval_set, early_stopping_rounds=early_stopping_rounds)

    monkeypatch.setattr(LogRegModel, "fit", recording_fit)
    cv = LogRegModel({"C": 1.0}).cross_validate(X, y, N_SPLITS, seed, ES_ROUNDS, ES_FRACTION)

    folds = list(StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed).split(X, y))
    assert [n for n, _ in calls] == [len(train_idx) for train_idx, _ in folds]
    assert all(es is None for _, es in calls)
    assert cv.best_iterations == []
    assert 0.5 < cv.mean_auc <= 1.0


def test_refit_params_uses_median_best_iteration():
    cv = CVResult(oof=np.zeros(3), models=[], fold_auc=[0.7] * 3, best_iterations=[10, 30, 20])
    params = {"n_estimators": 2000}
    assert LGBMModel(params).refit_params(cv)["n_estimators"] == 21
    assert params == {"n_estimators": 2000}  # original params are not mutated
    assert LogRegModel({"C": 1}).refit_params(cv) == {"C": 1}
