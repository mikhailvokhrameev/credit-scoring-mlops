"""Shared pytest fixtures: synthetic credit-scoring data, no Kaggle files needed."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

TARGET_RATE = 0.2
HOLDOUT_FRACTION = 0.2
NON_FEATURE_COLUMNS = ("SK_ID_CURR", "TARGET")
PROTECTED_COLUMNS = ("CODE_GENDER",)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def synthetic_df(n: int = 500, seed: int = 0) -> pd.DataFrame:
    """Synthetic application table with a learnable TARGET signal.

    TARGET ~ Bernoulli(sigmoid(b - 2 * z(mean EXT_SOURCE) + 0.3 * z(DAYS_BIRTH))),
    where the intercept b is solved so that the expected target rate is 0.2.
    """
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=n)
    ext = np.column_stack(
        [np.clip(0.5 + 0.15 * (0.8 * latent + 0.6 * rng.normal(size=n)), 0.0, 1.0) for _ in range(3)]
    )
    age_years = rng.integers(20, 70, size=n)
    days_birth = age_years * 365 + rng.integers(0, 365, size=n)

    ext_mean = ext.mean(axis=1)
    z_ext = (ext_mean - ext_mean.mean()) / ext_mean.std()
    z_age = (days_birth - days_birth.mean()) / days_birth.std()
    score = -2.0 * z_ext - 0.3 * z_age  # older and higher EXT_SOURCE -> lower risk

    lo, hi = -20.0, 20.0  # bisection for the intercept giving the target rate
    for _ in range(60):
        mid = (lo + hi) / 2
        if _sigmoid(mid + score).mean() < TARGET_RATE:
            lo = mid
        else:
            hi = mid
    target = (rng.random(n) < _sigmoid((lo + hi) / 2 + score)).astype(int)

    ext[rng.random(n) < 0.1, 0] = np.nan  # ~10% missing in EXT_SOURCE_1
    df = pd.DataFrame(
        {
            "SK_ID_CURR": np.arange(100_000, 100_000 + n),
            "TARGET": target,
            "EXT_SOURCE_1": ext[:, 0],
            "EXT_SOURCE_2": ext[:, 1],
            "EXT_SOURCE_3": ext[:, 2],
            "DAYS_BIRTH": days_birth,
            "CODE_GENDER": rng.integers(0, 2, size=n),
            "AMT_CREDIT": np.round(rng.lognormal(mean=12.8, sigma=0.5, size=n), 2),
        }
    )
    for i in range(10):
        df[f"NOISE_{i}"] = rng.normal(size=n)
    return df


@pytest.fixture
def synth() -> pd.DataFrame:
    return synthetic_df()


@pytest.fixture
def synthetic_data_dir(tmp_path: Path) -> Path:
    """Write the `processed/` layout of the training pipeline; return the root dir."""
    full = synthetic_df()
    train, holdout = train_test_split(
        full, test_size=HOLDOUT_FRACTION, stratify=full["TARGET"], random_state=0
    )
    train = train.reset_index(drop=True)
    holdout = holdout.reset_index(drop=True)
    test = synthetic_df(n=200, seed=1).drop(columns="TARGET")
    test["SK_ID_CURR"] += 1_000_000  # ids must not collide with train/holdout

    processed = tmp_path / "processed"
    processed.mkdir()
    train.to_parquet(processed / "train_features.parquet", index=False)
    holdout.to_parquet(processed / "holdout_features.parquet", index=False)
    test.to_parquet(processed / "test_features.parquet", index=False)

    excluded = set(NON_FEATURE_COLUMNS) | set(PROTECTED_COLUMNS)
    selected = [c for c in train.columns if c not in excluded]
    (processed / "selected_features.json").write_text(json.dumps(selected))
    (processed / "holdout_ids.json").write_text(
        json.dumps([int(i) for i in holdout["SK_ID_CURR"]])
    )

    filled = train[selected].fillna(train[selected].median())
    importance = filled.corrwith(train["TARGET"]).abs().sort_values(ascending=False)
    importance.rename_axis("feature").rename("importance").reset_index().to_csv(
        processed / "feature_importance.csv", index=False
    )
    return tmp_path


@pytest.fixture(autouse=True)
def _isolated_mlflow(tmp_path, monkeypatch):
    """Keep MLflow out of the repo: never write ./mlruns during tests."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
