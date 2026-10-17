def test_heavy_imports():
    import catboost  # noqa: F401
    import lightgbm  # noqa: F401
    import mlflow  # noqa: F401
    import optuna  # noqa: F401
    import shap  # noqa: F401
    import xgboost  # noqa: F401


def test_synthetic_df_shape(synth):
    assert len(synth) == 500 and synth["SK_ID_CURR"].is_unique
    assert 0.1 < synth["TARGET"].mean() < 0.3
