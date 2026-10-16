import argparse
import contextlib
import json
import logging
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

from src.config import ROOT, load_params
from src.evaluation.explainer import ModelExplainer
from src.evaluation.fairness import age_groups, fairness_report, fairness_summary
from src.evaluation.metrics import compute_all_metrics, evaluate_threshold, plot_reliability, reliability_table
from src.models.catboost_model import CatBoostModel
from src.models.lgbm_model import LGBMModel
from src.models.logreg_model import LogRegModel
from src.models.tuner import OptunaHPOTuner, make_study_name
from src.models.xgb_model import XGBModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("mlflow").setLevel(logging.ERROR)

MODEL_REGISTRY = {
    "logreg": LogRegModel,
    "lgbm": LGBMModel,
    "xgb": XGBModel,
    "catboost": CatBoostModel,
}

CHALLENGER_ALIAS = "challenger"
N_LOCAL_EXPLANATIONS = 3
SHAP_SAMPLE_SIZE = 1000
FEATURES_HINT = "run `dvc repro features` first"


def _load_split(data_dir: Path, base: bool) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Read train and holdout frames and the model feature list; the holdout is only loaded here."""
    processed = Path(data_dir) / ("processed_base" if base else "processed")
    required = [processed / "train_features.parquet", processed / "holdout_features.parquet",
                processed / "selected_features.json"]
    if not base:
        required.append(processed / "holdout_ids.json")
    missing = [p.name for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing {missing} in {processed}: {FEATURES_HINT}")

    train = pd.read_parquet(processed / "train_features.parquet")
    holdout = pd.read_parquet(processed / "holdout_features.parquet")
    features = json.loads((processed / "selected_features.json").read_text())

    overlap = set(train["SK_ID_CURR"]) & set(holdout["SK_ID_CURR"])
    if overlap:
        raise ValueError(f"Train and holdout overlap in {len(overlap)} SK_ID_CURR values")
    if not base:
        ids = set(json.loads((processed / "holdout_ids.json").read_text()))
        if ids != set(holdout["SK_ID_CURR"]):
            raise ValueError("holdout_features.parquet does not match holdout_ids.json: " + FEATURES_HINT)
    for name, frame in (("train", train), ("holdout", holdout)):
        absent = [c for c in features if c not in frame.columns]
        if absent:
            raise ValueError(f"{name} frame lacks selected features: {absent}")
    return train, holdout, features


def _log_shap(model, X: pd.DataFrame, xai_dir: Path, seed: int) -> None:
    """SHAP global and local explanations of a fitted model, logged to the active run."""
    logger.info("Starting SHAP explainability block...")
    X_sample = X.sample(min(SHAP_SAMPLE_SIZE, len(X)), random_state=seed)
    explainer = ModelExplainer(
        model=model, X_train=X, feature_names=X.columns.tolist(), output_dir=xai_dir, seed=seed
    )

    explainer.compute_global_shap(X_sample=X_sample)
    mlflow.log_artifact(str(xai_dir / "shap_summary_beeswarm.png"), artifact_path="xai/global")
    mlflow.log_artifact(str(xai_dir / "shap_summary_bar.png"), artifact_path="xai/global")

    n_local = min(N_LOCAL_EXPLANATIONS, len(X))
    for i in range(n_local):
        local_path = explainer.explain_local_shap(
            instance=X.iloc[i], save_path=str(xai_dir / f"local_shap_{i}.png")
        )
        mlflow.log_artifact(str(local_path), artifact_path="xai/local")

    importance_path = xai_dir / "shap_feature_importance.csv"
    explainer.get_feature_importance().to_csv(importance_path, index=False)
    mlflow.log_artifact(str(importance_path), artifact_path="xai/global")
    mlflow.log_dict(
        {
            "n_background_samples": len(X),
            "n_explain_samples": len(X_sample),
            "n_local_examples": n_local,
            "max_display": 20,
        },
        "xai_config.json",
    )


def _log_guardrails(
    holdout_df: pd.DataFrame, y_holdout: np.ndarray, p_holdout: np.ndarray,
    threshold: float, min_group_size: int, guardrail_dir: Path,
) -> dict[str, float]:
    """Log calibration and fairness guardrails of the holdout predictions.

    Guardrails are monitored, never optimized: the tables and the reliability plot go to the
    `guardrails/` artifacts, the summary metrics are returned (and logged to the active run).
    """
    guardrail_dir.mkdir(parents=True, exist_ok=True)
    reliability = reliability_table(y_holdout, p_holdout)
    reliability.to_csv(guardrail_dir / "reliability.csv", index=False)
    plot_reliability(reliability, guardrail_dir / "reliability.png")

    attributes = {
        "gender": holdout_df["CODE_GENDER"].astype(str),
        "age": age_groups(holdout_df["DAYS_BIRTH"]),
    }
    fairness: dict[str, float] = {}
    for attr, groups in attributes.items():
        report = fairness_report(y_holdout, p_holdout, groups, threshold)
        report.to_csv(guardrail_dir / f"fairness_{attr}.csv")
        summary = fairness_summary(report, min_group_size)
        fairness[f"holdout_fairness_{attr}_di_ratio"] = summary["di_ratio"]
        fairness[f"holdout_fairness_{attr}_eo_gap"] = summary["eo_gap"]
        if np.isnan(summary["di_ratio"]):
            logger.warning(f"Disparate impact ratio for '{attr}' is undefined (NaN); it is left out of min_di")
    ratios = [v for k, v in fairness.items() if k.endswith("_di_ratio") and not np.isnan(v)]
    fairness["holdout_fairness_min_di"] = min(ratios) if ratios else float("nan")
    if np.isnan(fairness["holdout_fairness_min_di"]):
        logger.warning("holdout_fairness_min_di is NaN: no attribute has a defined disparate impact ratio")

    mlflow.log_metrics(fairness)
    mlflow.log_artifacts(str(guardrail_dir), artifact_path="guardrails")
    mlflow.set_tag("guardrails", "brier,fairness")
    return fairness


def _register_challenger(model_info, registered_name: str) -> str:
    """Point the `challenger` alias of the registered model at the version just logged."""
    version = model_info.registered_model_version
    if version is None:
        client = MlflowClient()
        versions = client.search_model_versions(f"name = '{registered_name}'")
        version = max((v for v in versions if v.run_id == model_info.run_id), key=lambda v: int(v.version)).version
    MlflowClient().set_registered_model_alias(registered_name, CHALLENGER_ALIAS, str(version))
    return str(version)


def run_training(
    model_name: str,
    params: dict,
    *,
    data_dir: Path,
    artifact_dir: Path,
    trials: int,
    device: str = "cpu",
    base: bool = False,
    register: bool = True,
) -> dict[str, float]:
    """HPO and CV on train, refit on the whole train, evaluate on the holdout exactly once.

    Args:
        model_name: key of MODEL_REGISTRY.
        params: loaded params.yaml (seed, costs, folds, MLflow settings, ...).
        data_dir: directory that contains `processed/` (or `processed_base/` when `base`).
        artifact_dir: root of the per-model artifact folders (`<model>_model`, `<model>_base_model`).
        trials: number of Optuna trials; 0 skips HPO.
        device: `cpu` or `gpu`, passed to the models as a base parameter.
        base: train on the base (application table only) feature set.
        register: log the model to the MLflow Registry and set the `challenger` alias.

    Returns:
        Flat metrics: OOF metrics with the `final_oof_` prefix, holdout metrics with `holdout_`.

    Raises:
        FileNotFoundError: if the holdout (or other pipeline outputs) are missing.
        ValueError: if train and holdout share clients or the holdout does not match its ids.
    """
    previous_uri = mlflow.get_tracking_uri()  # set_tracking_uri is process-wide state
    try:
        return _run_training(
            model_name, params, data_dir=data_dir, artifact_dir=artifact_dir,
            trials=trials, device=device, base=base, register=register,
        )
    finally:
        mlflow.set_tracking_uri(previous_uri)


def _run_training(
    model_name: str,
    params: dict,
    *,
    data_dir: Path,
    artifact_dir: Path,
    trials: int,
    device: str = "cpu",
    base: bool = False,
    register: bool = True,
) -> dict[str, float]:
    ModelClass = MODEL_REGISTRY[model_name]
    seed = params["seed"]
    fn_cost, fp_cost = params["business"]["fn_cost"], params["business"]["fp_cost"]
    train_cfg = params["train"]
    suffix = "_base" if base else ""

    train_df, holdout_df, feature_cols = _load_split(data_dir, base)
    X = train_df[feature_cols].astype(np.float64)
    y = train_df["TARGET"]
    X_holdout = holdout_df[feature_cols].astype(np.float64)  # protected columns stay in holdout_df
    y_holdout = holdout_df["TARGET"]
    logger.info(f"Train shape: {X.shape}, target rate: {y.mean():.4f}; holdout rows: {len(X_holdout)}")

    model_dir = Path(artifact_dir) / f"{model_name}{suffix}_model"
    xai_dir = model_dir / "xai"
    xai_dir.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
    mlflow.set_experiment(params["mlflow"]["experiment"])

    base_params = {"device": device}
    study_name = make_study_name(model_name, X, y, params)

    with mlflow.start_run(run_name=f"{model_name}{suffix}"):
        if trials > 0:
            with mlflow.start_run(run_name="HPO", nested=True):
                tuner = OptunaHPOTuner(
                    ModelClass, f"sqlite:///{(ROOT / 'optuna.db').as_posix()}", study_name, seed, base_params
                )
                logger.info(f"Starting {trials} HPO trials for {model_name}...")
                best_params = tuner.optimize(
                    X, y,
                    n_trials=trials,
                    timeout=params["hpo"]["timeout"],
                    cv_kwargs=dict(
                        n_splits=params["hpo"]["n_splits"],
                        seed=seed,
                        early_stopping_rounds=train_cfg["early_stopping_rounds"],
                        es_fraction=train_cfg["early_stopping_fraction"],
                    ),
                )
                logger.info(f"Best params found: {best_params}")
        else:
            logger.info("Skipping HPO, using default parameters.")
            best_params = dict(base_params)

        # The base run keeps its metrics on the parent run (compare_models reads it there).
        final_run = contextlib.nullcontext() if base else mlflow.start_run(run_name="Final_Production", nested=True)
        with final_run:
            logger.info("Starting FINAL production training")
            mlflow.log_params({
                "fn_cost": fn_cost, "fp_cost": fp_cost, "top_n": params["features"]["top_n"],
                "seed": seed, "study_name": study_name,
            })

            cv_model = ModelClass(params=best_params)
            cv = cv_model.cross_validate(
                X, y,
                n_splits=params["cv"]["n_splits"],
                seed=seed,
                early_stopping_rounds=train_cfg["early_stopping_rounds"],
                es_fraction=train_cfg["early_stopping_fraction"],
            )
            metrics = compute_all_metrics(y.values, cv.oof, prefix="final_oof_")
            logger.info(
                f"OOF ROC-AUC={metrics['final_oof_roc_auc']:.4f} | "
                f"PR-AUC={metrics['final_oof_average_precision']:.4f}"
            )

            thresholds = cv_model.optimize_threshold(y.values, cv.oof, fn_cost=fn_cost, fp_cost=fp_cost)
            mlflow.log_dict(thresholds, "inference/thresholds.json")

            # Refit on the whole train with the median best iteration of the CV folds
            final_params = cv_model.refit_params(cv)
            mlflow.log_params(final_params)
            logger.info("Training final model on the whole train set")
            final_model = ModelClass(params=final_params).fit(X, y)

            # The only evaluation on the holdout
            p_holdout = final_model.predict_proba(X_holdout)
            metrics |= compute_all_metrics(y_holdout.values, p_holdout, prefix="holdout_")
            business = evaluate_threshold(
                y_holdout.values, p_holdout, thresholds["best_business_threshold"], fn_cost, fp_cost
            )
            holdout_business = {f"holdout_{k}": float(v) for k, v in business.items()}
            mlflow.log_metrics(holdout_business)
            metrics |= holdout_business
            metrics |= _log_guardrails(
                holdout_df, y_holdout.values, p_holdout,
                thresholds["best_business_threshold"], params["fairness"]["min_group_size"],
                model_dir / "guardrails",
            )
            metrics = {k: float(v) for k, v in metrics.items()}
            logger.info(
                f"Holdout ROC-AUC={metrics['holdout_roc_auc']:.4f} | "
                f"business loss={metrics['holdout_business_loss']:.1f}"
            )

            if model_name == "xgb":
                logger.info("Skipping SHAP for XGBoost due to known shap library compatibility bugs.")
            else:
                _log_shap(final_model, X, xai_dir, seed)

            registered_name = f"credit_scoring_{model_name}{suffix}" if register else None
            model_info = mlflow.sklearn.log_model(
                sk_model=final_model,
                name="model",
                input_example=X.head(5),
                pyfunc_predict_fn="predict_proba",
                serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,  # skops rejects custom classes
                registered_model_name=registered_name,
            )
            if register:
                version = _register_challenger(model_info, registered_name)
                logger.info(f"Registered {registered_name} v{version} with alias '{CHALLENGER_ALIAS}'")

            model_path = model_dir / "model.joblib"
            joblib.dump(final_model, model_path)
            (model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
            mlflow.log_artifact(str(model_path), artifact_path="model_backup")
            mlflow.log_artifacts(str(model_dir), artifact_path="bundle")

            logger.info(f"The pipeline for {model_name}{suffix} is successfully completed!")
    return metrics


def parse_args(params: dict) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pipeline for Credit Scoring Models")
    parser.add_argument("--model", type=str, required=True, choices=MODEL_REGISTRY.keys(),
                        help="Name of the model to train")
    parser.add_argument("--trials", type=int, default=params["hpo"]["trials"],
                        help="Number of the Optuna HPO iterations (default: hpo.trials)")
    parser.add_argument("--folds", type=int, default=params["cv"]["n_splits"],
                        help="Folds number for the final CV (default: cv.n_splits)")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "gpu"],
                        help="Device to use for training (cpu or gpu)")
    parser.add_argument("--base", action="store_true",
                        help="Train on the base feature set (data/processed_base)")
    parser.add_argument("--params", type=Path, default=None, help="Path to params.yaml")
    return parser.parse_args()


def main():
    # --params selects the config that supplies the other defaults, so peek at it first
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--params", type=Path, default=None)
    params = load_params(pre.parse_known_args()[0].params)

    args = parse_args(params)
    params["cv"]["n_splits"] = args.folds
    run_training(
        args.model, params,
        data_dir=ROOT / "data", artifact_dir=ROOT / "artifacts",
        trials=args.trials, device=args.device, base=args.base,
    )


if __name__ == "__main__":
    main()
