import argparse
import pandas as pd
import json
import mlflow
import logging
import joblib
from pathlib import Path
import numpy as np
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.models.logreg_model import LogRegModel
from src.models.lgbm_model import LGBMModel
from src.models.xgb_model import XGBModel 
from src.models.catboost_model import CatBoostModel

from src.models.tuner import OptunaHPOTuner
from src.evaluation.metrics import compute_all_metrics
from src.evaluation.explainer import ModelExplainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("mlflow").setLevel(logging.ERROR)

MODEL_REGISTRY = {
    "logreg": LogRegModel,
    "lgbm": LGBMModel,
    "xgb": XGBModel,
    "catboost": CatBoostModel
}

def parse_args():
    parser = argparse.ArgumentParser(description="Pipeline for Credit Scoring Models")
    parser.add_argument("--model", type=str, required=True, choices=MODEL_REGISTRY.keys(),
                        help="Name of the model to train (e.g. lgbm, logreg)")
    parser.add_argument("--trials", type=int, default=30,
                        help="Number of the Optuna HPO iterations")
    parser.add_argument("--folds", type=int, default=5,
                        help="Folds number for the final CV")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "gpu"],
                        help="Device to use for training (cpu or gpu)")
    return parser.parse_args()

def main():
    args = parse_args()
    model_name = args.model
    ModelClass = MODEL_REGISTRY[model_name]  # Dynamically get the required model class

    DATA_DIR = ROOT / "data"
    PROCESSED_DATA = DATA_DIR / "processed"

    TRAIN_DATA = PROCESSED_DATA / "train_features.parquet"
    FEATURES_JSON = PROCESSED_DATA / "selected_features.json"

    ARTIFACT_DIR = ROOT / "artifacts"
    MODEL_DIR = ARTIFACT_DIR / f"{model_name}_model"

    MODEL_PATH = MODEL_DIR / "model.joblib"
    XAI_DIR = MODEL_DIR / "xai"
    
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    XAI_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data
    logger.info("Loading data...")
    df = pd.read_parquet(TRAIN_DATA)
    
    with open(FEATURES_JSON, "r") as f:
        selected_features = json.load(f)

    feature_cols = [c for c in selected_features if c in df.columns]

    X = df[feature_cols].copy()
    X = X.astype(np.float64)
    
    y = df["TARGET"]

    logger.info(f"Data shape: {X.shape}, target rate: {y.mean():.4f}")

    # MLflow setup
    mlflow.set_experiment("Home_Credit_Default_Risk")
    
    base_params = {"device": args.device}
    
    with mlflow.start_run(run_name=model_name):
        
        # HPO
        with mlflow.start_run(run_name="HPO", nested=True):

            tuner = OptunaHPOTuner(model_class=ModelClass, db_path=f"sqlite:///{(ROOT / 'optuna.db').as_posix()}", study_name=f"{model_name}_hyperopt")

            if args.trials > 0:
                logger.info(f"Starting {args.trials} HPO trials for {model_name}...")
                tuner.optimize(X, y, n_trials=args.trials)
                best_params = tuner.study.best_params
                best_params = tuner.study.best_params
                best_params.update(base_params) 
                logger.info(f"Best params found: {best_params}")
            else:
                logger.info("Skipping HPO — using default parameters.")
                best_params = base_params.copy()
            
            

        # Final eval and training
        with mlflow.start_run(run_name="Final_Production", nested=True):

            logger.info("Starting FINAL production training")

            mlflow.log_params(best_params)

            # CV training
            final_model = ModelClass(params=best_params)

            oof_preds, cv_models = final_model.cross_validate(
                X, y, n_splits=args.folds
            )

            metrics = compute_all_metrics(
                y.values,
                oof_preds,
                prefix="final_oof_"
            )

            mlflow.log_metrics(metrics)

            logger.info(
                f"OOF ROC-AUC={metrics['final_oof_roc_auc']:.4f} | "
                f"PR-AUC={metrics['final_oof_average_precision']:.4f}"
            )

            # Threshold optimization
            thresholds = final_model.optimize_threshold(
                y.values,
                oof_preds,
                fn_cost=10.0,
                fp_cost=1.0,
            )

            mlflow.log_dict(thresholds, "inference/thresholds.json")

            # Final fit
            logger.info("Training final model on full dataset")

            final_model.fit(X, y)

            # SHAP explainability
            logger.info("Running SHAP explainability")

            X_sample = X.sample(min(1000, len(X)), random_state=42)

            explainer = ModelExplainer(
                model=final_model,
                X_train=X,
                feature_names=X.columns.tolist(),
                output_dir=XAI_DIR,
            )

            explainer.compute_global_shap(X_sample=X_sample)

            # Log global artifacts
            mlflow.log_artifact(
                str(XAI_DIR / "shap_summary_beeswarm.png"),
                artifact_path="xai/global",
            )
            mlflow.log_artifact(
                str(XAI_DIR / "shap_summary_bar.png"),
                artifact_path="xai/global",
            )

            # Local explanations
            for i in range(3):
                path = explainer.explain_local_shap(
                    instance=X.iloc[i],
                    save_path=str(XAI_DIR / f"local_shap_{i}.png"),
                )

                mlflow.log_artifact(path, artifact_path="xai/local")

            # Feature importance
            importance_df = explainer.get_feature_importance()

            importance_path = XAI_DIR / "shap_feature_importance.csv"
            importance_df.to_csv(importance_path, index=False)

            mlflow.log_artifact(
                str(importance_path),
                artifact_path="xai/global",
            )

            mlflow.log_dict(
                {
                    "n_background_samples": len(X),
                    "n_explain_samples": len(X_sample),
                    "n_local_examples": 3,
                },
                "xai_config.json",
            )

            # Model logging
            logger.info("Logging model artifacts")

            mlflow.sklearn.log_model(
                sk_model=final_model,
                artifact_path="model",
                input_example=X.head(5),
                pyfunc_predict_fn="predict_proba",
            )

            # Backup
            joblib.dump(final_model, MODEL_PATH)
            mlflow.log_artifact(
                str(MODEL_PATH),
                artifact_path="model_backup",
            )

            mlflow.log_params({
                "fn_cost": 10.0,
                "fp_cost": 1.0,
            })

            mlflow.log_artifacts(
                str(MODEL_DIR),
                artifact_path="bundle",
            )

            logger.info(f"Pipeline finished successfully for {model_name}")

if __name__ == "__main__":
    main()