import argparse
import pandas as pd
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

from src.evaluation.metrics import compute_all_metrics
from src.config import load_params
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

    TRAIN_DATA = DATA_DIR / "processed_base/application_train.parquet"

    ARTIFACT_DIR = ROOT / "artifacts"
    MODEL_DIR = ARTIFACT_DIR / f"{model_name}_base_model"

    MODEL_PATH = MODEL_DIR / "model_base.joblib"
    XAI_DIR = MODEL_DIR / "xai"
    
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    XAI_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data
    logger.info("Loading data...")
    df = pd.read_parquet(TRAIN_DATA)

    y = df["TARGET"]
    X = df.drop(columns=["TARGET"])

    X = X.astype(np.float64)

    logger.info(f"Data shape: {X.shape}, target rate: {y.mean():.4f}")

    # MLflow setup
    mlflow.set_experiment("Home_Credit_Default_Risk")
    
    base_params = {"device": args.device}
    
    with mlflow.start_run(run_name=f"{model_name}_base"):
        logger.info("Starting FINAL production training")

        mlflow.log_params(base_params)

        # CV training
        final_model = ModelClass(params=base_params)

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
        if model_name == "xgb":
            logger.info("Skipping SHAP explainability block for XGBoost due to known shap library compatibility bugs.")
        else:
            logger.info("Starting SHAP explainability block...")

            X_sample = X.sample(min(1000, len(X)), random_state=42)
            explainer = ModelExplainer(model=final_model, X_train=X, feature_names=X.columns.tolist(), output_dir=XAI_DIR, seed=load_params()["seed"])

            # Global explanations
            logger.info("Computing global SHAP explanations...")
            explainer.compute_global_shap(X_sample=X_sample)

            shap_beeswarm_path = XAI_DIR / "shap_summary_beeswarm.png"
            shap_bar_path = XAI_DIR / "shap_summary_bar.png"

            mlflow.log_artifact(str(shap_beeswarm_path), artifact_path="xai/global")
            mlflow.log_artifact(str(shap_bar_path), artifact_path="xai/global")

            # Local explanations
            logger.info("Generating local SHAP explanations...")
            local_paths =[]

            for i in range(3):
                instance = X.iloc[i]
                local_path = explainer.explain_local_shap(instance=instance, save_path=str(XAI_DIR / f"local_shap_{i}.png"))
                local_paths.append(local_path)
                mlflow.log_artifact(str(local_path), artifact_path="xai/local")

            # Feature importance
            logger.info("Computing SHAP feature importance...")
            importance_df = explainer.get_feature_importance()

            importance_path = XAI_DIR / "shap_feature_importance.csv"
            importance_df.to_csv(importance_path, index=False)

            mlflow.log_artifact(str(importance_path), artifact_path="xai/global")

            mlflow.log_dict(
                {
                    "n_background_samples": len(X),
                    "n_explain_samples": len(X_sample),
                    "n_local_examples": 3,
                    "max_display": 20
                },
                "xai_config.json"
            )

            logger.info("SHAP explainability finished.")

        logger.info("Logging final model to MLflow...")

        mlflow.sklearn.log_model(
            sk_model=final_model,
            artifact_path="model",
            input_example=X.head(5),
            pyfunc_predict_fn="predict_proba"
        )

        # Backup
        joblib.dump(final_model, MODEL_PATH)
        mlflow.log_artifact(str(MODEL_PATH), artifact_path="model_backup")

        mlflow.log_params({
            "fn_cost": 10.0,
            "fp_cost": 1.0
        })

        # Full artifact snapshot
        mlflow.log_artifacts(
            str(MODEL_DIR),
            artifact_path="bundle"
        )
        
        logger.info(f"The pipeline for {model_name} is successfully completed! The model is ready for usage!")

if __name__ == "__main__":
    main()