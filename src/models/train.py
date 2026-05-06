import argparse
import pandas as pd
import json
import mlflow
import logging
import joblib
from pathlib import Path
import numpy as np

# Models import
from src.models.logreg_model import LogRegModel
from src.models.lgbm_model import LGBMModel
from src.models.xgb_model import XGBModel 
from src.models.catboost_model import CatBoostModel

from src.models.tuner import OptunaHPOTuner
from src.evaluation.metrics import compute_all_metrics

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

    FEATURES_PATH = Path("data/processed/train_features.parquet")
    SELECTED_PATH = Path("data/processed/selected_features.json")
    ARTIFACT_DIR = Path(f"artifacts/{model_name}_model")
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data
    logger.info("Loading data...")
    df = pd.read_parquet(FEATURES_PATH)
    
    with open(SELECTED_PATH, "r") as f:
        selected_features = json.load(f)

    feature_cols = [c for c in selected_features if c in df.columns]

    X = df[feature_cols].copy()
    X = X.astype(np.float64)
    
    y = df["TARGET"]

    logger.info(f"Data shape: {X.shape}, target rate: {y.mean():.4f}")

    # MLflow setup
    mlflow.set_experiment("Home_Credit_Default_Risk")
    
    base_params = {"device": args.device}
    
    # HPO
    with mlflow.start_run(run_name=f"{model_name}_hpo"):

        tuner = OptunaHPOTuner(
            model_class=ModelClass,
            db_path="sqlite:///optuna.db",
            study_name=f"{model_name}_hyperopt",
        )
        
        logger.info(f"Starting {args.trials} HPO trials for {model_name}...")
        tuner.optimize(X, y, n_trials=args.trials)
        
        best_params = tuner.study.best_params
        best_params.update(base_params) 
        logger.info(f"Best params found: {best_params}")

    # Final eval and training
    with mlflow.start_run(run_name=f"{model_name}_Final_Production"):
        mlflow.log_params(best_params)

        # Chosen model with the best params
        final_model = ModelClass(params=best_params)

        logger.info(f"Run final {args.folds}-fold CV...")
        oof_preds, cv_models = final_model.cross_validate(X, y, n_splits=args.folds)

        # Compute metrics
        metrics = compute_all_metrics(y.values, oof_preds, prefix="final_oof_")
        logger.info(f"Final ROC AUC: {metrics['final_oof_roc_auc']:.4f}")
        logger.info(f"Final PR AUC: {metrics['final_oof_average_precision']:.4f}")

        # Find optimal threshold
        thresholds = final_model.optimize_threshold(y.values, oof_preds, fn_cost=10.0, fp_cost=1.0)
        logger.info(f"Optimal thresholds: {thresholds}")

        # Train on the all data
        logger.info("Train final model on the full dataset...")
        final_model.fit(X, y)

        # Save the model
        mlflow.sklearn.log_model(
            sk_model=final_model,
            artifact_path="model", 
            input_example=X.head(5),
            pyfunc_predict_fn="predict_proba"
        )
        # For backup
        model_path = ARTIFACT_DIR / "model.joblib"
        joblib.dump(final_model, model_path)

        # Feature importance
        fi_df = final_model.get_feature_importance()
        fi_path = ARTIFACT_DIR / "feature_importance.csv"
        fi_df.to_csv(fi_path, index=False)

        # Thresholds
        thresh_path = ARTIFACT_DIR / "thresholds.json"
        with open(thresh_path, "w") as f:
            json.dump(thresholds, f, indent=4)
            
        mlflow.log_artifacts(str(ARTIFACT_DIR), artifact_path="metadata")

        logger.info(f"The pipeline for {model_name} is successfully completed! The model is ready for usage!")

if __name__ == "__main__":
    main()