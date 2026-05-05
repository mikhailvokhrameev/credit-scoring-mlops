import argparse
import pandas as pd
import json
import joblib
import mlflow
import logging
from pathlib import Path

# Models import
from src.models.logreg_model import LogRegModel
#from src.models.lgbm_model import LGBMModel
# from src.models.xgb_model import XGBModel 
# from src.models.catboost_model import CatBoostModel

from src.models.tuner import OptunaHPOTuner
from src.evaluation.metrics import compute_all_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

MODEL_REGISTRY = {
    "logreg": LogRegModel,
    #"lgbm": LGBMModel,
    # "xgb": XGBModel,
    # "catboost": CatBoostModel
}

def parse_args():
    parser = argparse.ArgumentParser(description="Pipeline for Credit Scoring Models")
    parser.add_argument("--model", type=str, required=True, choices=MODEL_REGISTRY.keys(),
                        help="Name of the model to train (e.g. lgbm, logreg)")
    parser.add_argument("--trials", type=int, default=30,
                        help="Number of the Optuna HPO iterations")
    parser.add_argument("--folds", type=int, default=5,
                        help="Folds number for the final CV")
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

    X = df[feature_cols]
    y = df["TARGET"]

    logger.info(f"Data shape: {X.shape}, target rate: {y.mean():.4f}")

    # MLflow setup
    mlflow.set_experiment("Home_Credit_Default_Risk")
    
    # HPO
    with mlflow.start_run(run_name=f"{model_name}_hpo"):

        tuner = OptunaHPOTuner(
            model_class=ModelClass,
            db_path="sqlite:///optuna_tuning.db",
            study_name=f"{model_name}_hyperopt"
        )
        
        logger.info(f"Starting {args.trials} HPO trials for {model_name}...")
        tuner.optimize(X, y, n_trials=args.trials)
        
        best_params = tuner.study.best_params
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

        # Save artifacts
        model_path = ARTIFACT_DIR / "model.joblib"
        joblib.dump(final_model, model_path)
        mlflow.log_artifact(str(model_path), "models")

        # Feature importance
        fi_df = final_model.get_feature_importance()
        fi_path = ARTIFACT_DIR / "feature_importance.csv"
        fi_df.to_csv(fi_path, index=False)
        mlflow.log_artifact(str(fi_path), "insights")

        # Thresholds
        thresh_path = ARTIFACT_DIR / "thresholds.json"
        with open(thresh_path, "w") as f:
            json.dump(thresholds, f, indent=4)
        mlflow.log_artifact(str(thresh_path), "config")

        logger.info(f"The pipeline for {model_name} is successfully completed! The model is ready for usage!")

if __name__ == "__main__":
    main()