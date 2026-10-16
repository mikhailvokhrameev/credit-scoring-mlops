import pandas as pd
import numpy as np
import json
import mlflow
import logging
import gc
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from src.data.loader import DataLoader
from src.data.preprocessor import (
    ApplicationPreprocessor,
    PreviousPreprocessor,
    POSCashPreprocessor,
    CreditCardPreprocessor,
    InstallmentsPreprocessor,
    PassThroughPreprocessor,
    process_side_table
)
from src.features.application import build_application_features
from src.features.bureau import build_bureau_features
from src.features.previous import build_previous_features
from src.features.installments import build_installments_features
from src.features.pos_cash import build_pos_cash_features
from src.features.credit_card import build_credit_card_features
from src.config import ROOT, load_params
from src.data.split import load_holdout_ids


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)


def select_features_ridge(X: pd.DataFrame, y: pd.Series, top_n: int) -> tuple[list[str], pd.DataFrame]:
    """
    Select the `top_n` features by Ridge coefficient magnitude.

    Median imputation, standardisation and Ridge are all fitted on the rows of
    `X` only, so callers must pass fit rows (train without holdout).

    Args:
        X (pd.DataFrame): Numeric feature matrix.
        y (pd.Series): Binary target aligned with X.
        top_n (int): Maximum number of features to select (params.yaml: features.top_n).

    Returns:
        tuple: (selected column names, DataFrame[feature, importance] sorted by
        importance descending).
    """
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    X_prepared = pipeline.fit_transform(X)

    selector = SelectFromModel(Ridge(alpha=1.0), max_features=top_n, threshold=-np.inf)
    selector.fit(X_prepared, y)

    importance = np.abs(selector.estimator_.coef_)
    feature_importance = (
        pd.DataFrame({'feature': X.columns, 'importance': importance})
        .sort_values(by='importance', ascending=False)
        .reset_index(drop=True)
    )
    selected_cols = X.columns[selector.get_support()].tolist()
    return selected_cols, feature_importance


def write_feature_sets(
    output_dir: Path,
    train: pd.DataFrame,
    holdout: pd.DataFrame,
    test: pd.DataFrame,
    selected: list[str],
    exclude: list[str],
    protected: list[str],
) -> list[str]:
    """
    Write train/holdout/test feature parquet files and selected_features.json.

    Columns in `exclude` are dropped from the selected list (they are never model
    inputs) but columns in `protected` are kept in all three parquet files so that
    fairness checks can use them; ValueError if a protected column is missing from
    any frame. TARGET is dropped from the test frame.

    Returns:
        list[str]: The final selected feature list written to selected_features.json.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = [c for c in selected if c not in exclude]
    keep = ['SK_ID_CURR', 'TARGET'] + selected
    missing = sorted({c for c in protected for f in (train, holdout, test) if c not in f.columns})
    if missing:
        raise ValueError(f"Protected columns missing from the feature frames: {missing}")
    keep += [c for c in protected if c not in keep]

    train[keep].to_parquet(output_dir / "train_features.parquet", index=False)
    holdout[keep].to_parquet(output_dir / "holdout_features.parquet", index=False)
    test[[c for c in keep if c != 'TARGET']].to_parquet(output_dir / "test_features.parquet", index=False)
    (output_dir / "selected_features.json").write_text(json.dumps(selected, indent=2))
    return selected


class FeatureEngineeringPipeline:
    """
    Orchestrates the complete feature engineering pipeline for credit scoring.
    
    This pipeline:
    1. Loads raw data from CSV files
    2. Preprocesses application data (anomaly fixes, categorical encoding)
    3. Generates features from multiple data sources:
       - Application features (financial ratios, aggregations)
       - Bureau features (credit history, payment patterns)
       - Previous application features (historical borrowing behavior)
       - Installments features (payment performance metrics)
       - POS CASH features (point-of-sale credit patterns)
       - Credit card features (credit utilization patterns)
    4. Merges all features into a unified feature matrix
    5. Performs feature selection using Ridge regression
    6. Saves train/holdout/test splits to parquet files

    The holdout (SK_ID_CURR list from `python -m src.data.split`) is excluded from
    every fit: label/one-hot encoders, cardinality reduction and the Ridge selection
    are fitted on train minus holdout; holdout rows are only transformed.
    """
    
    def __init__(self, use_cache: bool = True):
        """
        Initialize the feature engineering pipeline.
        
        Args:
            use_cache (bool): Whether to use cached parquet files for faster loading.
                            If False, will reload from CSV. Default: True.
        """
        self.params = load_params()
        self.loader = DataLoader(raw_dir=str(ROOT / 'data' / 'raw'), cache_dir=str(ROOT / 'data' / 'cache'))
        self.use_cache = use_cache
        self.output_dir = ROOT / "data" / "processed"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.app_preprocessor = ApplicationPreprocessor()
        self.prev_preprocessor = PreviousPreprocessor()
        self.pos_preprocessor = POSCashPreprocessor()
        self.cc_preprocessor = CreditCardPreprocessor()
        self.inst_preprocessor = InstallmentsPreprocessor()
        self.generic_preprocessor = PassThroughPreprocessor()


    def run(self) -> pd.DataFrame:
        """
        Executes the end-to-end feature engineering pipeline with optimized memory management.

        This method orchestrates the loading, preprocessing, feature generation, 
        and selection stages. It utilizes a high-frequency garbage collection (GC) 
        strategy to maintain a low memory footprint by explicitly deleting large 
        intermediate DataFrames as soon as they are consumed.

        Key Stages:
        1. **Data Loading**: Loads all raw CSV/Parquet tables into memory.
        2. **Main Application Processing**: Fixes anomalies and applies Label 
           Encoding to the primary application tables.
        3. **Feature Construction**: Generates domain-specific features from 
           Bureau, Previous Apps, Installments, POS_CASH, and Credit Card tables. 
           Auxiliary tables are processed through `process_side_table` to 
           guarantee leakage-free encoding.
        4. **Memory Consolidation**: Uses `del` and `gc.collect()` after each 
           major feature block to free up RAM for subsequent joins.
        5. **Feature Merging**: Left-joins all generated feature sets onto the 
           main application matrix using 'SK_ID_CURR'.
        6. **Feature Selection**: Applies Ridge-based selection to reduce the 
           dimensionality to the most impactful `top_n` features.
        7. **Artifact Logging**: Logs the final feature count, saves the 
           feature list to JSON, and stores the final datasets to Parquet 
           via MLflow.

        Returns:
            pd.DataFrame: A unified feature matrix containing selected features, 
                the primary key ('SK_ID_CURR'), and the 'TARGET' (for training rows).

        Note:
            This function starts and manages its own nested MLflow run titled 
            "Feature_Engineering".
        """

        mlflow.set_tracking_uri(self.params["mlflow"]["tracking_uri"])
        mlflow.set_experiment(self.params["mlflow"]["experiment"])
        
        with mlflow.start_run(run_name="Feature_Engineering"):
            # Log input params
            mlflow.log_params({
                "top_n_selection": self.params["features"]["top_n"],
                "cache_enabled": self.use_cache,
            })
            
            cardinality_threshold = self.params["features"]["cardinality_threshold"]
            data = self.loader.load_all(use_cache=self.use_cache)
            
            # Application
            train_raw = self.app_preprocessor.fix_anomalies(data['application_train'])
            test_raw = self.app_preprocessor.fix_anomalies(data['application_test'])
            
            # Remove source files from the data dict after use
            del data['application_train'], data['application_test']
            gc.collect()
            
            holdout_ids = load_holdout_ids()
            train_ids = train_raw['SK_ID_CURR'].unique()
            unknown = np.setdiff1d(holdout_ids, train_ids)
            if len(unknown):
                raise ValueError(f"{len(unknown)} holdout ids are not in application_train, e.g. {unknown[:5].tolist()}")
            fit_ids = np.setdiff1d(train_ids, holdout_ids)
            logger.info(f"Fit rows: {len(fit_ids)}, holdout rows: {len(holdout_ids)}")
            
            # Encoders see fit rows only; holdout rows are transformed like test rows
            is_holdout = train_raw['SK_ID_CURR'].isin(holdout_ids)
            fit_part = train_raw[~is_holdout]
            rest = pd.concat([train_raw[is_holdout], test_raw], ignore_index=True)
            fit_app, rest_app = self.app_preprocessor.fit_transform_categorical(fit_part, rest)
            del train_raw, test_raw, fit_part, rest
            gc.collect()
            
            df = pd.concat([fit_app, rest_app], ignore_index=True)
            df = build_application_features(df)
            del fit_app, rest_app
            gc.collect()
            
            # Bureau and Bureau Balance
            bureau_curr_map = data['bureau'][['SK_ID_BUREAU', 'SK_ID_CURR']].drop_duplicates()
            
            bureau, bureau_cat = process_side_table(data['bureau'], fit_ids, self.generic_preprocessor, "Bureau", cardinality_threshold)
            del data['bureau']
            
            bureau_balance = data['bureau_balance'].merge(bureau_curr_map, on='SK_ID_BUREAU', how='inner')
            del data['bureau_balance'], bureau_curr_map
            
            bb, bb_cat = process_side_table(bureau_balance, fit_ids, self.generic_preprocessor, "Bureau Balance", cardinality_threshold)
            del bureau_balance
            gc.collect()

            bureau_features = build_bureau_features(bureau, bb, bb_cat, bureau_cat)
            del bureau, bb
            gc.collect()

            # Previous applications
            prev, prev_cat = process_side_table(data['previous_application'], fit_ids, self.prev_preprocessor, "Previous", cardinality_threshold)
            del data['previous_application']
            
            previous_features = build_previous_features(prev, prev_cat)
            del prev
            gc.collect()

            # Installments
            inst = self.inst_preprocessor.fix_anomalies(data['installments_payments'])
            del data['installments_payments']
            
            installments_features = build_installments_features(inst)
            del inst
            gc.collect()

            # Pos cash
            pos, pos_cat = process_side_table(data['POS_CASH_balance'], fit_ids, self.pos_preprocessor, "POS CASH", cardinality_threshold)
            del data['POS_CASH_balance']
            
            pos_features = build_pos_cash_features(pos, pos_cat)
            del pos
            gc.collect()

            # Credit card
            cc, cc_cat = process_side_table(data['credit_card_balance'], fit_ids, self.cc_preprocessor, "Credit Card", cardinality_threshold)
            del data['credit_card_balance']
            
            cc_features = build_credit_card_features(cc, cc_cat)
            del cc
            gc.collect()
            
            # Merge features
            df = df.merge(bureau_features, on='SK_ID_CURR', how='left')
            del bureau_features; gc.collect()
            
            df = df.merge(previous_features, on='SK_ID_CURR', how='left')
            del previous_features; gc.collect()
            
            df = df.merge(installments_features, on='SK_ID_CURR', how='left')
            del installments_features; gc.collect()
            
            df = df.merge(pos_features, on='SK_ID_CURR', how='left')
            del pos_features; gc.collect()
            
            df = df.merge(cc_features, on='SK_ID_CURR', how='left')
            del cc_features; gc.collect()
            
            gc.collect()
            
            # Feature selection
            logger.info("Performing feature selection...")
            selected = self._select_features_ridge(df, fit_ids, top_n=self.params["features"]["top_n"])
            df = self._save_results(df, fit_ids, holdout_ids, selected)
            
            # Log metrics and artifacts
            mlflow.log_metric("df_columns", df.shape[1])
            mlflow.log_metric("df_rows", df.shape[0])
            
            # Save data head
            sample_path = self.output_dir / "sample_features.csv"
            df.head(100).to_csv(sample_path, index=False)
            mlflow.log_artifact(str(sample_path))
            
            return df
        

    def _select_features_ridge(self, df: pd.DataFrame, fit_ids: np.ndarray, top_n: int) -> list[str]:
        """
        Select features with Ridge on the fit rows only (train without holdout).

        Columns listed in `features.exclude` are removed from the candidates before
        selection so they never take one of the `top_n` slots.

        Args:
            df (pd.DataFrame): Feature matrix with 'TARGET' and 'SK_ID_CURR' columns.
            fit_ids (np.ndarray): SK_ID_CURR of rows allowed to influence the selection.
            top_n (int): Maximum number of features to select (params.yaml: features.top_n).

        Returns:
            list[str]: Selected feature column names. Importances are saved to
            feature_importance.csv.
        """
        logger.info(f"Running Ridge-based feature selection (top_n={top_n})")

        fit_df = df[df['SK_ID_CURR'].isin(fit_ids)]
        X = fit_df.drop(columns=['SK_ID_CURR', 'TARGET']).select_dtypes(include=['number'])
        X = X.drop(columns=self.params["features"]["exclude"], errors='ignore')
        logger.info(f"Feature matrix shape before selection: {X.shape}")

        selected_cols, feature_importance = select_features_ridge(X, fit_df['TARGET'], top_n)

        importance_path = self.output_dir / "feature_importance.csv"
        feature_importance.to_csv(importance_path, index=False)
        mlflow.log_artifact(str(importance_path))

        logger.info(f"Selected {len(selected_cols)} features out of {X.shape[1]}")
        return selected_cols


    def _save_results(self, df: pd.DataFrame, fit_ids: np.ndarray, holdout_ids: np.ndarray,
                      selected: list[str]) -> pd.DataFrame:
        """
        Save train / holdout / test feature matrices and selected_features.json.

        Rows are split by SK_ID_CURR: `fit_ids` -> train, `holdout_ids` -> holdout,
        remaining rows (TARGET null) -> Kaggle test. Protected columns
        (params.yaml: features.protected) are kept in all three parquet files.

        Returns:
            pd.DataFrame: The saved columns for all rows (train + holdout + test).
        """
        feats = self.params["features"]
        train = df[df['SK_ID_CURR'].isin(fit_ids)]
        holdout = df[df['SK_ID_CURR'].isin(holdout_ids)]
        test = df[df['TARGET'].isnull()]

        selected = write_feature_sets(
            self.output_dir, train, holdout, test, selected, feats["exclude"], feats["protected"]
        )

        for name in ("train_features.parquet", "holdout_features.parquet",
                     "test_features.parquet", "selected_features.json"):
            mlflow.log_artifact(str(self.output_dir / name))

        logger.info(f"Saved train/holdout/test ({len(train)}/{len(holdout)}/{len(test)}) to {self.output_dir}")
        keep = ['SK_ID_CURR', 'TARGET'] + selected
        keep += [c for c in feats["protected"] if c not in keep]
        return df[keep]



if __name__ == "__main__":
    """
    Main entry point for the feature engineering pipeline.
    
    Run this script to execute the complete pipeline:
        python -m src.features.pipeline
    """
    pipeline = FeatureEngineeringPipeline(use_cache=True)
    df = pipeline.run()
    logger.info("Feature engineering pipeline completed successfully!")
