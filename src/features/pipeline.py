import pandas as pd
import numpy as np
import json
import mlflow
import logging
import gc
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
from pathlib import Path


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path.cwd()


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
    6. Saves train/test splits to parquet files
    """
    
    def __init__(self, use_cache: bool = True):
        """
        Initialize the feature engineering pipeline.
        
        Args:
            use_cache (bool): Whether to use cached parquet files for faster loading.
                            If False, will reload from CSV. Default: True.
        """
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
        with mlflow.start_run(run_name="Feature_Engineering"):
            # Log input params
            mlflow.log_params({
                "top_n_selection": 300,
                "cache_enabled": self.use_cache,
            })
            
            data = self.loader.load_all(use_cache=self.use_cache)
            
            # Application
            train_raw = self.app_preprocessor.fix_anomalies(data['application_train'])
            test_raw = self.app_preprocessor.fix_anomalies(data['application_test'])
            
            # Remove source files from the data dict after use
            del data['application_train'], data['application_test']
            gc.collect()
            
            train_ids = train_raw['SK_ID_CURR'].unique()
            
            train_app, test_app = self.app_preprocessor.fit_transform_categorical(train_raw, test_raw)
            del train_raw, test_raw
            gc.collect()
            
            df = pd.concat([train_app, test_app], ignore_index=True)
            df = build_application_features(df)
            del train_app, test_app
            gc.collect()
            
            # Bureau and Bureau Balance
            bureau_curr_map = data['bureau'][['SK_ID_BUREAU', 'SK_ID_CURR']].drop_duplicates()
            
            bureau, bureau_cat = process_side_table(data['bureau'], train_ids, self.generic_preprocessor, "Bureau")
            del data['bureau']
            
            bureau_balance = data['bureau_balance'].merge(bureau_curr_map, on='SK_ID_BUREAU', how='inner')
            del data['bureau_balance'], bureau_curr_map
            
            bb, bb_cat = process_side_table(bureau_balance, train_ids, self.generic_preprocessor, "Bureau Balance")
            del bureau_balance
            gc.collect()

            bureau_features = build_bureau_features(bureau, bb, bb_cat, bureau_cat)
            del bureau, bb
            gc.collect()

            # Previous applications
            prev, prev_cat = process_side_table(data['previous_application'], train_ids, self.prev_preprocessor, "Previous")
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
            pos, pos_cat = process_side_table(data['POS_CASH_balance'], train_ids, self.pos_preprocessor, "POS CASH")
            del data['POS_CASH_balance']
            
            pos_features = build_pos_cash_features(pos, pos_cat)
            del pos
            gc.collect()

            # Credit card
            cc, cc_cat = process_side_table(data['credit_card_balance'], train_ids, self.cc_preprocessor, "Credit Card")
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
            df = self._select_features_ridge(df)
            self._save_results(df)
            
            # Log metrics and artifacts
            mlflow.log_metric("df_columns", df.shape[1])
            mlflow.log_metric("df_rows", df.shape[0])
            
            # Save data head
            sample_path = self.output_dir / "sample_features.csv"
            df.head(100).to_csv(sample_path, index=False)
            mlflow.log_artifact(str(sample_path))
            
            return df
        

    def _select_features_ridge(self, df: pd.DataFrame, top_n: int = 300) -> pd.DataFrame:
        """
        Perform feature selection using Ridge regression coefficient magnitude.
        
        Uses Ridge regression to identify the most important features based on 
        the magnitude of their coefficients. This is a filter-based feature 
        selection method that works well with multicollinear data.
        
        Args:
            df (pd.DataFrame): Feature matrix with 'TARGET' and 'SK_ID_CURR' columns.
            top_n (int): Maximum number of features to select. Default: 300.
            
        Returns:
            pd.DataFrame: DataFrame with selected features plus 'SK_ID_CURR' and 'TARGET'.
        """
        logger.info(f"Running Ridge-based feature selection (top_n={top_n})")
        
        # Separate training data
        train_df = df[df['TARGET'].notnull()]
        X = train_df.drop(columns=['SK_ID_CURR', 'TARGET']).select_dtypes(include=['number'])
        y = train_df['TARGET']
        
        logger.info(f"Feature matrix shape before selection: {X.shape}")
        
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        X_prepared = pipeline.fit_transform(X)
        
        # Fit Ridge and select top features
        selector = SelectFromModel(Ridge(alpha=1.0), max_features=top_n, threshold=-np.inf)
        selector.fit(X_prepared, y)
        
        ridge_model = selector.estimator_
        importance = np.abs(ridge_model.coef_)
        
        feature_importance = pd.DataFrame({'feature': X.columns, 'importance': importance}).sort_values(by='importance', ascending=False)
    
        # Save and log feature importance
        importance_path = self.output_dir / "feature_importance.csv"
        feature_importance.to_csv(importance_path, index=False)
        mlflow.log_artifact(str(importance_path))

        selected_cols = X.columns[selector.get_support()].tolist()

        json_path = self.output_dir / "selected_features.json"
        with open(json_path, "w") as f:
            json.dump(selected_cols, f, indent=2)

        mlflow.log_artifact(str(json_path))

        final_cols = ['SK_ID_CURR', 'TARGET'] + selected_cols

        logger.info(f"Selected {len(selected_cols)} features out of {X.shape[1]}")
        return df[final_cols]


    def _save_results(self, df: pd.DataFrame) -> None:
        """
        Save train and test feature matrices to parquet files.
        
        Separates training and test data based on TARGET column 
        (non-null = train, null = test) and saves to disk in parquet format.
        
        Args:
            df (pd.DataFrame): Complete feature matrix with TARGET column.
            
        Returns:
            None
        """
        train = df[df["TARGET"].notnull()]
        test = df[df["TARGET"].isnull()].drop(columns=["TARGET"])

        train_path = self.output_dir / "train_features.parquet"
        test_path = self.output_dir / "test_features.parquet"

        train.to_parquet(train_path, index=False)
        test.to_parquet(test_path, index=False)

        mlflow.log_artifact(str(train_path))
        mlflow.log_artifact(str(test_path))

        logger.info(f"Saved train to {train_path}")
        logger.info(f"Saved test to {test_path}")



if __name__ == "__main__":
    """
    Main entry point for the feature engineering pipeline.
    
    Run this script to execute the complete pipeline:
        python -m src.features.pipeline
    """
    pipeline = FeatureEngineeringPipeline(use_cache=True)
    df = pipeline.run()
    logger.info("Feature engineering pipeline completed successfully!")
