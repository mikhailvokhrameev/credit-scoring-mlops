import pandas as pd
import logging
import gc

from src.config import ROOT
from src.data.loader import DataLoader
from src.data.preprocessor import ApplicationPreprocessor
from src.features.application import build_application_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

class FeatureEngineeringPipeline:
    """
    Base feature engineering pipeline.

    Processes ONLY:
        - application_train
        - application_test

    Output:
        data/processed_base/application_train.parquet
        data/processed_base/application_test.parquet
    """
    def __init__(self, use_cache: bool = True):

        self.loader = DataLoader(
            raw_dir=str(ROOT / "data" / "raw"),
            cache_dir=str(ROOT / "data" / "cache")
        )

        self.use_cache = use_cache

        self.output_dir = ROOT / "data" / "processed_base"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.app_preprocessor = ApplicationPreprocessor()


    def run(self):

        logger.info("Loading application data...")

        data = self.loader.load_all(use_cache=self.use_cache)

        train_raw = data["application_train"]
        test_raw = data["application_test"]

        # free memory
        del data
        gc.collect()

        logger.info("Fixing anomalies...")
        train_raw = self.app_preprocessor.fix_anomalies(train_raw)
        test_raw = self.app_preprocessor.fix_anomalies(test_raw)

        logger.info("Encoding categoricals...")
        train_app, test_app = self.app_preprocessor.fit_transform_categorical(
            train_raw,
            test_raw
        )

        del train_raw, test_raw
        gc.collect()

        logger.info("Building application features...")
        df = pd.concat([train_app, test_app], ignore_index=True)

        del train_app, test_app
        gc.collect()

        df = build_application_features(df)

        train = df[df["TARGET"].notnull()]
        test = df[df["TARGET"].isnull()].drop(columns=["TARGET"])

        train_path = self.output_dir / "application_train.parquet"
        test_path = self.output_dir / "application_test.parquet"

        train.to_parquet(train_path, index=False)
        test.to_parquet(test_path, index=False)

        logger.info(f"Saved train → {train_path}")
        logger.info(f"Saved test → {test_path}")

        return train, test


if __name__ == "__main__":

    pipeline = FeatureEngineeringPipeline(use_cache=True)
    pipeline.run()

    logger.info("Base application pipeline completed successfully!")