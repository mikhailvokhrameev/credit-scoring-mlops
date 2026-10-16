import pandas as pd
import logging
import gc

from src.config import ROOT, load_params
from src.data.loader import DataLoader
from src.data.split import load_holdout_ids
from src.data.preprocessor import ApplicationPreprocessor
from src.features.application import build_application_features
from src.features.pipeline import write_feature_sets

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

class FeatureEngineeringPipeline:
    """
    Base feature engineering pipeline.

    Processes ONLY:
        - application_train
        - application_test

    The holdout ids come from `python -m src.data.split`; label encoders are fitted
    on train minus holdout, holdout rows are only transformed.

    Output (same layout as data/processed/, minus feature_importance.csv):
        data/processed_base/train_features.parquet
        data/processed_base/holdout_features.parquet
        data/processed_base/test_features.parquet
        data/processed_base/selected_features.json
    """
    def __init__(self, use_cache: bool = True):

        self.params = load_params()
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

        holdout_ids = load_holdout_ids()
        unknown = set(holdout_ids) - set(train_raw["SK_ID_CURR"])
        if unknown:
            raise ValueError(f"{len(unknown)} holdout ids are not in application_train")

        logger.info("Fixing anomalies...")
        train_raw = self.app_preprocessor.fix_anomalies(train_raw)
        test_raw = self.app_preprocessor.fix_anomalies(test_raw)

        logger.info("Encoding categoricals (fit on train without holdout)...")
        is_holdout = train_raw["SK_ID_CURR"].isin(holdout_ids)
        fit_part = train_raw[~is_holdout]
        rest = pd.concat([train_raw[is_holdout], test_raw], ignore_index=True)
        fit_app, rest_app = self.app_preprocessor.fit_transform_categorical(fit_part, rest)

        del train_raw, test_raw, fit_part, rest
        gc.collect()

        logger.info("Building application features...")
        df = pd.concat([fit_app, rest_app], ignore_index=True)

        del fit_app, rest_app
        gc.collect()

        df = build_application_features(df)

        feats = self.params["features"]
        selected = [c for c in df.columns if c not in ("SK_ID_CURR", "TARGET", *feats["exclude"])]

        train = df[df["TARGET"].notnull() & ~df["SK_ID_CURR"].isin(holdout_ids)]
        holdout = df[df["SK_ID_CURR"].isin(holdout_ids)]
        test = df[df["TARGET"].isnull()]

        write_feature_sets(
            self.output_dir, train, holdout, test, selected, feats["exclude"], feats["protected"]
        )

        logger.info(f"Saved train/holdout/test ({len(train)}/{len(holdout)}/{len(test)}) to {self.output_dir}")

        return train, holdout, test


if __name__ == "__main__":

    pipeline = FeatureEngineeringPipeline(use_cache=True)
    pipeline.run()

    logger.info("Base application pipeline completed successfully!")