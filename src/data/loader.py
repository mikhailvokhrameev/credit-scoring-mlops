import os
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, raw_dir: str = 'data/raw', cache_dir: str = 'data/cache'):
        self.raw_dir = raw_dir
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def load_table(self, table_name: str, use_cache: bool = True):
        cache_path = os.path.join(self.cache_dir, f"{table_name}.parquet")
        raw_path = os.path.join(self.raw_dir, f"{table_name}.csv")

        if use_cache and os.path.exists(cache_path): # Load parquet file from cache
            logger.info("Loading from cache | table=%s", table_name)
            return pd.read_parquet(cache_path)
        
        logger.info("Loading from CSV | table=%s", table_name)
        df = pd.read_csv(raw_path, engine="pyarrow")
        df.to_parquet(cache_path, index=False) # Save the table to cache as a parquet file
        return df

    def load_all(self, use_cache: bool = True):
        tables =['application_train', 'application_test', 'bureau', 'bureau_balance', 
                  'previous_application', 'installments_payments', 'POS_CASH_balance', 'credit_card_balance']
        return {name: self.load_table(name, use_cache=use_cache) for name in tables}