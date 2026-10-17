import gc

import pandas as pd


def create_previous_row_features(prev_df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates row-level engineered features for individual previous applications.

    This function calculates financial ratios and differences between requested 
    and granted amounts. These features capture the client's behavior and the 
    bank's decisions for each historical loan application.

    Calculated features include:
    - Asked vs. Granted amount difference
    - Application-to-Credit ratio
    - Downpayment-to-Credit ratio
    - Annuity-to-Credit ratio

    Args:
        prev_df (pd.DataFrame): Raw previous applications data.

    Returns:
        pd.DataFrame: DataFrame with additional row-specific features.
    """
    prev_df = prev_df.copy()

    # Credit behavior features
    prev_df['ASKED_AMT_ID_CURR_DIFF'] = (prev_df['AMT_APPLICATION'] - prev_df['AMT_CREDIT'])
    prev_df['APPLICATION_CREDIT_RATIO'] = (prev_df['AMT_APPLICATION'] / (prev_df['AMT_CREDIT'] + 1e-5))
    prev_df['DOWNPAYMENT_CREDIT_RATIO'] = (prev_df['AMT_DOWN_PAYMENT'] / (prev_df['AMT_CREDIT'] + 1e-5))
    prev_df['ANNUITY_CREDIT_RATIO'] = (prev_df['AMT_ANNUITY'] / (prev_df['AMT_CREDIT'] + 1e-5))

    return prev_df


def aggregate_previous_features(prev_df: pd.DataFrame, prev_cat: list) -> pd.DataFrame:
    """
    Aggregates previous application features and captures the most recent loan status.

    This function reduces the granularity from multiple applications per customer 
    to a single record per SK_ID_CURR. It combines statistical aggregations 
    with snapshots of the customer's very last application.

    Args:
        prev_df (pd.DataFrame): Enriched previous applications data.
        prev_cat (list): List of categorical columns (One-Hot Encoded) to aggregate.

    Returns:
        pd.DataFrame: Customer-level features with SK_ID_CURR as index.
    """

    # Numerical aggregations
    num_aggregations = {
        'AMT_ANNUITY': ['mean', 'max'],
        'AMT_APPLICATION': ['mean', 'max'],
        'AMT_CREDIT': ['mean', 'max'],
        'AMT_DOWN_PAYMENT': ['mean', 'max'],
        'AMT_GOODS_PRICE': ['mean', 'max'],
        'HOUR_APPR_PROCESS_START': ['mean', 'max'],
        'RATE_DOWN_PAYMENT': ['mean', 'max'],
        'DAYS_DECISION': ['mean', 'max'],
        'CNT_PAYMENT': ['mean', 'sum'],
        'ASKED_AMT_ID_CURR_DIFF': ['mean', 'max', 'sum'],
        'APPLICATION_CREDIT_RATIO': ['mean', 'max'],
        'DOWNPAYMENT_CREDIT_RATIO': ['mean', 'max'],
        'ANNUITY_CREDIT_RATIO': ['mean', 'max'],
    }

    # One-hot categorical aggregations
    cat_aggregations = {col: ['mean'] for col in prev_cat}

    prev_agg = prev_df.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    prev_agg.columns = pd.Index([f"PREV_{col}_{stat.upper()}" for col, stat in prev_agg.columns.tolist()])

    # Last application info
    last_app = (prev_df.sort_values('DAYS_DECISION', ascending=False).groupby('SK_ID_CURR').first())
    last_cols = [c for c in last_app.columns if 'NAME_CONTRACT_STATUS' in c or 'PRODUCT_COMBINATION' in c]
    last_app = last_app[last_cols]
    last_app.columns = ['PREV_LAST_' + c for c in last_app.columns]
    prev_agg = prev_agg.join(last_app, how='left')

    gc.collect()
    return prev_agg


def build_previous_features(prev_df: pd.DataFrame, prev_cat: list) -> pd.DataFrame:
    """
    Orchestrates the feature engineering pipeline for previous applications data.

    Coordinates the transformation from raw application records to high-level 
    customer summaries by first engineering row-level features and then 
    performing multi-level aggregations.

    Args:
        prev_df (pd.DataFrame): Raw previous applications dataset.
        prev_cat (list): List of categorical column names.

    Returns:
        pd.DataFrame: Final consolidated features per customer (SK_ID_CURR).
    """

    # Loan-level feature engineering
    prev_rows = create_previous_row_features(prev_df)
    # Customer aggregation
    prev_features = aggregate_previous_features(prev_rows, prev_cat)

    return prev_features