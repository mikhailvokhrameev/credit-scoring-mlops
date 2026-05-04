import pandas as pd
import gc


def aggregate_credit_card_features(cc_df: pd.DataFrame, cc_cat: list) -> pd.DataFrame:
    """
    Aggregates credit card balance history into customer-level metrics.

    Applies a broad statistical aggregation across all available numerical 
    columns and calculates categorical means.

    Args:
        cc_df (pd.DataFrame): Preprocessed and Encoded credit card data.
        cc_cat (list): List of OHE categorical columns.

    Returns:
        pd.DataFrame: Aggregated customer-level credit card features.
    """
    
    # Exclude non-numeric/identifier columns before computing general stats
    exclude_cols = ['SK_ID_CURR'] + cc_cat
    num_cols = [c for c in cc_df.columns if c not in exclude_cols]
    
    # Define aggregation dictionary
    aggregations = {col: ['min', 'max', 'mean', 'sum', 'var'] for col in num_cols}
    for col in cc_cat:
        aggregations[col] = ['mean']
        
    cc_agg = cc_df.groupby('SK_ID_CURR').agg(aggregations)
    cc_agg.columns = pd.Index([f"CC_{col}_{stat.upper()}" for col, stat in cc_agg.columns.tolist()])
    
    del cc_df
    gc.collect()
    
    return cc_agg


def create_credit_card_customer_features(cc_agg: pd.DataFrame) -> pd.DataFrame:
    """
    Creates derived features at the customer level after aggregation.

    Calculates holistic metrics such as the average credit limit utilization.

    Args:
        cc_agg (pd.DataFrame): Aggregated credit card features.

    Returns:
        pd.DataFrame: DataFrame enriched with customer-level ratios.
    """
    cc_agg = cc_agg.copy()
    
    # Calculate average utilization of the credit limit
    if 'CC_AMT_BALANCE_MEAN' in cc_agg.columns and 'CC_AMT_CREDIT_LIMIT_ACTUAL_MEAN' in cc_agg.columns:
        cc_agg['CC_LIMIT_USE_MEAN'] = (cc_agg['CC_AMT_BALANCE_MEAN'] / (cc_agg['CC_AMT_CREDIT_LIMIT_ACTUAL_MEAN'] + 1e-5))
        
    return cc_agg


def build_credit_card_features(cc_df: pd.DataFrame, cc_cat: list) -> pd.DataFrame:
    """
    Orchestrates the feature engineering pipeline for Credit Card balance.

    Args:
        cc_df (pd.DataFrame): Preprocessed Credit Card data.
        cc_cat (list): List of OHE categorical columns.

    Returns:
        pd.DataFrame: Final consolidated features per customer.
    """
    cc_agg = aggregate_credit_card_features(cc_df, cc_cat)
    cc_features = create_credit_card_customer_features(cc_agg)
    return cc_features