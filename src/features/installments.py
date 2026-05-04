import pandas as pd
import gc


def create_installments_row_features(ins_df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates row-level features for individual installment payments.

    Calculates payment fractions, absolute differences, and dynamic 
    delay metrics (Days Past Due and Days Before Due).

    Args:
        ins_df (pd.DataFrame): Raw installments_payments data.

    Returns:
        pd.DataFrame: DataFrame enriched with row-level payment behaviors.
    """
    ins_df = ins_df.copy()
    
    # Epsilon added to prevent ZeroDivisionError
    ins_df['PAYMENT_PERC'] = ins_df['AMT_PAYMENT'] / (ins_df['AMT_INSTALMENT'] + 1e-5)
    ins_df['PAYMENT_DIFF'] = ins_df['AMT_INSTALMENT'] - ins_df['AMT_PAYMENT']
    
    # Days Past Due (DPD) and Days Before Due (DBD)
    ins_df['DPD'] = (ins_df['DAYS_ENTRY_PAYMENT'] - ins_df['DAYS_INSTALMENT']).clip(lower=0)
    ins_df['DBD'] = (ins_df['DAYS_INSTALMENT'] - ins_df['DAYS_ENTRY_PAYMENT']).clip(lower=0)
    
    return ins_df


def aggregate_installments_features(ins_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates installment features at the customer level (SK_ID_CURR).

    Computes global statistics across all payments and extracts localized 
    behavioral trends using time-window aggregations (last 365 and 1000 days).

    Args:
        ins_df (pd.DataFrame): Enriched installments data.

    Returns:
        pd.DataFrame: Customer-level installments features.
    """
    aggregations = {
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'DPD': ['max', 'mean', 'sum'],
        'DBD': ['max', 'mean', 'sum'],
        'PAYMENT_PERC': ['max', 'mean', 'var'],
        'PAYMENT_DIFF': ['max', 'mean', 'sum', 'var'],
        'AMT_INSTALMENT': ['max', 'mean', 'sum'],
        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
        'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum']
    }
    
    # Global Aggregation
    ins_agg = ins_df.groupby('SK_ID_CURR').agg(aggregations)
    ins_agg.columns = pd.Index([f"INS_{col}_{stat.upper()}" for col, stat in ins_agg.columns.tolist()])
    ins_agg['INS_COUNT'] = ins_df.groupby('SK_ID_CURR').size()

    # Time Window: Last 1000 days
    ins_1000 = ins_df[ins_df['DAYS_INSTALMENT'] > -1000]
    ins_1000_agg = ins_1000.groupby('SK_ID_CURR').agg({'PAYMENT_DIFF': ['mean']})
    ins_1000_agg.columns = pd.Index(['INS_1000_PAYMENT_DIFF_MEAN'])
    ins_agg = ins_agg.join(ins_1000_agg, how='left')

    # Time Window: Last 365 days
    ins_365 = ins_df[ins_df['DAYS_INSTALMENT'] > -365]
    ins_365_agg = ins_365.groupby('SK_ID_CURR').agg({'DPD': ['sum'], 'PAYMENT_DIFF': ['mean']})
    ins_365_agg.columns = pd.Index(['INS_365_DPD_SUM', 'INS_365_PAYMENT_DIFF_MEAN'])
    ins_agg = ins_agg.join(ins_365_agg, how='left')

    del ins_df
    gc.collect()
    
    return ins_agg


def build_installments_features(ins_df: pd.DataFrame) -> pd.DataFrame:
    """
    Orchestrates the feature engineering pipeline for installments payments.

    Args:
        ins_df (pd.DataFrame): Raw installments data.

    Returns:
        pd.DataFrame: Final consolidated features per customer.
    """
    ins_rows = create_installments_row_features(ins_df)
    ins_features = aggregate_installments_features(ins_rows)
    return ins_features