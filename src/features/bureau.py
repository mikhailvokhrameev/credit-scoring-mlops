import pandas as pd
import gc


def aggregate_bureau_balance(bureau_balance_df: pd.DataFrame, bb_cat: list) -> pd.DataFrame:
    """
    Aggregates monthly bureau balance history into credit-level features.

    This function performs the first level of aggregation, moving from monthly 
    granularity to a single record per credit account (SK_ID_BUREAU).

    Args:
        bureau_balance_df (pd.DataFrame): Raw monthly history of bureau credits.
        bb_cat (list): List of categorical column names (One-Hot Encoded) to aggregate.

    Returns:
        pd.DataFrame: Aggregated features with SK_ID_BUREAU as index.
    """
    bureau_balance_df = bureau_balance_df.copy()
    
    bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}

    for col in bb_cat:
        bb_aggregations[col] = ['mean']

    bb_agg = bureau_balance_df.groupby('SK_ID_BUREAU').agg(bb_aggregations)
    bb_agg.columns = pd.Index([f"{col}_{stat.upper()}" for col, stat in bb_agg.columns.tolist()])

    return bb_agg


def create_bureau_row_features(bureau_df: pd.DataFrame, bb_agg: pd.DataFrame) -> pd.DataFrame:
    """
    Joins aggregated monthly history and creates row-level credit features.

    This function operates at the credit-account level, enriching each loan 
    with its historical summary and calculating financial ratios.

    Args:
        bureau_df (pd.DataFrame): Main bureau data (one row per credit).
        bb_agg (pd.DataFrame): Aggregated monthly data from aggregate_bureau_balance.

    Returns:
        pd.DataFrame: Enriched bureau data with new engineered features.
    """
    bureau_df = bureau_df.copy()

    # Join monthly credit history
    bureau_df = bureau_df.join(bb_agg, how='left', on='SK_ID_BUREAU')
    bureau_df.drop('SK_ID_BUREAU', axis=1, inplace=True)

    # Feature engineering
    bureau_df['BUREAU_DEBT_CREDIT_RATIO'] = (bureau_df['AMT_CREDIT_SUM_DEBT'] / (bureau_df['AMT_CREDIT_SUM'] + 1e-5))

    return bureau_df


def aggregate_bureau_features(bureau_df: pd.DataFrame, bureau_cat: list) -> pd.DataFrame:
    """
    Aggregates all credit-level features into final customer-level features.

    This is the final aggregation step that collapses multiple credits into 
    one record per applicant (SK_ID_CURR). It also handles memory cleanup.

    Args:
        bureau_df (pd.DataFrame): Enriched bureau data from create_bureau_row_features.
        bureau_cat (list): List of categorical bureau column names to aggregate.

    Returns:
        pd.DataFrame: Final feature set with SK_ID_CURR as index.
    """

    # Numerical aggregations
    num_aggregations = {
        'DAYS_CREDIT': ['mean', 'max', 'min', 'var'],
        'DAYS_CREDIT_ENDDATE': ['mean', 'max'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean', 'max'],
        'AMT_CREDIT_SUM': ['mean', 'max', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['mean', 'max', 'sum'],
        'BUREAU_DEBT_CREDIT_RATIO': ['mean', 'max']
    }

    # Categorical aggregations
    cat_aggregations = {col: ['mean'] for col in bureau_cat}

    bureau_agg = bureau_df.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    bureau_agg.columns = pd.Index([f"BURO_{col}_{stat.upper()}" for col, stat in bureau_agg.columns.tolist()])

    # Last active credit
    active = bureau_df[bureau_df['CREDIT_ACTIVE_Active'] == 1]
    active_agg = active.groupby('SK_ID_CURR').agg({'DAYS_CREDIT': ['max']})
    active_agg.columns = pd.Index(['BURO_LAST_ACTIVE_DAYS_CREDIT_MAX'])
    bureau_agg = bureau_agg.join(active_agg, how='left')

    del bureau_df
    gc.collect()

    return bureau_agg


def build_bureau_features(bureau_df: pd.DataFrame, bureau_balance_df: pd.DataFrame, bb_cat: list, bureau_cat: list) -> pd.DataFrame:
    """
    Orchestrates the end-to-end bureau feature engineering pipeline.

    The pipeline follows a bottom-up approach:
    1. Monthly status to Credit level aggregation.
    2. Feature engineering at the individual credit level.
    3. Credit level to Customer level aggregation.

    Args:
        bureau_df (pd.DataFrame): Raw bureau dataset.
        bureau_balance_df (pd.DataFrame): Raw bureau_balance dataset.
        bb_cat (list): Categorical features for bureau_balance.
        bureau_cat (list): Categorical features for bureau.

    Returns:
        pd.DataFrame: Consolidated customer-level features (one row per SK_ID_CURR).
    """
    
    # Monthly to credit
    bb_agg = aggregate_bureau_balance(bureau_balance_df, bb_cat)
    # Credit feature engineering
    bureau_rows = create_bureau_row_features(bureau_df, bb_agg)
    # Customer aggregation
    bureau_features = aggregate_bureau_features(bureau_rows, bureau_cat)

    return bureau_features