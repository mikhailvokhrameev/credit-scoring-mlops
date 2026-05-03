import pandas as pd
import gc

def build_bureau_features(bureau_df: pd.DataFrame, bureau_balance_df: pd.DataFrame, bb_cat: list, bureau_cat: list) -> pd.DataFrame:
    """
    Build aggregated bureau and bureau_balance features for each customer.
    
    Args:
        bureau_df: Encoded bureau credit history dataframe
        bureau_balance_df: Encoded bureau monthly balance history dataframe
        bb_cat: List of one-hot encoded columns in bureau_balance_df
        bureau_cat: List of one-hot encoded columns in bureau_df
        
    Returns:
        Aggregated features indexed by SK_ID_CURR
    """
    bureau_df = bureau_df.copy()
    bureau_balance_df = bureau_balance_df.copy()
    
    # Bureau Balance: Aggregate status indicators
    bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}
    for col in bb_cat:
        bb_aggregations[col] = ['mean']
    bb_agg = bureau_balance_df.groupby('SK_ID_BUREAU').agg(bb_aggregations)
    bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])
    bureau_df = bureau_df.join(bb_agg, how='left', on='SK_ID_BUREAU')
    bureau_df.drop(['SK_ID_BUREAU'], axis=1, inplace=True)
    del bureau_balance_df, bb_agg
    gc.collect() # Free up memory after join

    # Bureau: Numerical and categorical aggregations
    
    # Feature engineering
    bureau_df['BUREAU_DEBT_CREDIT_RATIO'] = bureau_df['AMT_CREDIT_SUM_DEBT'] / (bureau_df['AMT_CREDIT_SUM'] + 1e-5)
    
    # Define aggregations for numerical columns
    num_aggregations = {
        'DAYS_CREDIT': ['mean', 'max', 'min', 'var'],
        'DAYS_CREDIT_ENDDATE': ['mean', 'max'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean', 'max'],
        'AMT_CREDIT_SUM': ['mean', 'max', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['mean', 'max', 'sum'],
        'BUREAU_DEBT_CREDIT_RATIO': ['mean', 'max']
    }
    
    # Define aggregations for categorical columns
    cat_aggregations = {}
    for col in bureau_cat:
        cat_aggregations[col] = ['mean']
    
    # Aggregate by customer
    bureau_agg = bureau_df.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    bureau_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])
    
    # Last active credit days
    active = bureau_df[bureau_df['CREDIT_ACTIVE_Active'] == 1]
    active_agg = active.groupby('SK_ID_CURR').agg({'DAYS_CREDIT': ['max']})
    active_agg.columns = pd.Index(['BURO_LAST_ACTIVE_DAYS_CREDIT_MAX'])
    bureau_agg = bureau_agg.join(active_agg, how='left', on='SK_ID_CURR')
    
    del bureau_df
    gc.collect() # Free up memory after join
    
    return bureau_agg
