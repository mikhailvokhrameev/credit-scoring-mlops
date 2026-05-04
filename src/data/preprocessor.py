import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import KFold
import logging

logger = logging.getLogger(__name__)


def one_hot_encoder(train: pd.DataFrame, test: pd.DataFrame) -> tuple:
    """
    Applies One-Hot Encoding.
    
    Fits the sklearn OneHotEncoder only on the training data. This ensures that 
    the model does not learn about categories existing exclusively in the test set. 
    Unseen categories in the test set are safely ignored (encoded as all zeros).
    
    Args:
        train (pd.DataFrame): Training dataset.
        test (pd.DataFrame): Testing dataset.
        
    Returns:
        tuple: A tuple containing:
            - train (pd.DataFrame): Train dataset with categorical columns replaced by OHE columns.
            - test (pd.DataFrame): Test dataset with categorical columns replaced by OHE columns.
            - new_columns (list): List of the newly created One-Hot Encoded column names.
    """
    train = train.copy()
    test = test.copy()
    
    categorical_columns = [col for col in train.columns if train[col].dtype == 'object']
    
    if not categorical_columns:
        return train, test, []

    # Initialize encoder to ignore unknown categories in the test set
    enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    
    # Fit strictly on train data
    enc.fit(train[categorical_columns])
    
    # Transform both sets
    train_encoded = pd.DataFrame(enc.transform(train[categorical_columns]), columns=enc.get_feature_names_out(categorical_columns), index=train.index)
    test_encoded = pd.DataFrame(enc.transform(test[categorical_columns]), columns=enc.get_feature_names_out(categorical_columns), index=test.index)
    
    # Replace original categorical columns with encoded ones
    train = pd.concat([train.drop(columns=categorical_columns), train_encoded], axis=1)
    test = pd.concat([test.drop(columns=categorical_columns), test_encoded], axis=1)
    
    new_columns = list(enc.get_feature_names_out(categorical_columns))
    
    return train, test, new_columns


def reduce_cardinality(train: pd.DataFrame, test: pd.DataFrame, cardinality_threshold: int = 10) -> tuple:
    """
    Reduces the cardinality of categorical features to prevent overfitting.
    
    To avoid data leakage, the top categories are determined exclusively from 
    the training dataset. Both train and test datasets are then transformed using 
    this train-derived mapping. Rare categories are grouped into 'Other'.
    
    Args:
        train (pd.DataFrame): Training dataset.
        test (pd.DataFrame): Testing dataset.
        cardinality_threshold (int): Number of top categories to keep. Default is 10.
        
    Returns:
        tuple: (train_transformed, test_transformed, modified_cols)
    """
    train = train.copy()
    test = test.copy()
    modified_cols = []
    
    categorical_columns = [col for col in train.columns if train[col].dtype == 'object']
    
    for col in categorical_columns:
        n_unique_train = train[col].nunique()
        
        if n_unique_train > cardinality_threshold:
            # Determine top categories based on train only
            top_categories = train[col].value_counts().head(cardinality_threshold).index.tolist()
            
            # Apply transformation to both datasets
            train[col] = train[col].apply(lambda x: x if x in top_categories else 'Other')
            test[col] = test[col].apply(lambda x: x if x in top_categories else 'Other')
            
            modified_cols.append(col)
            logger.info(f"Reduced cardinality for {col}: kept top {cardinality_threshold} from train")
            
    return train, test, modified_cols


def target_encode_cv(train: pd.DataFrame, test: pd.DataFrame, col: str, target: pd.Series, n_splits: int = 5) -> tuple:
    """
    Applies K-Fold Target Encoding to a categorical feature.
    
    Prevents the model from memorizing the target variable by using Out-of-Fold (OOF) 
    estimations for the training set. The mapping applied to the test set is derived 
    from the global means of the entire training set. Missing values are filled with 
    the global target mean.
    
    Args:
        train (pd.DataFrame): Training dataset.
        test (pd.DataFrame): Testing dataset.
        col (str): The name of the categorical column to encode.
        target (pd.Series): The target variable series aligned with the train index.
        n_splits (int): Number of folds for cross-validation. Default is 5.
        
    Returns:
        tuple: (train_encoded_series, test_encoded_series)
    """
    train = train.copy()
    test = test.copy()
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    global_mean = target.mean()
    train_encoded = pd.Series(index=train.index, dtype=float)
    
    # OOF encoding for train data
    for train_idx, val_idx in kf.split(train):
        X_train, X_val = train.iloc[train_idx], train.iloc[val_idx]
        y_train = target.iloc[train_idx]
        
        means = X_train.groupby(col)[target.name].mean()
        train_encoded.iloc[val_idx] = X_val[col].map(means)
        
    train_encoded.fillna(global_mean, inplace=True)
    
    # Global encoding for test data based entirely on train stats
    final_means = train.groupby(col)[target.name].mean()
    test_encoded = test[col].map(final_means).fillna(global_mean)
    
    return train_encoded, test_encoded


def process_side_table(df, train_ids, preprocessor, table_name):
    """
    Cleans and encodes child tables while strictly preventing data leakage.

    This utility manages the end-to-end preprocessing flow for auxiliary tables 
    (e.g., Bureau, Previous Applications). It ensures that statistical 
    transformations�such as cardinality reduction and One-Hot Encoding�are 
    informed solely by the training distribution.

    The process follows these steps:
    1. Fixes domain-specific anomalies using the provided stateless preprocessor.
    2. Splits the table into training and testing segments based on the 
        'SK_ID_CURR' identifiers present in the main application training set.
    3. Reduces categorical cardinality by grouping rare categories into 
        an 'Other' label, fitting only on the training segment.
    4. Applies One-Hot Encoding, ensuring the encoder is fitted strictly on 
        training data to prevent the model from learning test-only categories.

    Args:
        df (pd.DataFrame): The raw child table to process.
        train_ids (np.array or list): List of 'SK_ID_CURR' identifiers belonging 
            to the training set.
        preprocessor (object): A preprocessor instance (e.g., PreviousPreprocessor) 
            containing a `fix_anomalies` method.
        table_name (str): Name of the table for logging purposes.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: The processed and recombined (train + test) DataFrame.
            - list: Names of the newly generated categorical columns.
    """
    logger.info(f"Preprocessing {table_name}...")
        
    # Anomalies fix
    df = preprocessor.fix_anomalies(df)

    # If not SK_ID_CURR
    if 'SK_ID_CURR' not in df.columns:
        raise KeyError(f"'SK_ID_CURR' missing in {table_name}")

    # Split into train and test sets for training encoders
    train_part = df[df['SK_ID_CURR'].isin(train_ids)]
    test_part = df[~df['SK_ID_CURR'].isin(train_ids)]

    # Reduce cardinality
    train_part, test_part, _ = reduce_cardinality(train_part, test_part)
    # One-Hot Encoding
    train_part, test_part, cat_cols = one_hot_encoder(train_part, test_part)

    return (pd.concat([train_part, test_part], ignore_index=True), cat_cols)


class ApplicationPreprocessor:
    """Preprocessor for the main application dataset"""
    def __init__(self):
        self.label_encoders = {}


    def fix_anomalies(self, df: pd.DataFrame):
        """
        Cleans anomalous and artifact values in a stateless manner.
        
        This method operates independently on single datasets (train or test) 
        without introducing leakage since it handles known structural data errors 
        (like specific integer codes representing missing values).
        
        Args:
            df (pd.DataFrame): Input dataset.
            
        Returns:
            pd.DataFrame: Cleaned dataset.
        """
        df = df.copy()

        if 'DAYS_EMPLOYED' in df.columns:
            df['DAYS_EMPLOYED_ANOM'] = (df['DAYS_EMPLOYED'] == 365243).astype(np.int8) # Create anomaly flag
            df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].replace(365243, np.nan)
        
        if 'DAYS_LAST_PHONE_CHANGE' in df.columns:
            df['DAYS_LAST_PHONE_CHANGE'] = df['DAYS_LAST_PHONE_CHANGE'].replace(0, np.nan)
            
        if 'DAYS_BIRTH' in df.columns:
            df['DAYS_BIRTH'] = df['DAYS_BIRTH'].abs() # Domain correction
            
        df.replace({'XNA': np.nan, 'Unknown': np.nan}, inplace=True) # Replace artifacts with NaN
        return df


    def fit_transform_categorical(self, train: pd.DataFrame, test: pd.DataFrame):
        """
        Applies Label Encoding strictly fitting on train data.
        
        Maps string categories to integers. To prevent leakage, the encoder is fitted 
        only on the train set. Unseen categories in the test set are explicitly mapped 
        to a safe 'UNKNOWN' class to avoid transform errors.
        
        Args:
            train (pd.DataFrame): Training dataset.
            test (pd.DataFrame): Testing dataset.
            
        Returns:
            tuple: (train_encoded, test_encoded)
        """
        train = train.copy()
        test = test.copy()
        
        cat_cols = [col for col in train.columns if train[col].dtype == 'object']
        
        for col in cat_cols:
            le = LabelEncoder()
            le.fit(train[col])
            train[col] = le.transform(train[col])

            # Handle unseen categories
            test[col] = test[col].apply(lambda x: x if x in le.classes_ else "UNKNOWN")
            le.classes_ = np.append(le.classes_, "UNKNOWN")
            test[col] = le.transform(test[col])
            
            self.label_encoders[col] = le
            
        logger.info(f"Categorical features label-encoded: {len(cat_cols)} columns")
        return train, test
    

class PreviousPreprocessor:
    """Preprocessor for the previous applications dataset"""
    def __init__(self):
        pass
    
    def fix_anomalies(self, df: pd.DataFrame):
        """
        Cleans anomalous values (e.g., placeholder 365243 for missing days) 
        in previous application records. Stateless and leakage-free.
        
        Args:
            df (pd.DataFrame): Input previous applications dataset.
            
        Returns:
            pd.DataFrame: Cleaned dataset.
        """
        df = df.copy()
        
        if 'DAYS_FIRST_DRAWING' in df.columns:
            df['DAYS_FIRST_DRAWING'] = df['DAYS_FIRST_DRAWING'].replace(365243, np.nan)
        
        if 'DAYS_FIRST_DUE' in df.columns: 
            df['DAYS_FIRST_DUE'] = df['DAYS_FIRST_DUE'].replace(365243, np.nan)
        
        if 'DAYS_LAST_DUE_1ST_VERSION' in df.columns:
            df['DAYS_LAST_DUE_1ST_VERSION'] = df['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan)
            
        if 'DAYS_LAST_DUE' in df.columns:
            df['DAYS_LAST_DUE'] = df['DAYS_LAST_DUE'].replace(365243, np.nan)
            
        if 'DAYS_TERMINATION' in df.columns:
            df['DAYS_TERMINATION'] = df['DAYS_TERMINATION'].replace(365243, np.nan)
        
        return df
    
class POSCashPreprocessor:
    """Preprocessor for POS_CASH_balance dataset"""
    
    def fix_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Stateless anomaly fixing for POS Cash"""
        df = df.copy()
        return df


class CreditCardPreprocessor:
    """Preprocessor for credit_card_balance dataset"""
    
    def fix_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Stateless anomaly fixing for Credit Card balance"""
        df = df.copy()
        # Previous loan ID is not needed for aggregation by customer
        if 'SK_ID_PREV' in df.columns:
            df.drop(columns=['SK_ID_PREV'], inplace=True)
        return df


class InstallmentsPreprocessor:
    """Preprocessor for installments_payments dataset"""
    
    def fix_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Stateless anomaly fixing for Installments"""
        df = df.copy()
        return df
    
    
class PassThroughPreprocessor:
    """Fallback preprocessor for tables without specific anomaly handling"""
    def fix_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.copy()
    
    