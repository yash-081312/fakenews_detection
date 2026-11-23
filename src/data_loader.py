"""
data_loader.py
Dataset ingestion, exploratory inspection, duplicate handling, and train/test splitting.
"""

import os
import sys

# Ensure root project directory is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from sklearn.model_selection import train_test_split
from src.preprocessor import clean_text

LABEL_MAP = {"Fake": 1, "Real": 0}
INV_LABEL_MAP = {1: "Fake", 0: "Real"}


def load_raw_data(filepath: str = "data/raw/fake_and_real_news.csv") -> pd.DataFrame:
    """
    Loads raw CSV data from path.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")
    
    df = pd.read_csv(filepath)
    print(f"[DataLoader] Loaded raw data with shape: {df.shape}")
    return df


def preprocess_dataframe(df: pd.DataFrame, text_col: str = "Text", label_col: str = "label") -> pd.DataFrame:
    """
    Cleans raw DataFrame:
    - Drops missing values (if any)
    - Deduplicates records
    - Encodes labels: Fake -> 1, Real -> 0
    - Computes raw text length and token count
    - Applies NLP text preprocessing
    """
    initial_len = len(df)
    
    # 1. Drop NA
    df = df.dropna(subset=[text_col, label_col]).copy()
    
    # 2. Drop duplicates
    df = df.drop_duplicates(subset=[text_col]).reset_index(drop=True)
    dedup_len = len(df)
    print(f"[DataLoader] Dropped {initial_len - dedup_len} duplicate/null rows. Remaining: {dedup_len}")

    # 3. Standardize & map labels
    df['label_name'] = df[label_col].astype(str).str.strip().str.capitalize()
    df['target'] = df['label_name'].map(LABEL_MAP)
    
    # Ensure no unmapped labels
    if df['target'].isnull().any():
        unmapped = df[df['target'].isnull()]['label_name'].unique()
        raise ValueError(f"Unrecognized labels found: {unmapped}")

    # 4. Compute statistics on raw text
    df['raw_char_len'] = df[text_col].apply(lambda x: len(str(x)))
    df['raw_word_count'] = df[text_col].apply(lambda x: len(str(x).split()))

    # 5. Clean text
    print("[DataLoader] Cleaning text through NLP pipeline...")
    df['cleaned_text'] = df[text_col].apply(clean_text)
    df['cleaned_word_count'] = df['cleaned_text'].apply(lambda x: len(x.split()))

    # Remove rows where cleaned text is empty
    valid_mask = df['cleaned_word_count'] > 0
    df = df[valid_mask].reset_index(drop=True)
    print(f"[DataLoader] Final cleaned rows: {len(df)}")
    
    return df


def prepare_and_cache_dataset(
    raw_path: str = "data/raw/fake_and_real_news.csv",
    processed_path: str = "data/processed/cleaned_news.csv",
    force_refresh: bool = False
) -> pd.DataFrame:
    """
    Loads raw data, pre-processes, caches to disk, and returns clean DataFrame.
    """
    if os.path.exists(processed_path) and not force_refresh:
        print(f"[DataLoader] Loading cached processed dataset from: {processed_path}")
        return pd.read_csv(processed_path)

    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    df_raw = load_raw_data(raw_path)
    df_clean = preprocess_dataframe(df_raw)
    df_clean.to_csv(processed_path, index=False)
    print(f"[DataLoader] Processed data saved successfully to: {processed_path}")
    return df_clean


def get_train_test_split(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Performs stratified train/test split.
    Returns: X_train, X_test, y_train, y_test
    """
    X = df['cleaned_text']
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    print(f"[DataLoader] Train split: {len(X_train)} samples | Test split: {len(X_test)} samples")
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    df = prepare_and_cache_dataset(force_refresh=True)
    print("Class distribution:")
    print(df['label_name'].value_counts())
    print(df.head(2))
