"""Data Preprocessing Module."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler


def encode_categorical(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Melakukan label encoding pada kolom kategorik."""
    df = df.copy()
    le = LabelEncoder()
    for col in columns:
        df[col] = le.fit_transform(df[col].astype(str))
    return df


def separate_features_target(
    df: pd.DataFrame,
    target_col: str = "Is_Fraud",
    drop_cols: list = None,
) -> tuple:
    """Memisahkan fitur dan target variable.

    Args:
        df: DataFrame input
        target_col: Nama kolom target
        drop_cols: Kolom tambahan yang di-drop (default: Transaction_ID)

    Returns:
        X: DataFrame fitur
        y: Series target
    """
    if drop_cols is None:
        drop_cols = ["Transaction_ID"]
    X = df.drop(columns=[target_col] + drop_cols)
    y = df[target_col]
    return X, y


def standardize_features(X: np.ndarray, scaler: StandardScaler = None) -> tuple:
    """Menstandardisasi fitur numerik menggunakan StandardScaler.

    Args:
        X: Array fitur
        scaler: Scaler yang sudah di-fit (untuk transformasi test data)

    Returns:
        X_scaled: Array yang sudah distandardisasi
        scaler: Object StandardScaler
    """
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
    return X_scaled, scaler


def handle_missing_values(df: pd.DataFrame, strategy: str = "mean") -> pd.DataFrame:
    """Menangani missing values."""
    df = df.copy()
    if strategy == "mean":
        df.fillna(df.mean(numeric_only=True), inplace=True)
    elif strategy == "median":
        df.fillna(df.median(numeric_only=True), inplace=True)
    elif strategy == "drop":
        df.dropna(inplace=True)
    return df
