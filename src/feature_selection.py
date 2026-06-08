"""Feature Selection Module."""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif
from mlxtend.feature_selection import SequentialFeatureSelector


def correlation_based_selection(
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float = 0.05,
) -> list:
    """Memilih fitur berdasarkan korelasi dengan target.

    Args:
        X: DataFrame fitur
        y: Series target
        threshold: Batas minimum korelasi

    Returns:
        List nama fitur yang terpilih
    """
    correlation = X.corrwith(y).abs().sort_values(ascending=False)
    selected = correlation[correlation > threshold].index.tolist()
    return selected


def backward_elimination(
    X: pd.DataFrame,
    y: pd.Series,
    k_features: int = 5,
    cv: int = 5,
    scoring: str = "accuracy",
) -> list:
    """Backward Elimination menggunakan SequentialFeatureSelector.

    Metode ini memulai dari semua fitur, lalu menghilangkan fitur
    satu per satu berdasarkan kontribusi terhadap performa model.

    Args:
        X: DataFrame fitur
        y: Series target
        k_features: Jumlah fitur yang ingin dipertahankan
        cv: Jumlah fold untuk cross-validation
        scoring: Metrik evaluasi

    Returns:
        List nama fitur yang terpilih
    """
    lr = LogisticRegression(max_iter=1000, solver="lbfgs")
    sfs = SequentialFeatureSelector(
        lr,
        k_features=k_features,
        forward=False,
        floating=False,
        scoring=scoring,
        cv=cv,
    )
    sfs.fit(X, y)
    return list(sfs.k_feature_names_)


def select_top_k_features(
    X: pd.DataFrame,
    y: pd.Series,
    k: int = 5,
) -> tuple:
    """Memilih top-K fitur menggunakan SelectKBest (ANOVA F-test).

    Args:
        X: DataFrame fitur
        y: Series target
        k: Jumlah fitur terbaik

    Returns:
        Tuple (X_terpilih, nama_kolom_terpilih)
    """
    selector = SelectKBest(score_func=f_classif, k=k)
    X_selected = selector.fit_transform(X, y)
    selected_indices = selector.get_support(indices=True)
    selected_columns = X.columns[selected_indices].tolist()
    return X[selected_columns], selected_columns
