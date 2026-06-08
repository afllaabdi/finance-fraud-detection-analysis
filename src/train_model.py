"""Model Training Module."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import joblib


# Default hyperparameter grid untuk GridSearchCV
DEFAULT_PARAM_GRID = {
    "n_estimators": [100, 200],
    "max_depth": [5, 10, None],
    "min_samples_split": [2, 5],
}


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> tuple:
    """Memisahkan data menjadi training dan test set.

    Args:
        X: DataFrame fitur
        y: Series target
        test_size: Proporsi data test (0.0 - 1.0)
        random_state: Seed untuk reproduktibilitas
        stratify: Gunakan stratified split (diaktifkan secara default)

    Returns:
        Tuple (X_train, X_test, y_train, y_test)
    """
    stratify_param = y if stratify else None
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_param,
    )


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42,
    **kwargs,
) -> RandomForestClassifier:
    """Melatih Random Forest Classifier.

    Args:
        X_train: Array fitur training
        y_train: Array label training
        random_state: Seed untuk reproduktibilitas
        **kwargs: Hyperparameter tambahan untuk RandomForest

    Returns:
        Object RandomForestClassifier yang sudah di-fit
    """
    model = RandomForestClassifier(random_state=random_state, **kwargs)
    model.fit(X_train, y_train)
    return model


def hyperparameter_tuning(
    X_train: np.ndarray,
    y_train: np.ndarray,
    param_grid: dict = None,
    cv: int = 5,
    scoring: str = "f1",
) -> GridSearchCV:
    """Melakukan hyperparameter tuning menggunakan GridSearchCV.

    Args:
        X_train: Array fitur training
        y_train: Array label training
        param_grid: Dictionary hyperparameter yang diuji
        cv: Jumlah fold untuk cross-validation
        scoring: Metrik evaluasi ('f1', 'accuracy', 'precision', 'recall')

    Returns:
        Object GridSearchCV dengan model terbaik
    """
    if param_grid is None:
        param_grid = DEFAULT_PARAM_GRID

    model = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
    )
    grid_search.fit(X_train, y_train)
    return grid_search


def save_model(model, path: str) -> None:
    """Menyimpan model ke file menggunakan Joblib."""
    joblib.dump(model, path)


def load_model(path: str):
    """Memuat model dari file."""
    return joblib.load(path)
