"""PCA Analysis Module."""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def apply_pca(
    X: np.ndarray,
    n_components: int = 2,
    random_state: int = 42,
) -> tuple:
    """Melakukan PCA dan mengembalikan hasil transformasi.

    Args:
        X: Array fitur yang sudah distandardisasi
        n_components: Jumlah komponen utama
        random_state: Seed untuk reproduktibilitas

    Returns:
        Tuple (X_pca, pca_object)
    """
    pca = PCA(n_components=n_components, random_state=random_state)
    X_pca = pca.fit_transform(X)
    return X_pca, pca


def get_variance_explained(pca) -> pd.DataFrame:
    """Mengembalikan DataFrame variance yang dijelaskan tiap komponen.

    Args:
        pca: Object PCA yang sudah di-fit

    Returns:
        DataFrame dengan kolom Component, Variance_Ratio, Cumulative
    """
    variance_ratio = pca.explained_variance_ratio_
    cumulative = np.cumsum(variance_ratio)
    return pd.DataFrame({
        "Component": [f"PC{i+1}" for i in range(len(variance_ratio))],
        "Variance_Ratio": variance_ratio,
        "Cumulative": cumulative,
    })


def plot_pca_projection(
    X_pca: np.ndarray,
    y: np.ndarray,
    save_path: str = None,
) -> None:
    """Memvisualisasikan hasil PCA projection 2D.

    Args:
        X_pca: Array hasil PCA (n_samples, 2)
        y: Array label target
        save_path: Path untuk menyimpan gambar (opsional)
    """
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=y,
        alpha=0.6,
        cmap="coolwarm",
    )
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA Projection - Fraud Detection")
    plt.colorbar(scatter, label="Is_Fraud")
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
