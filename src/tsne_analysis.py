"""t-SNE Analysis Module."""

import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def apply_tsne(
    X: np.ndarray,
    n_components: int = 2,
    perplexity: int = 30,
    random_state: int = 42,
    n_iter: int = 1000,
) -> np.ndarray:
    """Melakukan t-SNE dimensionality reduction.

    Args:
        X: Array fitur yang sudah distandardisasi
        n_components: Jumlah komponen output
        perplexity: Ukuran ketetanggaan (default: 30)
        random_state: Seed untuk reproduktibilitas
        n_iter: Jumlah iterasi optimasi

    Returns:
        Array hasil transformasi t-SNE
    """
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        random_state=random_state,
        n_iter=n_iter,
    )
    return tsne.fit_transform(X)


def plot_tsne_projection(
    X_tsne: np.ndarray,
    y: np.ndarray,
    title: str = "t-SNE Projection - Fraud Detection",
    save_path: str = None,
) -> None:
    """Memvisualisasikan hasil t-SNE projection 2D.

    Args:
        X_tsne: Array hasil t-SNE (n_samples, 2)
        y: Array label target
        title: Judul plot
        save_path: Path untuk menyimpan gambar (opsional)
    """
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(
        X_tsne[:, 0],
        X_tsne[:, 1],
        c=y,
        alpha=0.6,
        cmap="coolwarm",
    )
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.title(title)
    plt.colorbar(scatter, label="Is_Fraud")
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
