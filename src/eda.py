"""Exploratory Data Analysis Module."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(path: str) -> pd.DataFrame:
    """Memuat dataset dari file CSV."""
    return pd.read_csv(path)


def basic_info(df: pd.DataFrame) -> None:
    """Menampilkan informasi dasar dataset."""
    print(f"Shape: {df.shape}")
    print(f"\nData Types:\n{df.dtypes}")
    print(f"\nMissing Values:\n{df.isnull().sum()}")
    print(f"\nDescriptive Stats:\n{df.describe()}")


def fraud_distribution(
    df: pd.DataFrame,
    target_col: str = "Is_Fraud",
) -> pd.Series:
    """Menganalisis distribusi kelas target."""
    dist = df[target_col].value_counts()
    print(f"\nFraud Distribution:\n{dist}")
    print(f"\nProportion (%):\n{dist / len(df) * 100}")
    return dist


def correlation_analysis(df: pd.DataFrame, target_col: str = "Is_Fraud") -> pd.Series:
    """Menghitung dan mengembalikan korelasi dengan target."""
    numeric_df = df.select_dtypes(include=[np.number])
    return numeric_df.corr()[target_col].abs().sort_values(ascending=False)


def plot_fraud_distribution(
    df: pd.DataFrame,
    target_col: str = "Is_Fraud",
    save_path: str = None,
) -> None:
    """Memvisualisasikan distribusi fraud dengan countplot."""
    plt.figure(figsize=(6, 4))
    ax = sns.countplot(x=df[target_col], palette=["#2ecc71", "#e74c3c"])
    ax.set_xticklabels(["Non-Fraud", "Fraud"])
    ax.set_xlabel("Transaction Type")
    ax.set_ylabel("Count")
    ax.set_title("Fraud Distribution")
    for container in ax.containers:
        ax.bar_label(container, fmt="%d")
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_distributions(
    df: pd.DataFrame,
    numeric_cols: list,
    save_path: str = None,
) -> None:
    """Memvisualisasikan distribusi fitur numerik."""
    df[numeric_cols].hist(figsize=(15, 10), bins=30, edgecolor="black")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_correlation_heatmap(
    df: pd.DataFrame,
    save_path: str = None,
) -> None:
    """Memvisualisasikan heatmap korelasi."""
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        df.select_dtypes(include=["number"]).corr(),
        annot=True,
        fmt=".3f",
        cmap="coolwarm",
        linewidths=0.5,
    )
    plt.title("Correlation Heatmap")
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_feature_boxplots(
    df: pd.DataFrame,
    numeric_cols: list,
    save_path: str = None,
) -> None:
    """Memvisualisasikan boxplot untuk deteksi outliers."""
    n_cols = len(numeric_cols)
    n_rows = (n_cols + 2) // 3
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 4 * n_rows))
    axes = axes.flatten()
    for i, col in enumerate(numeric_cols):
        sns.boxplot(x=df[col], ax=axes[i], color="#3498db")
        axes[i].set_title(col)
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
