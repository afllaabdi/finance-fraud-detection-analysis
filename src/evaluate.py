"""Model Evaluation Module."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Evaluasi model dan mengembalikan dictionary metrics.

    Args:
        y_true: Label sebenarnya
        y_pred: Label prediksi

    Returns:
        Dictionary dengan metric dan skor
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
    }


def print_evaluation_report(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Menampilkan laporan evaluasi lengkap."""
    print("=" * 50)
    print("        MODEL EVALUATION REPORT")
    print("=" * 50)
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred):.4f}")
    print(f"F1 Score:  {f1_score(y_true, y_pred):.4f}")
    print("=" * 50)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["Non-Fraud", "Fraud"]))


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list = None,
    save_path: str = None,
) -> None:
    """Memvisualisasikan confusion matrix dengan heatmap.

    Args:
        y_true: Label sebenarnya
        y_pred: Label prediksi
        labels: Nama label untuk axis
        save_path: Path untuk menyimpan gambar (opsional)
    """
    cm = confusion_matrix(y_true, y_pred)
    if labels is None:
        labels = ["Non-Fraud", "Fraud"]

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        annot_kws={"size": 14},
    )
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("Actual", fontsize=12)
    plt.title("Confusion Matrix - Fraud Detection", fontsize=14)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    save_path: str = None,
) -> float:
    """Memvisualisasikan ROC Curve dan mengembalikan AUC score.

    Args:
        y_true: Label sebenarnya
        y_proba: Array probabilitas kelas positif (fraud)
        save_path: Path untuk menyimpan gambar (opsional)

    Returns:
        Nilai AUC score
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC (AUC = {auc:.4f})", linewidth=2)
    plt.plot([0, 1], [0, 1], "k--", label="Random Classifier")
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curve - Fraud Detection", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return auc


def create_metrics_dataframe(results: dict) -> pd.DataFrame:
    """Membuat DataFrame dari hasil evaluasi."""
    return pd.DataFrame({
        "Metric": list(results.keys()),
        "Score": list(results.values()),
    })
