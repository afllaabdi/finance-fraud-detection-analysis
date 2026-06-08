"""Prediction Module."""

import pandas as pd
import numpy as np
import joblib


# Fitur yang digunakan model (sesuai dengan hasil Feature Selection)
FEATURE_COLUMNS = [
    "Amount",
    "Distance_from_Home",
    "Device_Type",
    "IP_Risk_Score",
    "Is_Night_Transaction",
]


def load_model(model_path: str):
    """Memuat trained model dari file."""
    return joblib.load(model_path)


def predict(model, X: pd.DataFrame) -> np.ndarray:
    """Melakukan prediksi kelas."""
    return model.predict(X)


def predict_proba(model, X: pd.DataFrame) -> np.ndarray:
    """Melakukan prediksi probabilitas."""
    return model.predict_proba(X)


def predict_transaction(
    model,
    amount: float,
    distance_from_home: float,
    device_type: int,
    ip_risk_score: float,
    is_night_transaction: int,
) -> dict:
    """Melakukan prediksi untuk satu transaksi baru.

    Args:
        model: Object model yang sudah di-fit
        amount: Nominal transaksi
        distance_from_home: Jarak dari rumah (normalisasi)
        device_type: Tipe perangkat (0/1)
        ip_risk_score: Skor risiko IP (0.0 - 1.0)
        is_night_transaction: Transaksi malam hari (0/1)

    Returns:
        Dictionary dengan hasil prediksi
    """
    data = pd.DataFrame({
        "Amount": [amount],
        "Distance_from_Home": [distance_from_home],
        "Device_Type": [device_type],
        "IP_Risk_Score": [ip_risk_score],
        "Is_Night_Transaction": [is_night_transaction],
    })

    prediction = predict(model, data)[0]
    probability = predict_proba(model, data)[0]

    return {
        "prediction": "Fraud" if prediction == 1 else "Non-Fraud",
        "is_fraud": bool(prediction),
        "confidence": float(max(probability)),
        "fraud_probability": float(probability[1]),
        "non_fraud_probability": float(probability[0]),
    }


def predict_from_csv(model_path: str, csv_path: str, output_path: str = None) -> pd.DataFrame:
    """Melakukan prediksi dari file CSV.

    Args:
        model_path: Path ke file model .pkl
        csv_path: Path ke file CSV input
        output_path: Path untuk menyimpan hasil (opsional)

    Returns:
        DataFrame dengan hasil prediksi
    """
    model = load_model(model_path)
    df = pd.read_csv(csv_path)

    # Pilih hanya kolom fitur yang diperlukan
    X = df[FEATURE_COLUMNS]

    df["Is_Fraud_Predicted"] = predict(model, X)
    df["Fraud_Probability"] = predict_proba(model, X)[:, 1]

    if output_path:
        df.to_csv(output_path, index=False)

    return df
