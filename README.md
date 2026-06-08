# Finance Fraud Detection Analysis

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> Proyek Data Mining end-to-end untuk deteksi transaksi fraud menggunakan teknik Machine Learning, PCA, dan t-SNE.

---

## Daftar Isi

- [Gambaran Proyek](#gambaran-proyek)
- [Fitur Dataset](#fitur-dataset)
- [Metodologi](#metodologi)
- [Workflow](#workflow)
- [Hasil Model](#hasil-model)
- [Pemasangan](#pemasangan)
- [Struktur Folder](#struktur-folder)

---

## Gambaran Proyek

**Finance Fraud Detection Analysis** adalah proyek Data Mining untuk mendeteksi transaksi fraud menggunakan Random Forest Classifier dengan optimasi GridSearchCV.

### Hasil Utama

| Metrik | Skor |
|--------|------|
| **Accuracy** | 98.27% |
| **Precision** | 93.15% |
| **Recall** | 92.64% |
| **F1 Score** | 92.90% |

---

## Fitur Dataset

Dataset: **15.000 transaksi**, **12 fitur**

| Fitur | Tipe | Deskripsi |
|-------|------|-----------|
| Transaction_ID | String | Identifier transaksi |
| Timestamp | Datetime | Waktu transaksi |
| Customer_ID | String | Identifier pelanggan |
| Amount | Float | Nominal transaksi |
| Merchant_Category | String | Kategori merchant |
| Distance_from_Home | Float | Jarak dari rumah (0-5) |
| Device_Type | String | Android / Web Browser / iOS |
| IP_Risk_Score | Float | Skor risiko IP (0-1) |
| Avg_Spending_Habit | Float | Rata-rata belanja |
| Is_Weekend | Integer | Indikator weekend (0/1) |
| Is_Night_Transaction | Integer | Indikator malam (0/1) |
| Is_Fraud | Integer | **Target** (0/1) |

### Distribusi Target

```
Non-Fraud: 13.165 (87.77%)
Fraud:      1.835 (12.23%)
```

---

## Metodologi

### 1. EDA
- Struktur dataset, missing values, statistik deskriptif
- Korelasi fitur: Amount (0.56), Is_Night_Transaction (0.31), Distance_from_Home (0.22)
- Deteksi outliers pada Amount dan Distance_from_Home

### 2. Preprocessing
- Label Encoding pada kolom kategorik
- Feature-Target Separation
- StandardScaler untuk normalisasi

### 3. Feature Selection
- **Correlation Analysis**: Filter fitur dengan korelasi > 0.05
- **Backward Elimination**: SequentialFeatureSelector dengan LogisticRegression

**Fitur Terpilih:**
```
['Transaction_ID', 'Amount', 'Distance_from_Home',
 'Device_Type', 'IP_Risk_Score', 'Is_Night_Transaction']
```

### 4. Dimensionality Reduction
- **PCA** (`n_components=2`): Variance explained 33.78%
- **t-SNE** (`perplexity=30`): Visualisasi cluster

### 5. Model Training
- Random Forest Classifier
- Default accuracy: 98.07%

### 6. Hyperparameter Tuning
- GridSearchCV dengan 5-fold CV
- Scoring: F1 (untuk dataset tidak seimbang)

**Best Parameters:**
```python
{'max_depth': 10, 'min_samples_split': 5, 'n_estimators': 200}
```

### 7. Evaluasi

| Metrik | Nilai |
|--------|-------|
| TN | 2.608 |
| FP | 25 |
| FN | 27 |
| TP | 340 |

- False Positive Rate: 0.95%
- False Negative Rate: 7.36%

---

## Workflow

```
Data Loading → EDA → Preprocessing → Feature Selection
                                          ↓
                     PCA ← Dimensionality Reduction → t-SNE
                                          ↓
                              Model Training (Random Forest)
                                          ↓
                              Hyperparameter Tuning (GridSearchCV)
                                          ↓
                              Model Evaluation & Persistence
```

---

## Hasil Model

### Confusion Matrix

```
              Predicted
              Non-Fraud   Fraud
Non-Fraud        2608       25
Fraud              27      340
```

### Interpretasi Metrik

| Metrik | Formula | Hasil |
|--------|---------|-------|
| Accuracy | (TN+TP)/Total | 98.27% |
| Precision | TP/(TP+FP) | 93.15% |
| Recall | TP/(TP+FN) | 92.64% |
| F1 Score | 2×P×R/(P+R) | 92.90% |

---

## Pemasangan

```bash
# Install dependencies
pip install -r requirements.txt

# Jalankan notebook
cd notebooks
jupyter notebook finance_fraud_analysis.ipynb
```

### Prediksi dengan Model

```python
import joblib
import pandas as pd

model = joblib.load('models/best_fraud_model.pkl')

new_transaction = pd.DataFrame({
    'Amount': [1500.00],
    'Distance_from_Home': [0.25],
    'Device_Type': [1],
    'IP_Risk_Score': [0.85],
    'Is_Night_Transaction': [1]
})

prediction = model.predict(new_transaction)
print(f"Prediksi: {'Fraud' if prediction[0] == 1 else 'Non-Fraud'}")
```

---

## Struktur Folder

```
finance-fraud-detection-analysis/
|
├── data/
│   ├── raw/
│   │   └── finance_fraud_data.csv      # Dataset asli
│   └── processed/
│       ├── processed_dataset.csv        # Dataset siap modeling
│       ├── pca_dataset.csv             # Hasil PCA
│       └── tsne_dataset.csv            # Hasil t-SNE
│
├── models/
│   └── best_fraud_model.pkl            # Model final
│
├── notebooks/
│   └── finance_fraud_analysis.ipynb     # Notebook analisis
│
├── src/                                # Modul Python reusable
│   ├── eda.py
│   ├── preprocessing.py
│   ├── feature_selection.py
│   ├── pca_analysis.py
│   ├── tsne_analysis.py
│   ├── train_model.py
│   ├── evaluate.py
│   └── predict.py
│
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

---

## Lisensi

MIT License - lihat [LICENSE](LICENSE)

---

**Author:** Aflla Abdi
**Mata Kuliah:** Data Mining