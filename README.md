# Finance Fraud Detection Analysis

## 📌 Overview

Finance Fraud Detection Analysis adalah proyek Data Mining yang bertujuan untuk mendeteksi transaksi fraud menggunakan teknik Exploratory Data Analysis (EDA), Feature Selection, Dimensionality Reduction, dan Machine Learning.

Proyek ini mengimplementasikan alur Data Science end-to-end mulai dari eksplorasi data hingga pembangunan model klasifikasi fraud.

---

## 🎯 Objectives

* Memahami karakteristik dataset transaksi keuangan.
* Mengidentifikasi pola yang berkaitan dengan aktivitas fraud.
* Melakukan preprocessing dan feature selection.
* Mengurangi dimensi data menggunakan PCA dan t-SNE.
* Membangun model klasifikasi fraud.
* Mengevaluasi performa model dalam mendeteksi transaksi fraud.

---

## 📂 Dataset Features

| Feature              | Description                                    |
| -------------------- | ---------------------------------------------- |
| Transaction_ID       | Unique transaction identifier                  |
| Amount               | Transaction amount                             |
| Distance_from_Home   | Distance between transaction location and home |
| Device_Type          | Device used for transaction                    |
| IP_Risk_Score        | Risk score of IP address                       |
| Is_Night_Transaction | Night transaction indicator                    |
| Is_Fraud             | Fraud label (Target Variable)                  |

---

## 🔍 Exploratory Data Analysis (EDA)

Analysis performed:

* Dataset overview
* Missing value analysis
* Descriptive statistics
* Fraud distribution analysis
* Correlation analysis
* Data visualization

### Findings

* Dataset structure identified successfully.
* Missing values checked and handled.
* Fraud distribution analyzed.
* Initial insights documented.

---

## ⚙️ Data Preprocessing

Activities:

* Missing value handling
* Categorical encoding
* Data validation
* Feature-target separation

Output:

* Cleaned dataset ready for feature selection.

---

## 🎯 Feature Selection

Methods used:

### Correlation Analysis

Identified features that have significant relationships with fraud labels.

### Backward Elimination

Selected the most relevant features by removing less significant variables.

Output:

* Reduced feature set
* Modeling-ready dataset

---

## 📉 Dimensionality Reduction

### Principal Component Analysis (PCA)

Used to reduce dimensionality while preserving maximum variance.

### t-SNE

Used for data visualization and cluster exploration.

Output:

* PCA projection
* t-SNE visualization

---

## 🤖 Machine Learning Model

### Model Used

* Random Forest Classifier

### Hyperparameter Tuning

GridSearchCV was used to identify the optimal model configuration.

Best Parameters:

```python
{
    'max_depth': 10,
    'min_samples_split': 5,
    'n_estimators': 200
}
```

### Model Persistence

The best model was successfully saved using Joblib:

```text
best_fraud_model.pkl
```

### Prediction Example

```text
[1 0 1 0 0]
```

Where:

* 1 = Fraud Transaction
* 0 = Non-Fraud Transaction

---

## 🔄 Project Workflow

1. Data Loading
2. Exploratory Data Analysis
3. Data Preprocessing
4. Feature Selection
5. PCA Analysis
6. t-SNE Analysis
7. Model Training
8. Model Evaluation
9. Documentation

---

## 🛠️ Technologies

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn
* Joblib
* MLxtend
* Jupyter Notebook

---

## 📈 Current Progress

* [x] Project Setup
* [x] Exploratory Data Analysis
* [x] Data Preprocessing
* [x] Feature Selection
* [x] PCA Analysis
* [x] t-SNE Analysis
* [x] Model Training
* [ ] Model Evaluation
* [ ] Final Documentation

---

## 🚀 Next Steps

* Evaluate model performance
* Generate confusion matrix
* Calculate precision, recall, and F1-score
* Compare model metrics
* Finalize project documentation

---

## 👨‍💻 Author

Aflla Abdi

Data Mining Project – University Coursework
