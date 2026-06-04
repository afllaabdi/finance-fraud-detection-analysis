# Finance Fraud Detection Analysis

## 📌 Overview

Finance Fraud Detection Analysis adalah proyek Data Mining yang bertujuan untuk menganalisis transaksi keuangan dan mengidentifikasi aktivitas fraud menggunakan teknik Feature Selection, Dimensionality Reduction, dan Machine Learning.

Proyek ini mengikuti alur kerja Data Science mulai dari Exploratory Data Analysis (EDA), Data Preprocessing, Feature Selection, hingga pembangunan model klasifikasi fraud.

---

## 🎯 Objectives

* Memahami karakteristik dataset transaksi keuangan.
* Mengidentifikasi pola yang berkaitan dengan aktivitas fraud.
* Melakukan preprocessing data untuk meningkatkan kualitas dataset.
* Memilih fitur yang paling relevan menggunakan Feature Selection.
* Mempersiapkan dataset untuk proses dimensionality reduction dan machine learning.
* Membangun model klasifikasi fraud yang efektif.

---

## 📂 Project Structure

```text
finance-fraud-detection-analysis/
│
├── data/
│   ├── raw_dataset.csv
│   └── processed_dataset.csv
│
├── notebooks/
│   └── finance_fraud_analysis.ipynb
│
├── images/
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 🔍 Exploratory Data Analysis (EDA)

Tahap EDA dilakukan untuk memahami struktur dan karakteristik dataset.

### Analysis Performed

* Dataset Overview
* Missing Value Analysis
* Descriptive Statistics
* Fraud Distribution Analysis
* Data Visualization
* Correlation Analysis

### Key Findings

* Struktur dataset berhasil diidentifikasi.
* Missing value telah dianalisis.
* Distribusi target fraud telah diperiksa.
* Karakteristik fitur numerik dan kategorikal telah dipahami.
* Insight awal untuk preprocessing dan feature selection telah diperoleh.

---

## ⚙️ Data Preprocessing

Tahap preprocessing dilakukan untuk mempersiapkan dataset sebelum modeling.

### Activities

* Data cleaning
* Handling missing values
* Encoding categorical variables
* Feature-target separation
* Dataset validation

### Output

Dataset yang telah dibersihkan dan siap digunakan untuk proses feature selection.

---

## 🎯 Feature Selection

Untuk mengurangi fitur yang tidak relevan dan meningkatkan performa model, digunakan beberapa metode feature selection:

### 1. Correlation Analysis

Mengukur hubungan antara fitur dengan target fraud dan memilih fitur yang memiliki korelasi signifikan.

### 2. Backward Elimination

Menghapus fitur yang kurang berkontribusi secara bertahap hingga diperoleh kombinasi fitur terbaik.

### Output

* Selected Features Dataset
* Reduced Feature Space
* Modeling-ready Dataset

---

## 🔄 Project Workflow

1. Data Loading
2. Exploratory Data Analysis (EDA)
3. Data Preprocessing
4. Feature Selection
5. PCA Analysis
6. t-SNE Analysis
7. Machine Learning Modeling
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
* MLxtend
* Jupyter Notebook

---

## 📈 Current Progress

- [x] Project Setup
- [x] Exploratory Data Analysis
- [x] Data Preprocessing
- [x] Feature Selection
- [x] PCA Analysis
- [x] t-SNE Analysis
- [ ] Model Training
- [ ] Model Evaluation
- [ ] Final Documentation

--

## 🚀 Future Work

* Implement Principal Component Analysis (PCA)
* Implement t-SNE Visualization
* Train Classification Models
* Compare Model Performance
* Optimize Hyperparameters
* Deploy Fraud Detection Pipeline

---

## 👨‍💻 Author

Aflla Abdi

Data Mining Project – University Coursework
