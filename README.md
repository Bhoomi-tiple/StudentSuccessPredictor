# 🎓 Student Success Predictor
### AI/ML-Based Student Academic Performance Prediction & Decision-Support System

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![Streamlit App](https://img.shields.io/badge/Frontend-Streamlit-red.svg)](https://streamlit.io/)
[![Status](https://img.shields.io/badge/Build-Passing-brightgreen.svg)]()

---

## 📌 Executive Summary
**Student Success Predictor** is an end-to-end Machine Learning and Expert Decision-Support System built using the **UCI Student Performance Dataset (`student-mat.csv`)**. The system addresses a critical challenge in modern education: early detection of academically vulnerable students, accurate grade forecasting, learning pattern discovery via unsupervised clustering, and actionable pedagogical intervention through an expert rule-based engine.

The project strictly follows ML best practices with **stratified train-test splitting (80/20)**, **zero data leakage**, **standardized feature scaling**, **one-hot encoding**, and empirical benchmark comparisons across multiple paradigms.

---

## 🚀 Key Features & Syllabus Concepts Implemented

1. **Continuous Grade Prediction (Regression)**:
   - **Linear Regression (Ordinary Least Squares)** and **Gradient Descent (`SGDRegressor`)** to predict final score **G3 (0–20)**.
   - Evaluated using **MAE, MSE, RMSE, and R² Score**.
2. **Academic Risk Classification**:
   - Classifies students into **At Risk (`G3 < 10`)** vs. **Not At Risk (`G3 ≥ 10`)**.
   - Implements **Support Vector Machine (SVM)**, **Decision Tree**, **K-Nearest Neighbors (KNN)**, and **Gaussian Naive Bayes**.
   - Evaluated using **Accuracy, Precision, Recall, F1-Score, and Confusion Matrix**.
3. **Ensemble Learning Techniques**:
   - **Soft Voting Classifier**, **AdaBoost Classifier**, and **Stacking Classifier** (with a Logistic Regression meta-estimator).
   - Demonstrates empirical performance superiority over individual classifiers (Stacking achieves **92.41% accuracy** and **0.889 F1-score**).
4. **Unsupervised Behavioral Clustering**:
   - **K-Means Clustering** applied to core behavioral, attendance, and academic indicators.
   - Evaluated across $K \in [2, 6]$ using **Silhouette Score** and **Davies-Bouldin Index**.
   - Profiles 3 interpretable student personas: *Academically At-Risk*, *Socially Distracted / High Absenteeism*, and *Consistent High Achievers*.
   - 2D dimensionality reduction via **Principal Component Analysis (PCA)**.
5. **Rule-Based Expert System**:
   - Pedagogical inference engine generating personalized study routines, attendance warnings, and resource allocation.
   - Generates downloadable, structured intervention reports for counselors, teachers, and parents.
6. **Streamlined Primary Workflow & Triple Format Report Downloads**:
   - **Primary Journey**: Home ➔ Enter Student Details ➔ Generate Student Report ➔ View Results ➔ Download Report.
   - **Working Multi-Format Export**:
     - 📄 **PDF**: Publication-quality official student assessment report generated via ReportLab.
     - 📊 **CSV**: Tabular data export containing all student attributes, predictions, and recommendations.
     - 📝 **TXT**: Human-readable text dossier.
   - **Secondary Section (Explore ML Insights)**: Model Comparison benchmarks and K-Means Student Clustering personas kept neatly organized without interfering with the primary user flow.

---

## 📊 Empirical Results & Benchmark Summary

All models were evaluated on the **unseen 20% test set (79 student records)** under `random_state=42` with stratified class balance.

### 1. Regression Models (Target: $G3 \in [0, 20]$)
| Model | MAE (Lower is better) | MSE | RMSE (Lower is better) | $R^2$ Score (Higher is better) |
| :--- | :---: | :---: | :---: | :---: |
| **Linear Regression (OLS)** | 1.486 | 4.324 | 2.080 | 0.796 |
| **Gradient Descent (SGD Regressor)** | **1.459** | **4.136** | **2.034** | **0.805** |

### 2. Classification & Ensemble Models (Target: Academic Risk)
| Model | Category | Accuracy (%) | Precision (%) | Recall (%) | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **KNN Classifier** | Individual | 82.28% | 77.27% | 65.38% | 0.708 |
| **Gaussian Naive Bayes** | Individual | 83.54% | 80.95% | 65.38% | 0.723 |
| **Decision Tree (depth=4)** | Individual | 87.34% | 78.57% | 84.62% | 0.815 |
| **SVM (RBF Kernel)** | Individual | 89.87% | **90.91%** | 76.92% | 0.833 |
| **Soft Voting Classifier** | Ensemble | 88.61% | 86.96% | 76.92% | 0.816 |
| **AdaBoost Classifier** | Ensemble | 91.14% | 88.00% | 84.62% | 0.863 |
| **Stacking Classifier** | **Ensemble** | **92.41%** | 85.71% | **92.31%** | **0.889** |

### 3. Clustering Metrics (K-Means on Academic & Lifestyle Space)
- **Optimal Clusters**: $K = 3$
- **Silhouette Score**: `0.166` (highest separation across candidate clusters)
- **Davies-Bouldin Index**: `1.869`
- **Identified Personas**:
  - **Cluster 0 (Academically At-Risk)**: Lower baseline grades ($G3 \approx 6.58$), high failures ($0.45$), low alcohol.
  - **Cluster 1 (High Absenteeism & Distracted)**: Highest absences ($8.68$ days), highest alcohol consumption ($Walc \approx 4.02$), borderline grades ($G3 \approx 9.44$).
  - **Cluster 2 (Consistent High Achievers)**: Highest grades ($G3 \approx 13.90$), near-zero failures ($0.10$), disciplined study patterns.

---

## 🛠️ Technology Stack
- **Language**: Python 3.10+
- **Data Manipulation**: Pandas, NumPy
- **Machine Learning & Pipeline**: Scikit-Learn
- **Model Serialization**: Joblib
- **Visualization**: Matplotlib, Seaborn
- **Web Interface**: Streamlit

---

## 📁 Repository Structure
```text
StudentSuccessPredictor/
├── data/
│   ├── student-mat.csv                 # Raw UCI Student Performance dataset (395 records)
│   └── student_mat_with_clusters.csv   # Dataset tagged with K-Means cluster labels & PCA
├── models/
│   ├── regression_linear.joblib        # Trained Linear Regression model
│   ├── regression_sgd.joblib           # Trained SGD Regressor model
│   ├── classifier_svm.joblib           # Trained Support Vector Classifier
│   ├── classifier_dt.joblib            # Trained Decision Tree Classifier
│   ├── classifier_knn.joblib           # Trained K-Nearest Neighbors Classifier
│   ├── classifier_nb.joblib            # Trained Gaussian Naive Bayes Classifier
│   ├── ensemble_voting.joblib          # Trained Soft Voting Ensemble Classifier
│   ├── ensemble_adaboost.joblib        # Trained AdaBoost Ensemble Classifier
│   ├── ensemble_stacking.joblib        # Trained Stacking Ensemble Classifier
│   ├── clustering_kmeans.joblib        # Trained K-Means Clusterer (K=3)
│   ├── clustering_scaler.joblib        # StandardScaler for clustering features
│   ├── clustering_pca.joblib           # 2D PCA Dimensionality Reducer
│   ├── preprocessor.joblib             # ColumnTransformer (StandardScaler + OneHotEncoder)
│   ├── feature_metadata.joblib         # Column names, feature mappings, and metadata
│   ├── evaluation_results.joblib       # Serialized test evaluation benchmark dict
│   └── metrics.json                    # Formatted JSON report of test benchmark metrics
├── .streamlit/
│   └── config.toml                     # Streamlit theme and headless configuration
├── app.py                              # Streamlit interactive multi-page web application
├── train_model.py                      # Complete training, evaluation, and serialization pipeline
├── expert_system.py                    # Deterministic rule-based expert recommendation engine
├── test_system.py                      # Automated test suite (6 comprehensive tests)
├── requirements.txt                    # Project dependency specification
├── PROJECT_REPORT.md                   # Comprehensive academic project report for college submission
└── README.md                           # Documentation and viva preparation guide
```

---

## ⚡ Installation & Execution Guide

### Step 1: Open PowerShell or Terminal
Navigate into the project directory:
```powershell
cd c:\Users\bhumi\OneDrive\Desktop\Projects\StudentSuccessPredictor
```

### Step 2: Set Up Virtual Environment (Optional but Recommended)
```powershell
python -m venv venv
.\venv\Scripts\activate
```

### Step 3: Install Required Dependencies
```powershell
pip install -r requirements.txt
```

### Step 4: Run the Training Pipeline
Trains all regression, classification, ensemble, and clustering models, computes real test metrics, and saves model artifacts to `models/`:
```powershell
python train_model.py
```

### Step 5: Run the Verification Test Suite
Executes all unit tests verifying data integrity, model loading, predictions, and metrics:
```powershell
python test_system.py
```

### Step 6: Launch the Streamlit Web Application
```powershell
streamlit run app.py
```
After running, open your web browser at: **`http://localhost:8501`**.

---

## 📜 License & Acknowledgments
- Dataset source: P. Cortez and A. Silva. *Using Data Mining to Predict Secondary School Student Performance*. A. Brito and J. Teixeira Eds., Proceedings of 5th FUture BUsiness TEChnology Conference (FUBUTEC 2008) pp. 5-12, Porto, Portugal, April 2008.
- UCI Machine Learning Repository: [https://archive.ics.uci.edu/dataset/320/student+performance](https://archive.ics.uci.edu/dataset/320/student+performance)
