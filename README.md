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
6. **Interactive SaaS-Style Streamlit Web Dashboard**:
   - Modern 6-stage guided journey designed for seamless first-time user experience:
     1. **Overview & Welcome**: System introduction, "How It Works" 4-step workflow, and platform KPI benchmarks.
     2. **Enter Details & Predict Grade**: 3-tab categorized input form with one-click student presets and regression predictions.
     3. **Academic Failure Risk**: Multi-model consensus risk evaluation across 7 distinct algorithms with failure probability gauges.
     4. **AI Model Benchmarks**: Transparent empirical evaluation results with a plain-English metric glossary and confusion matrix explorer.
     5. **Student Groups (Clustering)**: Unsupervised K-Means persona discovery with 2D PCA cluster visualization.
     6. **Action Plan & Study Advice**: Rule-based expert pedagogical recommendations, weekly timetables, and downloadable reports.

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

## 🎓 Viva & Practical Examination Q&A (Cheat Sheet)

### Q1: Why did you choose the UCI Student Performance dataset?
> **Answer**: The UCI Student Performance dataset (Cortez and Silva, 2008) is a benchmark educational data mining dataset. It contains 395 secondary school students evaluated across 33 attributes spanning demographic backgrounds, parental education, study time, absences, social activities, and historical grades (G1, G2, G3). It offers an ideal balance of numeric and categorical features for both regression and classification.

### Q2: How did you prevent Data Leakage?
> **Answer**: Data leakage occurs when information from the test dataset influences training. We strictly prevented this by:
> 1. Performing train-test splitting (80/20 stratified) **before** applying any transformers.
> 2. Fitting `StandardScaler` and `OneHotEncoder` **only** on the training partition (`X_train`).
> 3. Transforming the test partition (`X_test`) using the fitted preprocessor without calling `.fit()`.

### Q3: What is the difference between Linear Regression and SGDRegressor?
> **Answer**:
> - **Linear Regression (OLS)** computes closed-form normal equations ($\theta = (X^T X)^{-1} X^T y$). It finds exact analytical minima but scales with $\mathcal{O}(d^3)$ computational complexity.
> - **SGDRegressor** iteratively updates weights via Stochastic Gradient Descent using mini-batches/samples:
>   $$\theta_{t+1} = \theta_t - \eta \nabla J(\theta_t)$$
>   It requires scaled features and is highly efficient for large datasets or streaming data. In our experiments, SGD achieved an $R^2$ of **0.805** and RMSE of **2.034**.

### Q4: Why does Stacking outperform individual classifiers?
> **Answer**: Stacking (Stacked Generalization) trains diverse base learners (SVM, Decision Tree, KNN, Naive Bayes) with distinct inductive biases. Instead of taking a simple average, Stacking trains a meta-learner (Logistic Regression) on out-of-fold predictions. The meta-learner learns which classifier is reliable in specific regions of the feature space, yielding our top test accuracy of **92.41%** and F1-score of **0.889**.

### Q5: How was the optimal K chosen in K-Means clustering?
> **Answer**: We evaluated $K \in [2, 6]$ using two unsupervised metrics:
> 1. **Silhouette Score**: Measures intra-cluster cohesion vs. inter-cluster separation (higher is better).
> 2. **Davies-Bouldin Index**: Measures the similarity between each cluster and its most similar counterpart (lower is better).
> $K=3$ was selected as the optimal trade-off providing the highest pedagogical interpretability into three distinct personas.

### Q6: How does the Expert System function?
> **Answer**: The Expert System is a deterministic inference engine containing categorized pedagogical production rules (`IF condition THEN recommendation`). It evaluates student attributes alongside predicted grade and risk status to prescribe actionable guidance across Academic Remediation, Attendance Monitoring, Lifestyle Optimization, and Timetable Allocation.

---

## 📜 License & Acknowledgments
- Dataset source: P. Cortez and A. Silva. *Using Data Mining to Predict Secondary School Student Performance*. A. Brito and J. Teixeira Eds., Proceedings of 5th FUture BUsiness TEChnology Conference (FUBUTEC 2008) pp. 5-12, Porto, Portugal, April 2008.
- UCI Machine Learning Repository: [https://archive.ics.uci.edu/dataset/320/student+performance](https://archive.ics.uci.edu/dataset/320/student+performance)
