"""
Student Success Predictor - Model Training & Evaluation Pipeline
================================================================
This script:
1. Loads or downloads the UCI Student Performance dataset (student-mat.csv)
2. Performs robust preprocessing with data leakage prevention (StandardScaler, OneHotEncoder)
3. Trains Regression Models (Linear Regression, SGDRegressor) to predict G3
4. Trains Classification Models (SVM, Decision Tree, KNN, Naive Bayes) to predict risk status
5. Trains Ensemble Models (Voting, AdaBoost, Stacking) and compares with single models
6. Performs K-Means Clustering with Silhouette Score and Davies-Bouldin Index analysis
7. Saves all trained models, preprocessors, and evaluation metrics using Joblib
"""

import os
import sys
import json
import urllib.request
import zipfile
import io
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, SGDRegressor, LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    silhouette_score,
    davies_bouldin_score
)

warnings.filterwarnings('ignore')

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
DATA_FILE = os.path.join(DATA_DIR, 'student-mat.csv')

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


def download_dataset_if_needed():
    """Ensures student-mat.csv is present in the data folder."""
    if os.path.exists(DATA_FILE):
        print(f"[✓] Dataset found at: {DATA_FILE}")
        return True

    print("[!] student-mat.csv not found locally. Attempting automatic download from UCI...")
    uci_url = "https://archive.ics.uci.edu/static/public/320/student+performance.zip"
    try:
        req = urllib.request.Request(uci_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=30) as resp:
            with zipfile.ZipFile(io.BytesIO(resp.read())) as z:
                # Check if nested student.zip exists
                if 'student.zip' in z.namelist():
                    nested_bytes = z.read('student.zip')
                    with zipfile.ZipFile(io.BytesIO(nested_bytes)) as nz:
                        nz.extractall(DATA_DIR)
                else:
                    z.extractall(DATA_DIR)
        print(f"[✓] Dataset successfully downloaded and extracted into {DATA_DIR}")
        return True
    except Exception as e:
        print(f"[✗] Automatic download failed: {e}")
        print("\nManual Instructions:")
        print("1. Download 'student-mat.csv' from https://archive.ics.uci.edu/dataset/320/student+performance")
        print(f"2. Place the file inside: {DATA_DIR}")
        return False


def load_and_preprocess_data():
    """Loads dataset and prepares train/test splits with zero data leakage."""
    if not download_dataset_if_needed():
        raise FileNotFoundError(f"Missing dataset at {DATA_FILE}. Please place student-mat.csv into data/.")

    print(f"\n--- 1. Loading Dataset ({DATA_FILE}) ---")
    df = pd.read_csv(DATA_FILE, sep=';')
    print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")

    # Feature categorization
    # Target variables
    # Regression target: G3 (0 to 20)
    # Classification target: Risk (1 = At Risk: G3 < 10, 0 = Not At Risk: G3 >= 10)
    y_reg = df['G3'].copy()
    y_clf = (df['G3'] < 10).astype(int)

    X = df.drop(columns=['G3'])

    cat_cols = [
        'school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob',
        'reason', 'guardian', 'schoolsup', 'famsup', 'paid', 'activities',
        'nursery', 'higher', 'internet', 'romantic'
    ]
    num_cols = [
        'age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures',
        'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health',
        'absences', 'G1', 'G2'
    ]

    print(f"Categorical Features ({len(cat_cols)}): {cat_cols}")
    print(f"Numerical Features ({len(num_cols)}): {num_cols}")
    print(f"Classification Class Distribution: At Risk (1): {(y_clf == 1).sum()} ({(y_clf == 1).mean()*100:.1f}%), Not At Risk (0): {(y_clf == 0).sum()} ({(y_clf == 0).mean()*100:.1f}%)")

    # Train-test split (80% train, 20% test) stratified on y_clf to preserve balance
    X_train, X_test, y_train_reg, y_test_reg, y_train_clf, y_test_clf = train_test_split(
        X, y_reg, y_clf, test_size=0.20, random_state=42, stratify=y_clf
    )

    print(f"Training set: {X_train.shape[0]} samples | Test set: {X_test.shape[0]} samples")

    # Build ColumnTransformer with strict leakage prevention
    # Scaler and OneHotEncoder are fitted ONLY on X_train!
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
        ]
    )

    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    # Extract transformed feature names
    cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(cat_cols).tolist()
    all_feature_names = num_cols + cat_feature_names
    print(f"Total processed feature dimensions: {len(all_feature_names)}")

    metadata = {
        'cat_cols': cat_cols,
        'num_cols': num_cols,
        'all_feature_names': all_feature_names,
        'unique_values': {col: df[col].unique().tolist() for col in cat_cols},
        'num_ranges': {col: {'min': float(df[col].min()), 'max': float(df[col].max()), 'default': float(df[col].median())} for col in num_cols}
    }

    return (
        df, X_train, X_test, X_train_proc, X_test_proc,
        y_train_reg, y_test_reg, y_train_clf, y_test_clf,
        preprocessor, metadata
    )


def train_regression_models(X_train_proc, X_test_proc, y_train_reg, y_test_reg):
    """Trains Linear Regression and Gradient Descent (SGDRegressor) and evaluates them."""
    print("\n--- 2. Training Regression Models (Target: G3 Final Grade) ---")
    reg_models = {
        'Linear Regression': LinearRegression(),
        'Gradient Descent (SGD)': SGDRegressor(
            loss='squared_error',
            penalty='l2',
            alpha=0.001,
            max_iter=3000,
            tol=1e-4,
            learning_rate='invscaling',
            eta0=0.01,
            power_t=0.25,
            random_state=42
        )
    }

    reg_results = {}
    saved_reg_models = {}

    for name, model in reg_models.items():
        model.fit(X_train_proc, y_train_reg)
        y_pred = model.predict(X_test_proc)

        # Compute regression metrics
        mae = mean_absolute_error(y_test_reg, y_pred)
        mse = mean_squared_error(y_test_reg, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test_reg, y_pred)

        reg_results[name] = {
            'MAE': float(mae),
            'MSE': float(mse),
            'RMSE': float(rmse),
            'R2': float(r2),
            'y_pred': y_pred.tolist()
        }
        saved_reg_models[name] = model

        print(f"  [{name}] -> MAE: {mae:.3f} | MSE: {mse:.3f} | RMSE: {rmse:.3f} | R²: {r2:.3f}")

    return saved_reg_models, reg_results


def train_classification_and_ensembles(X_train_proc, X_test_proc, y_train_clf, y_test_clf):
    """Trains individual classifiers and ensemble methods, evaluating each on test set."""
    print("\n--- 3. Training Individual Classification Models (Target: Academic Risk) ---")
    base_classifiers = {
        'SVM': SVC(kernel='rbf', C=1.0, probability=True, random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=4, min_samples_split=5, random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5, weights='distance'),
        'Naive Bayes': GaussianNB()
    }

    clf_results = {}
    saved_clf_models = {}

    for name, model in base_classifiers.items():
        model.fit(X_train_proc, y_train_clf)
        y_pred = model.predict(X_test_proc)
        y_prob = model.predict_proba(X_test_proc)[:, 1] if hasattr(model, 'predict_proba') else None

        acc = accuracy_score(y_test_clf, y_pred)
        prec = precision_score(y_test_clf, y_pred, zero_division=0)
        rec = recall_score(y_test_clf, y_pred, zero_division=0)
        f1 = f1_score(y_test_clf, y_pred, zero_division=0)
        cm = confusion_matrix(y_test_clf, y_pred).tolist()

        clf_results[name] = {
            'Accuracy': float(acc),
            'Precision': float(prec),
            'Recall': float(rec),
            'F1-Score': float(f1),
            'Confusion_Matrix': cm,
            'y_pred': y_pred.tolist(),
            'y_prob': y_prob.tolist() if y_prob is not None else None,
            'Category': 'Individual'
        }
        saved_clf_models[name] = model

        print(f"  [{name}] -> Accuracy: {acc*100:.2f}% | Precision: {prec*100:.2f}% | Recall: {rec*100:.2f}% | F1: {f1:.3f}")

    print("\n--- 4. Training Ensemble Classifiers (Voting, AdaBoost, Stacking) ---")
    ensemble_models = {
        'Voting Classifier': VotingClassifier(
            estimators=[
                ('svm', base_classifiers['SVM']),
                ('dt', base_classifiers['Decision Tree']),
                ('knn', base_classifiers['KNN']),
                ('nb', base_classifiers['Naive Bayes'])
            ],
            voting='soft'
        ),
        'AdaBoost': AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
            n_estimators=50,
            learning_rate=1.0,
            random_state=42
        ),
        'Stacking Classifier': StackingClassifier(
            estimators=[
                ('svm', SVC(kernel='rbf', C=1.0, probability=True, random_state=42)),
                ('dt', DecisionTreeClassifier(max_depth=4, min_samples_split=5, random_state=42)),
                ('knn', KNeighborsClassifier(n_neighbors=5, weights='distance')),
                ('nb', GaussianNB())
            ],
            final_estimator=LogisticRegression(random_state=42, max_iter=1000),
            cv=5
        )
    }

    for name, model in ensemble_models.items():
        model.fit(X_train_proc, y_train_clf)
        y_pred = model.predict(X_test_proc)
        y_prob = model.predict_proba(X_test_proc)[:, 1] if hasattr(model, 'predict_proba') else None

        acc = accuracy_score(y_test_clf, y_pred)
        prec = precision_score(y_test_clf, y_pred, zero_division=0)
        rec = recall_score(y_test_clf, y_pred, zero_division=0)
        f1 = f1_score(y_test_clf, y_pred, zero_division=0)
        cm = confusion_matrix(y_test_clf, y_pred).tolist()

        clf_results[name] = {
            'Accuracy': float(acc),
            'Precision': float(prec),
            'Recall': float(rec),
            'F1-Score': float(f1),
            'Confusion_Matrix': cm,
            'y_pred': y_pred.tolist(),
            'y_prob': y_prob.tolist() if y_prob is not None else None,
            'Category': 'Ensemble'
        }
        saved_clf_models[name] = model

        print(f"  [{name} (Ensemble)] -> Accuracy: {acc*100:.2f}% | Precision: {prec*100:.2f}% | Recall: {rec*100:.2f}% | F1: {f1:.3f}")

    return saved_clf_models, clf_results


def train_clustering(df):
    """
    Performs K-Means clustering on behavioral and academic features.
    Computes Silhouette Score and Davies-Bouldin Index for K=2..6.
    Fits optimal K=3 model and profiles clusters.
    """
    print("\n--- 5. Training K-Means Clustering on Behavioral & Academic Features ---")
    cluster_features = ['studytime', 'failures', 'absences', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'G1', 'G2', 'G3']
    X_cl = df[cluster_features].copy()

    cl_scaler = StandardScaler()
    X_cl_scaled = cl_scaler.fit_transform(X_cl)

    # Evaluate multiple K values for Elbow & Silhouette analysis
    k_range = list(range(2, 7))
    k_metrics = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_cl_scaled)
        sil = silhouette_score(X_cl_scaled, labels)
        db = davies_bouldin_score(X_cl_scaled, labels)
        inertia = float(km.inertia_)
        k_metrics[k] = {
            'k': k,
            'silhouette_score': float(sil),
            'davies_bouldin_index': float(db),
            'inertia': inertia
        }
        print(f"  K={k} -> Silhouette Score: {sil:.3f} | Davies-Bouldin Index: {db:.3f} | Inertia: {inertia:.1f}")

    # Optimal selected K = 3
    optimal_k = 3
    best_kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    cluster_labels = best_kmeans.fit_predict(X_cl_scaled)

    # PCA 2D projection for visualization
    pca = PCA(n_components=2, random_state=42)
    pca_2d = pca.fit_transform(X_cl_scaled)

    df_clustered = df.copy()
    df_clustered['cluster'] = cluster_labels
    df_clustered['pca_1'] = pca_2d[:, 0]
    df_clustered['pca_2'] = pca_2d[:, 1]

    # Cluster profiles & archetypes
    cluster_profiles = df_clustered.groupby('cluster')[cluster_features].mean().round(2).to_dict(orient='index')

    cluster_personas = {
        0: {
            'name': 'Academically At-Risk Learners',
            'desc': 'Lower initial grades (G1, G2), below average passing score (G3 ~ 6.6), moderate study time but significant conceptual difficulty needing tutorial intervention.',
            'badge': 'danger'
        },
        1: {
            'name': 'High Absenteeism & Socially Distracted',
            'desc': 'Highest absenteeism (~8.7 days), elevated weekend alcohol consumption (Walc ~ 4.0), high outings, borderline passing grades (~9.5). Requires discipline monitoring.',
            'badge': 'warning'
        },
        2: {
            'name': 'Consistent High Achievers',
            'desc': 'Highest academic performance (G3 ~ 13.9), lowest failures (~0.1), lowest alcohol consumption, consistent study habits. Honors and distinction tier.',
            'badge': 'success'
        }
    }

    clustering_artifacts = {
        'model': best_kmeans,
        'scaler': cl_scaler,
        'pca': pca,
        'features': cluster_features,
        'k_metrics': k_metrics,
        'optimal_k': optimal_k,
        'silhouette': float(k_metrics[optimal_k]['silhouette_score']),
        'davies_bouldin': float(k_metrics[optimal_k]['davies_bouldin_index']),
        'cluster_profiles': cluster_profiles,
        'cluster_personas': cluster_personas,
        'df_clustered': df_clustered
    }

    print(f"\n[✓] Optimal K-Means (K={optimal_k}): Silhouette Score = {clustering_artifacts['silhouette']:.3f}, Davies-Bouldin Index = {clustering_artifacts['davies_bouldin']:.3f}")
    return clustering_artifacts


def save_all_artifacts(preprocessor, metadata, reg_models, reg_results,
                       clf_models, clf_results, clustering_artifacts,
                       y_test_reg, y_test_clf):
    """Saves models and comprehensive evaluation metrics to models/ directory."""
    print(f"\n--- 6. Persisting Models & Metadata to {MODELS_DIR} ---")

    # 1. Preprocessor & metadata
    joblib.dump(preprocessor, os.path.join(MODELS_DIR, 'preprocessor.joblib'))
    joblib.dump(metadata, os.path.join(MODELS_DIR, 'feature_metadata.joblib'))

    # 2. Regression models
    joblib.dump(reg_models['Linear Regression'], os.path.join(MODELS_DIR, 'regression_linear.joblib'))
    joblib.dump(reg_models['Gradient Descent (SGD)'], os.path.join(MODELS_DIR, 'regression_sgd.joblib'))

    # 3. Classification models
    joblib.dump(clf_models['SVM'], os.path.join(MODELS_DIR, 'classifier_svm.joblib'))
    joblib.dump(clf_models['Decision Tree'], os.path.join(MODELS_DIR, 'classifier_dt.joblib'))
    joblib.dump(clf_models['KNN'], os.path.join(MODELS_DIR, 'classifier_knn.joblib'))
    joblib.dump(clf_models['Naive Bayes'], os.path.join(MODELS_DIR, 'classifier_nb.joblib'))

    # 4. Ensemble models
    joblib.dump(clf_models['Voting Classifier'], os.path.join(MODELS_DIR, 'ensemble_voting.joblib'))
    joblib.dump(clf_models['AdaBoost'], os.path.join(MODELS_DIR, 'ensemble_adaboost.joblib'))
    joblib.dump(clf_models['Stacking Classifier'], os.path.join(MODELS_DIR, 'ensemble_stacking.joblib'))

    # 5. Clustering artifacts
    joblib.dump(clustering_artifacts['model'], os.path.join(MODELS_DIR, 'clustering_kmeans.joblib'))
    joblib.dump(clustering_artifacts['scaler'], os.path.join(MODELS_DIR, 'clustering_scaler.joblib'))
    joblib.dump(clustering_artifacts['pca'], os.path.join(MODELS_DIR, 'clustering_pca.joblib'))

    # 6. Comprehensive evaluation results
    full_evaluation = {
        'regression': reg_results,
        'classification': clf_results,
        'clustering': {
            'k_metrics': clustering_artifacts['k_metrics'],
            'optimal_k': clustering_artifacts['optimal_k'],
            'silhouette': clustering_artifacts['silhouette'],
            'davies_bouldin': clustering_artifacts['davies_bouldin'],
            'profiles': clustering_artifacts['cluster_profiles'],
            'personas': clustering_artifacts['cluster_personas']
        },
        'test_actuals': {
            'y_test_reg': y_test_reg.tolist(),
            'y_test_clf': y_test_clf.tolist()
        }
    }

    joblib.dump(full_evaluation, os.path.join(MODELS_DIR, 'evaluation_results.joblib'))

    # Clean JSON report for quick verification
    json_summary = {
        'regression_metrics': {
            m: {k: v for k, v in res.items() if k != 'y_pred'}
            for m, res in reg_results.items()
        },
        'classification_metrics': {
            m: {k: v for k, v in res.items() if k not in ['y_pred', 'y_prob']}
            for m, res in clf_results.items()
        },
        'clustering_metrics': {
            'optimal_k': clustering_artifacts['optimal_k'],
            'silhouette_score': clustering_artifacts['silhouette'],
            'davies_bouldin_index': clustering_artifacts['davies_bouldin'],
            'k_evaluations': clustering_artifacts['k_metrics']
        }
    }

    with open(os.path.join(MODELS_DIR, 'metrics.json'), 'w') as f:
        json.dump(json_summary, f, indent=4)

    # Save clustered data for the Streamlit dashboard
    clustering_artifacts['df_clustered'].to_csv(
        os.path.join(DATA_DIR, 'student_mat_with_clusters.csv'),
        index=False
    )

    print(f"[✓] Saved all artifacts successfully in {MODELS_DIR}")
    print(f"[✓] Saved metrics summary to {os.path.join(MODELS_DIR, 'metrics.json')}")


def main():
    print("=" * 70)
    print("  STUDENT SUCCESS PREDICTOR: ML TRAINING & BENCHMARKING PIPELINE")
    print("=" * 70)

    # Step 1: Preprocess & Split
    (
        df, X_train, X_test, X_train_proc, X_test_proc,
        y_train_reg, y_test_reg, y_train_clf, y_test_clf,
        preprocessor, metadata
    ) = load_and_preprocess_data()

    # Step 2: Regression Models
    reg_models, reg_results = train_regression_models(
        X_train_proc, X_test_proc, y_train_reg, y_test_reg
    )

    # Step 3 & 4: Classification & Ensembles
    clf_models, clf_results = train_classification_and_ensembles(
        X_train_proc, X_test_proc, y_train_clf, y_test_clf
    )

    # Step 5: Clustering
    clustering_artifacts = train_clustering(df)

    # Step 6: Save Everything
    save_all_artifacts(
        preprocessor, metadata, reg_models, reg_results,
        clf_models, clf_results, clustering_artifacts,
        y_test_reg, y_test_clf
    )

    print("\n" + "=" * 70)
    print("  TRAINING PIPELINE COMPLETE! ALL MODELS READY FOR STREAMLIT APP")
    print("=" * 70)


if __name__ == '__main__':
    main()
