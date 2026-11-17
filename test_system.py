"""
Comprehensive Test Suite for Student Success Predictor
======================================================
Tests:
1. Dataset integrity (student-mat.csv)
2. Preprocessor & feature metadata loading
3. Regression models (Linear Regression, SGDRegressor)
4. Classification models (SVM, Decision Tree, KNN, Naive Bayes)
5. Ensemble models (Voting, AdaBoost, Stacking)
6. Clustering pipeline (K-Means, PCA, Silhouette, Davies-Bouldin)
7. Expert system rule-based recommendation engine
8. Metrics consistency and file verification
"""

import os
import json
import numpy as np
import pandas as pd
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR = os.path.join(BASE_DIR, 'data')
DATA_FILE = os.path.join(DATA_DIR, 'student-mat.csv')


def test_dataset_exists():
    """Verify raw dataset is present, valid, and non-empty."""
    assert os.path.exists(DATA_FILE), f"Dataset file missing at {DATA_FILE}"
    df = pd.read_csv(DATA_FILE, sep=';')
    assert len(df) == 395, f"Expected 395 rows, got {len(df)}"
    assert 'G3' in df.columns, "G3 column missing in dataset"
    assert 'G1' in df.columns and 'G2' in df.columns, "G1 or G2 missing"
    assert df.isnull().sum().sum() == 0, "Dataset contains unexpected NaN values"
    print("  [PASS] test_dataset_exists")


def test_models_exist():
    """Verify all 16 required model artifacts and metrics files exist."""
    expected_files = [
        'preprocessor.joblib',
        'feature_metadata.joblib',
        'regression_linear.joblib',
        'regression_sgd.joblib',
        'classifier_svm.joblib',
        'classifier_dt.joblib',
        'classifier_knn.joblib',
        'classifier_nb.joblib',
        'ensemble_voting.joblib',
        'ensemble_adaboost.joblib',
        'ensemble_stacking.joblib',
        'clustering_kmeans.joblib',
        'clustering_scaler.joblib',
        'clustering_pca.joblib',
        'evaluation_results.joblib',
        'metrics.json'
    ]
    for filename in expected_files:
        filepath = os.path.join(MODELS_DIR, filename)
        assert os.path.exists(filepath), f"Missing artifact: {filepath}"
    print("  [PASS] test_models_exist")


def test_preprocessing_and_regression():
    """Verify preprocessor transforms sample data and regression models predict valid scores."""
    preprocessor = joblib.load(os.path.join(MODELS_DIR, 'preprocessor.joblib'))
    lr = joblib.load(os.path.join(MODELS_DIR, 'regression_linear.joblib'))
    sgd = joblib.load(os.path.join(MODELS_DIR, 'regression_sgd.joblib'))

    sample_student = pd.DataFrame([{
        'school': 'GP', 'sex': 'F', 'age': 17, 'address': 'U', 'famsize': 'GT3',
        'Pstatus': 'T', 'Medu': 4, 'Fedu': 3, 'Mjob': 'health', 'Fjob': 'services',
        'reason': 'course', 'guardian': 'mother', 'traveltime': 1, 'studytime': 3,
        'failures': 0, 'schoolsup': 'no', 'famsup': 'yes', 'paid': 'yes',
        'activities': 'yes', 'nursery': 'yes', 'higher': 'yes', 'internet': 'yes',
        'romantic': 'no', 'famrel': 4, 'freetime': 3, 'goout': 2, 'Dalc': 1,
        'Walc': 1, 'health': 5, 'absences': 2, 'G1': 15, 'G2': 16
    }])

    X_proc = preprocessor.transform(sample_student)
    assert X_proc.shape[0] == 1, "Expected single sample transformed row"
    assert X_proc.shape[1] == 58, f"Expected 58 features, got {X_proc.shape[1]}"

    pred_lr = lr.predict(X_proc)[0]
    pred_sgd = sgd.predict(X_proc)[0]

    assert 0 <= pred_lr <= 25, f"Linear Regression prediction out of realistic bounds: {pred_lr}"
    assert 0 <= pred_sgd <= 25, f"SGD prediction out of realistic bounds: {pred_sgd}"
    print(f"  [PASS] test_preprocessing_and_regression: LR={pred_lr:.2f}, SGD={pred_sgd:.2f}")


def test_classifiers_and_ensembles():
    """Verify all 4 individual classifiers and 3 ensembles produce valid predictions."""
    preprocessor = joblib.load(os.path.join(MODELS_DIR, 'preprocessor.joblib'))
    
    models = {
        'SVM': joblib.load(os.path.join(MODELS_DIR, 'classifier_svm.joblib')),
        'Decision Tree': joblib.load(os.path.join(MODELS_DIR, 'classifier_dt.joblib')),
        'KNN': joblib.load(os.path.join(MODELS_DIR, 'classifier_knn.joblib')),
        'Naive Bayes': joblib.load(os.path.join(MODELS_DIR, 'classifier_nb.joblib')),
        'Voting': joblib.load(os.path.join(MODELS_DIR, 'ensemble_voting.joblib')),
        'AdaBoost': joblib.load(os.path.join(MODELS_DIR, 'ensemble_adaboost.joblib')),
        'Stacking': joblib.load(os.path.join(MODELS_DIR, 'ensemble_stacking.joblib'))
    }

    sample_at_risk = pd.DataFrame([{
        'school': 'MS', 'sex': 'M', 'age': 18, 'address': 'R', 'famsize': 'GT3',
        'Pstatus': 'A', 'Medu': 1, 'Fedu': 1, 'Mjob': 'other', 'Fjob': 'other',
        'reason': 'course', 'guardian': 'other', 'traveltime': 3, 'studytime': 1,
        'failures': 2, 'schoolsup': 'no', 'famsup': 'no', 'paid': 'no',
        'activities': 'no', 'nursery': 'no', 'higher': 'no', 'internet': 'no',
        'romantic': 'yes', 'famrel': 2, 'freetime': 4, 'goout': 5, 'Dalc': 3,
        'Walc': 5, 'health': 2, 'absences': 20, 'G1': 5, 'G2': 6
    }])

    X_proc = preprocessor.transform(sample_at_risk)

    for name, model in models.items():
        pred = model.predict(X_proc)[0]
        assert pred in [0, 1], f"Model {name} predicted non-binary label: {pred}"
        if hasattr(model, 'predict_proba'):
            prob = model.predict_proba(X_proc)[0]
            assert len(prob) == 2, f"Model {name} probability vector must have length 2"
            assert np.isclose(sum(prob), 1.0, atol=1e-3), f"Probabilities do not sum to 1 in {name}"

    print("  [PASS] test_classifiers_and_ensembles")


def test_clustering_pipeline():
    """Verify K-Means clustering, PCA transformer, and metrics."""
    kmeans = joblib.load(os.path.join(MODELS_DIR, 'clustering_kmeans.joblib'))
    scaler = joblib.load(os.path.join(MODELS_DIR, 'clustering_scaler.joblib'))
    pca = joblib.load(os.path.join(MODELS_DIR, 'clustering_pca.joblib'))

    assert kmeans.n_clusters == 3, f"Expected 3 clusters, got {kmeans.n_clusters}"

    cluster_features = ['studytime', 'failures', 'absences', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'G1', 'G2', 'G3']
    sample_df = pd.DataFrame([[2, 0, 4, 3, 3, 1, 1, 4, 14, 14, 15]], columns=cluster_features)
    scaled = scaler.transform(sample_df)
    cluster = kmeans.predict(scaled)[0]
    assert cluster in [0, 1, 2], f"Predicted cluster out of range: {cluster}"

    pca_proj = pca.transform(scaled)
    assert pca_proj.shape == (1, 2), f"Expected PCA 2D coordinates, got {pca_proj.shape}"

    print(f"  [PASS] test_clustering_pipeline: Assigned Cluster {cluster}")


def test_expert_system():
    """Verify rule-based expert recommendation engine generates complete recommendations."""
    from expert_system import generate_recommendations

    sample_dict = {
        'studytime': 1, 'failures': 2, 'absences': 18,
        'G1': 6, 'G2': 7, 'Dalc': 2, 'Walc': 4, 'goout': 4,
        'schoolsup': 'no', 'famsup': 'no', 'paid': 'no',
        'higher': 'yes', 'internet': 'yes'
    }
    rec = generate_recommendations(sample_dict, predicted_g3=6.8, is_at_risk=True)

    assert rec['is_at_risk'] is True
    assert "HIGH RISK" in rec['urgency']
    assert len(rec['academic_actions']) > 0
    assert len(rec['attendance_actions']) > 0
    assert len(rec['weekly_schedule']) == 8
    print("  [PASS] test_expert_system")


def test_metrics_json():
    """Verify metrics.json contains genuine, non-trivial performance numbers."""
    with open(os.path.join(MODELS_DIR, 'metrics.json'), 'r') as f:
        metrics = json.load(f)

    reg = metrics['regression_metrics']
    assert 'Linear Regression' in reg and 'Gradient Descent (SGD)' in reg
    assert reg['Linear Regression']['R2'] > 0.70
    assert reg['Gradient Descent (SGD)']['R2'] > 0.70

    clf = metrics['classification_metrics']
    assert 'Stacking Classifier' in clf
    assert clf['Stacking Classifier']['Accuracy'] > 0.85
    assert clf['Stacking Classifier']['F1-Score'] > 0.80

    cl = metrics['clustering_metrics']
    assert cl['optimal_k'] == 3
    assert cl['silhouette_score'] > 0.10
    print("  [PASS] test_metrics_json")


if __name__ == '__main__':
    print("=" * 65)
    print("  RUNNING SYSTEM VERIFICATION TEST SUITE")
    print("=" * 65)
    test_dataset_exists()
    test_models_exist()
    test_preprocessing_and_regression()
    test_classifiers_and_ensembles()
    test_clustering_pipeline()
    test_expert_system()
    test_metrics_json()
    print("=" * 65)
    print("  ALL 6 TEST SUITES PASSED SUCCESSFULLY (100% GREEN)")
    print("=" * 65)
