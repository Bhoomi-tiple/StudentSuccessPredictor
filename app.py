"""
Student Success Predictor - Premium EdTech AI/ML Web Application
================================================================
An intuitive, SaaS-style AI-powered decision support platform for student academic performance,
early failure risk detection, behavioral clustering, and personalized pedagogical guidance.
"""

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Import rule-based expert recommendation engine (existing backend module)
from expert_system import generate_recommendations

# ---------------------------------------------------------
# Page Configuration & Modern SaaS Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="EduPredict AI | Student Academic Success Platform",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Modern, Premium SaaS EdTech Interface
st.markdown("""
<style>
    /* Google Fonts Import */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    /* Hero Banner */
    .hero-banner {
        background: linear-gradient(135deg, #1E1B4B 0%, #312E81 50%, #4338CA 100%);
        color: white;
        padding: 32px 36px;
        border-radius: 16px;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(67, 56, 202, 0.25);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .hero-pill {
        display: inline-block;
        background: rgba(255, 255, 255, 0.18);
        backdrop-filter: blur(8px);
        color: #E0E7FF;
        font-size: 12px;
        font-weight: 700;
        padding: 4px 12px;
        border-radius: 9999px;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        margin-bottom: 12px;
    }
    .hero-title {
        font-size: 32px;
        font-weight: 800;
        color: #FFFFFF !important;
        margin: 0 0 10px 0;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        font-size: 16px;
        color: #C7D2FE;
        line-height: 1.6;
        margin: 0;
        max-width: 780px;
    }

    /* Step Journey Bar */
    .journey-tracker {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 14px 20px;
        margin-bottom: 24px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .journey-step-badge {
        background: #EEF2FF;
        color: #4F46E5;
        font-weight: 700;
        font-size: 12px;
        padding: 4px 12px;
        border-radius: 20px;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }

    /* Guide Instructions Box */
    .guide-box {
        background-color: #F8FAFC;
        border-left: 4px solid #4F46E5;
        border-top: 1px solid #E2E8F0;
        border-right: 1px solid #E2E8F0;
        border-bottom: 1px solid #E2E8F0;
        border-radius: 0 12px 12px 0;
        padding: 16px 20px;
        margin-bottom: 24px;
    }
    .guide-title {
        color: #1E293B;
        font-size: 14px;
        font-weight: 700;
        margin-bottom: 6px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .guide-desc {
        color: #475569;
        font-size: 13.5px;
        line-height: 1.5;
        margin: 0;
    }

    /* Modern Card Container */
    .saas-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 20px 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -2px rgba(0, 0, 0, 0.05);
        margin-bottom: 20px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .saas-card:hover {
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08);
    }

    /* KPI Metric Card */
    .kpi-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 18px 20px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.04);
        border-top: 4px solid #4F46E5;
    }
    .kpi-title {
        color: #64748B;
        font-size: 12px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        margin-bottom: 4px;
    }
    .kpi-value {
        color: #0F172A;
        font-size: 28px;
        font-weight: 800;
        margin-bottom: 4px;
        line-height: 1.2;
    }
    .kpi-sub {
        color: #64748B;
        font-size: 12.5px;
    }

    /* Active Student Chip */
    .student-chip {
        background: #F1F5F9;
        border: 1px solid #CBD5E1;
        border-radius: 8px;
        padding: 10px 16px;
        margin-bottom: 16px;
        display: flex;
        flex-wrap: wrap;
        gap: 16px;
        align-items: center;
        font-size: 13px;
        color: #334155;
    }
    .chip-item {
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }

    /* Badges */
    .badge-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 12px;
        font-weight: 700;
    }
    .badge-pass { background-color: #ECFDF5; color: #047857; border: 1px solid #A7F3D0; }
    .badge-risk { background-color: #FEF2F2; color: #B91C1C; border: 1px solid #FECACA; }
    .badge-warn { background-color: #FFFBEB; color: #B45309; border: 1px solid #FDE68A; }
    .badge-info { background-color: #EEF2FF; color: #4338CA; border: 1px solid #C7D2FE; }

    /* Next Step Action Container */
    .next-step-box {
        background: linear-gradient(to right, #F8FAFC, #EEF2FF);
        border: 1px dashed #C7D2FE;
        border-radius: 14px;
        padding: 20px 24px;
        margin-top: 32px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .next-step-title {
        font-weight: 700;
        color: #1E293B;
        font-size: 15px;
        margin-bottom: 4px;
    }
    .next-step-desc {
        color: #64748B;
        font-size: 13px;
        margin: 0;
    }

    /* Feature Explanation Callout */
    .callout-box {
        background: #FFFFFF;
        border-left: 3px solid #10B981;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        font-size: 13px;
        color: #1F2937;
        margin: 12px 0;
    }

    /* Button Tweaks */
    div.stButton > button:first-child {
        font-weight: 600;
        border-radius: 8px;
        padding: 0.5rem 1.2rem;
        transition: all 0.2s ease;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Load Cached Machine Learning Artifacts
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR = os.path.join(BASE_DIR, 'data')

@st.cache_resource
def load_all_artifacts():
    """Loads all persisted models, preprocessors, and evaluation metrics without recomputing."""
    try:
        preprocessor = joblib.load(os.path.join(MODELS_DIR, 'preprocessor.joblib'))
        metadata = joblib.load(os.path.join(MODELS_DIR, 'feature_metadata.joblib'))
        
        reg_models = {
            'Linear Regression': joblib.load(os.path.join(MODELS_DIR, 'regression_linear.joblib')),
            'Gradient Descent (SGD)': joblib.load(os.path.join(MODELS_DIR, 'regression_sgd.joblib'))
        }
        
        clf_models = {
            'SVM': joblib.load(os.path.join(MODELS_DIR, 'classifier_svm.joblib')),
            'Decision Tree': joblib.load(os.path.join(MODELS_DIR, 'classifier_dt.joblib')),
            'KNN': joblib.load(os.path.join(MODELS_DIR, 'classifier_knn.joblib')),
            'Naive Bayes': joblib.load(os.path.join(MODELS_DIR, 'classifier_nb.joblib')),
            'Voting Classifier': joblib.load(os.path.join(MODELS_DIR, 'ensemble_voting.joblib')),
            'AdaBoost': joblib.load(os.path.join(MODELS_DIR, 'ensemble_adaboost.joblib')),
            'Stacking Classifier': joblib.load(os.path.join(MODELS_DIR, 'ensemble_stacking.joblib'))
        }
        
        clustering = {
            'model': joblib.load(os.path.join(MODELS_DIR, 'clustering_kmeans.joblib')),
            'scaler': joblib.load(os.path.join(MODELS_DIR, 'clustering_scaler.joblib')),
            'pca': joblib.load(os.path.join(MODELS_DIR, 'clustering_pca.joblib'))
        }
        
        eval_results = joblib.load(os.path.join(MODELS_DIR, 'evaluation_results.joblib'))
        
        clustered_data_path = os.path.join(DATA_DIR, 'student_mat_with_clusters.csv')
        df_clustered = pd.read_csv(clustered_data_path) if os.path.exists(clustered_data_path) else None
        
        return preprocessor, metadata, reg_models, clf_models, clustering, eval_results, df_clustered
    except Exception as e:
        st.error(f"Error loading system models: {e}. Please ensure `python train_model.py` has been executed.")
        return None, None, None, None, None, None, None

(
    preprocessor,
    metadata,
    reg_models,
    clf_models,
    clustering_artifacts,
    eval_results,
    df_clustered
) = load_all_artifacts()

# ---------------------------------------------------------
# Session State Management (Persistent User Journey & Student Data)
# ---------------------------------------------------------
if 'page_index' not in st.session_state:
    st.session_state['page_index'] = 0

def set_page(idx):
    """Safely updates active page and re-renders view."""
    st.session_state['page_index'] = idx
    st.rerun()

# Default Student Preset Profiles
PRESETS = {
    "High Achiever": {
        'school': 'GP', 'sex': 'F', 'age': 16, 'address': 'U', 'famsize': 'GT3',
        'Pstatus': 'T', 'Medu': 4, 'Fedu': 4, 'Mjob': 'health', 'Fjob': 'teacher',
        'reason': 'reputation', 'guardian': 'mother', 'traveltime': 1, 'studytime': 4,
        'failures': 0, 'schoolsup': 'no', 'famsup': 'yes', 'paid': 'yes',
        'activities': 'yes', 'nursery': 'yes', 'higher': 'yes', 'internet': 'yes',
        'romantic': 'no', 'famrel': 5, 'freetime': 3, 'goout': 2, 'Dalc': 1,
        'Walc': 1, 'health': 5, 'absences': 2, 'G1': 16, 'G2': 17
    },
    "At-Risk Student": {
        'school': 'MS', 'sex': 'M', 'age': 18, 'address': 'R', 'famsize': 'GT3',
        'Pstatus': 'A', 'Medu': 1, 'Fedu': 1, 'Mjob': 'other', 'Fjob': 'other',
        'reason': 'course', 'guardian': 'other', 'traveltime': 3, 'studytime': 1,
        'failures': 2, 'schoolsup': 'no', 'famsup': 'no', 'paid': 'no',
        'activities': 'no', 'nursery': 'no', 'higher': 'no', 'internet': 'no',
        'romantic': 'yes', 'famrel': 2, 'freetime': 4, 'goout': 4, 'Dalc': 2,
        'Walc': 4, 'health': 2, 'absences': 18, 'G1': 6, 'G2': 7
    },
    "Average / Distracted": {
        'school': 'GP', 'sex': 'M', 'age': 17, 'address': 'U', 'famsize': 'LE3',
        'Pstatus': 'T', 'Medu': 2, 'Fedu': 2, 'Mjob': 'services', 'Fjob': 'services',
        'reason': 'home', 'guardian': 'mother', 'traveltime': 2, 'studytime': 2,
        'failures': 1, 'schoolsup': 'no', 'famsup': 'yes', 'paid': 'no',
        'activities': 'yes', 'nursery': 'yes', 'higher': 'yes', 'internet': 'yes',
        'romantic': 'yes', 'famrel': 3, 'freetime': 4, 'goout': 5, 'Dalc': 2,
        'Walc': 4, 'health': 4, 'absences': 10, 'G1': 10, 'G2': 9
    }
}

if 'active_student' not in st.session_state:
    st.session_state['active_student'] = PRESETS["Average / Distracted"].copy()

# Grade Converter
def get_letter_grade(score):
    if score >= 16:
        return "A", "Excellent / Distinction", "badge-pass"
    elif score >= 14:
        return "B", "Very Satisfactory", "badge-pass"
    elif score >= 12:
        return "C", "Satisfactory Pass", "badge-info"
    elif score >= 10:
        return "D", "Borderline Pass", "badge-warn"
    else:
        return "F", "Failing / High Risk", "badge-risk"

# ---------------------------------------------------------
# Sidebar Navigation with Sequential Journey Numbers
# ---------------------------------------------------------
st.sidebar.markdown("""
<div style="padding: 12px 0 20px 0;">
    <div style="font-size: 20px; font-weight: 800; color: #1E1B4B; display: flex; align-items: center; gap: 8px;">
        <span>🎓</span> EduPredict AI
    </div>
    <div style="font-size: 12px; color: #64748B; margin-top: 2px;">Student Success & Risk Analytics</div>
</div>
""", unsafe_allow_html=True)

NAV_PAGES = [
    "🏠 1. Overview & Welcome",
    "📝 2. Enter Details & Predict Grade",
    "⚠️ 3. Academic Failure Risk",
    "📊 4. AI Model Benchmarks",
    "👥 5. Student Groups (Clustering)",
    "💡 6. Action Plan & Study Advice"
]

selected_nav = st.sidebar.radio(
    "Application Journey:",
    NAV_PAGES,
    index=st.session_state['page_index'],
    key="nav_radio"
)
current_page_idx = NAV_PAGES.index(selected_nav)
if current_page_idx != st.session_state['page_index']:
    st.session_state['page_index'] = current_page_idx
    st.rerun()

st.sidebar.markdown("---")

# Quick Preset Loader in Sidebar
st.sidebar.markdown("##### ⚡ One-Click Student Presets")
st.sidebar.caption("Load real-world student profiles instantly to test predictions without typing:")

p_col1, p_col2 = st.sidebar.columns(2)
with p_col1:
    if st.button("👩‍🎓 High Achiever", use_container_width=True):
        st.session_state['active_student'] = PRESETS["High Achiever"].copy()
        st.success("Loaded: High Achiever")
with p_col2:
    if st.button("⚠️ At-Risk", use_container_width=True):
        st.session_state['active_student'] = PRESETS["At-Risk Student"].copy()
        st.warning("Loaded: At-Risk")

if st.sidebar.button("⚖️ Average / Distracted", use_container_width=True):
    st.session_state['active_student'] = PRESETS["Average / Distracted"].copy()
    st.info("Loaded: Average")

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="font-size: 11px; color: #94A3B8; line-height: 1.5;">
    <b>Dataset:</b> UCI Student Performance (Cortez & Silva)<br>
    <b>Predictive Target:</b> Final Mathematics Grade ($G3$)<br>
    <b>Environment:</b> Scikit-Learn • Streamlit
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Reusable Active Student Summary Bar Component
# ---------------------------------------------------------
def render_active_student_summary():
    """Shows a concise, professional summary pill of the currently loaded student profile."""
    s = st.session_state['active_student']
    study_map = {1: '< 2 hrs/wk', 2: '2-5 hrs/wk', 3: '5-10 hrs/wk', 4: '> 10 hrs/wk'}
    st.markdown(f"""
    <div class="student-chip">
        <span style="font-weight: 700; color: #1E293B;">👤 Current Student Profile:</span>
        <span class="chip-item"><b>Term 1 (G1):</b> {s['G1']}/20</span>
        <span class="chip-item"><b>Term 2 (G2):</b> {s['G2']}/20</span>
        <span class="chip-item"><b>Study Time:</b> {study_map.get(s['studytime'], '2-5 hrs/wk')}</span>
        <span class="chip-item"><b>Absences:</b> {s['absences']} days</span>
        <span class="chip-item"><b>Past Failures:</b> {s['failures']}</span>
        <span class="chip-item"><b>Higher Ed Goal:</b> {'Yes' if s['higher']=='yes' else 'No'}</span>
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# PAGE 1: OVERVIEW & WELCOME DASHBOARD
# =========================================================
if st.session_state['page_index'] == 0:
    st.markdown("""
    <div class="hero-banner">
        <span class="hero-pill">✨ AI-Powered Educational Decision Support System</span>
        <h1 class="hero-title">Empowering Student Success Through Predictive Intelligence</h1>
        <p class="hero-subtitle">
            Detect academic risks early, forecast final course grades, discover natural learner behavioral patterns, 
            and generate actionable, personalized study recommendations for teachers, mentors, and students.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Call-to-Action Start Section
    st.markdown("""
    <div class="saas-card" style="background: linear-gradient(to right, #FFFFFF, #F8FAFC); border: 2px solid #C7D2FE;">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 16px;">
            <div>
                <h3 style="margin: 0 0 6px 0; color: #1E1B4B; font-size: 20px;">🚀 Ready to predict a student's academic journey?</h3>
                <p style="margin: 0; color: #475569; font-size: 14px;">
                    Start with a pre-filled student profile or enter custom details to see real-time AI predictions.
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("👉 Get Started — Enter Student Details & Predict Performance", type="primary", use_container_width=True):
        set_page(1)

    st.markdown("### 🧭 How EduPredict AI Works in 4 Steps")
    col_w1, col_w2, col_w3, col_w4 = st.columns(4)

    with col_w1:
        st.markdown("""
        <div class="saas-card" style="height: 100%;">
            <div style="font-size: 24px; margin-bottom: 8px;">📝</div>
            <div style="font-weight: 700; color: #1E293B; margin-bottom: 6px;">1. Enter Details</div>
            <p style="color: #64748B; font-size: 13px; line-height: 1.5; margin: 0;">
                Input past term grades, weekly study hours, attendance records, and lifestyle factors.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col_w2:
        st.markdown("""
        <div class="saas-card" style="height: 100%;">
            <div style="font-size: 24px; margin-bottom: 8px;">📈</div>
            <div style="font-weight: 700; color: #1E293B; margin-bottom: 6px;">2. Predict Grade & Risk</div>
            <p style="color: #64748B; font-size: 13px; line-height: 1.5; margin: 0;">
                Trained AI models forecast final grade (0–20 scale) and compute failure risk probability.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col_w3:
        st.markdown("""
        <div class="saas-card" style="height: 100%;">
            <div style="font-size: 24px; margin-bottom: 8px;">🧩</div>
            <div style="font-weight: 700; color: #1E293B; margin-bottom: 6px;">3. Behavioral Grouping</div>
            <p style="color: #64748B; font-size: 13px; line-height: 1.5; margin: 0;">
                K-Means clustering categorizes the student into archetypes (e.g., High Achiever, At-Risk).
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col_w4:
        st.markdown("""
        <div class="saas-card" style="height: 100%;">
            <div style="font-size: 24px; margin-bottom: 8px;">💡</div>
            <div style="font-weight: 700; color: #1E293B; margin-bottom: 6px;">4. Personalized Plan</div>
            <p style="color: #64748B; font-size: 13px; line-height: 1.5; margin: 0;">
                Receive an expert-generated study roadmap, weekly schedule, and downloadable report.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📊 Platform Benchmark Statistics (Actual Test Results)")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    with kpi1:
        st.markdown("""
        <div class="kpi-card" style="border-top-color: #3B82F6;">
            <div class="kpi-title">Benchmark Dataset</div>
            <div class="kpi-value">395</div>
            <div class="kpi-sub">Students Analyzed (UCI Dataset)</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi2:
        st.markdown("""
        <div class="kpi-card" style="border-top-color: #10B981;">
            <div class="kpi-title">Top Classifier Accuracy</div>
            <div class="kpi-value">92.41%</div>
            <div class="kpi-sub">Stacking Ensemble Classifier</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi3:
        st.markdown("""
        <div class="kpi-card" style="border-top-color: #8B5CF6;">
            <div class="kpi-title">Grade Prediction Fit (R²)</div>
            <div class="kpi-value">0.805</div>
            <div class="kpi-sub">Over 80% Variance Explained</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi4:
        st.markdown("""
        <div class="kpi-card" style="border-top-color: #F59E0B;">
            <div class="kpi-title">Grade Error Margin</div>
            <div class="kpi-value">±1.46</div>
            <div class="kpi-sub">Mean Absolute Error (0-20 scale)</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📂 Dataset Distribution & Academic Context")
    if df_clustered is not None:
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(7, 3.8))
            sns.histplot(df_clustered['G3'], bins=20, kde=True, color='#4F46E5', ax=ax)
            ax.axvline(10, color='#EF4444', linestyle='--', linewidth=2, label='Passing Threshold (10 Marks)')
            ax.set_title("Distribution of Final Mathematics Grade (G3)", fontsize=11, fontweight='bold', pad=12)
            ax.set_xlabel("Final Score (Scale: 0 to 20)", fontsize=10)
            ax.set_ylabel("Student Count", fontsize=10)
            ax.legend(frameon=True)
            sns.despine()
            st.pyplot(fig)
            plt.close(fig)

        with c2:
            fig, ax = plt.subplots(figsize=(7, 3.8))
            risk_counts = [(df_clustered['G3'] >= 10).sum(), (df_clustered['G3'] < 10).sum()]
            labels = ['Passing / Not At Risk (≥ 10)', 'Failing / At Risk (< 10)']
            colors = ['#10B981', '#EF4444']
            ax.pie(risk_counts, labels=labels, autopct='%1.1f%%', colors=colors, startangle=140, explode=(0, 0.08),
                   textprops={'fontsize': 10, 'weight': 'bold'})
            ax.set_title("Historical Student Pass / Fail Distribution", fontsize=11, fontweight='bold', pad=12)
            st.pyplot(fig)
            plt.close(fig)


# =========================================================
# PAGE 2: ENTER STUDENT DETAILS & PREDICT GRADE
# =========================================================
elif st.session_state['page_index'] == 1:
    st.markdown("""
    <div class="guide-box">
        <div class="guide-title">
            <span>📍 Step 1 of 5</span> • <span>Enter Student Information & Predict Final Grade</span>
        </div>
        <p class="guide-desc">
            <b>What to do here:</b> Review or modify the student's attributes below (or pick a Quick Preset). 
            Then click <b>"🚀 Predict Final Grade (G3)"</b> to compute the forecasted score using our trained regression models.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📝 Student Information Input Form")

    # Quick Presets Buttons right on the form
    st.caption("⚡ Quick-Fill Presets (Click any button to populate the form instantly):")
    b1, b2, b3 = st.columns(3)
    with b1:
        if st.button("👩‍🎓 High Achiever Preset", key="form_p1", use_container_width=True):
            st.session_state['active_student'] = PRESETS["High Achiever"].copy()
            st.rerun()
    with b2:
        if st.button("⚠️ At-Risk Student Preset", key="form_p2", use_container_width=True):
            st.session_state['active_student'] = PRESETS["At-Risk Student"].copy()
            st.rerun()
    with b3:
        if st.button("⚖️ Average / Distracted Preset", key="form_p3", use_container_width=True):
            st.session_state['active_student'] = PRESETS["Average / Distracted"].copy()
            st.rerun()

    current_s = st.session_state['active_student']

    # Clean 3-Tab Form Organization
    tab_acad, tab_demo, tab_life = st.tabs([
        "🎓 1. Academic Performance & Study Habits",
        "👤 2. Demographics & Educational Goals",
        "🏡 3. Family Background & Lifestyle"
    ])

    with tab_acad:
        st.markdown("##### Prior Term Grades & Effort Commitment")
        col_a1, col_a2 = st.columns(2)
        with col_a1:
            g1 = st.slider("Term 1 Exam Grade (G1) [0 to 20 scale]", 0, 20, int(current_s.get('G1', 10)),
                           help="Portuguese grading scale: 0-9 Fail, 10-20 Pass")
            g2 = st.slider("Term 2 Exam Grade (G2) [0 to 20 scale]", 0, 20, int(current_s.get('G2', 10)),
                           help="Second term score is strongly correlated with final performance")
            studytime = st.selectbox(
                "Weekly Dedicated Study Hours",
                options=[1, 2, 3, 4],
                index=[1, 2, 3, 4].index(current_s.get('studytime', 2)),
                format_func=lambda x: {1: "Less than 2 hours/week", 2: "2 to 5 hours/week", 3: "5 to 10 hours/week", 4: "More than 10 hours/week"}[x]
            )
        with col_a2:
            failures = st.slider("Past Course Failures", 0, 4, int(current_s.get('failures', 0)),
                                 help="Number of times student previously failed this subject")
            absences = st.slider("Total Missed School Days (Absences)", 0, 93, int(current_s.get('absences', 4)),
                                 help="High absenteeism is a strong risk indicator")
            schoolsup = st.selectbox("School-Sponsored Extra Tutoring", ["yes", "no"],
                                     index=0 if current_s.get('schoolsup', 'no') == 'yes' else 1,
                                     format_func=lambda x: "Yes (Receiving School Support)" if x == "yes" else "No (No School Support)")
            paid = st.selectbox("Extra Paid Mathematics Classes", ["yes", "no"],
                                index=0 if current_s.get('paid', 'no') == 'yes' else 1,
                                format_func=lambda x: "Yes (Private Tutoring)" if x == "yes" else "No (No Private Tutoring)")

    with tab_demo:
        st.markdown("##### Personal Background & Living Environment")
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            school = st.selectbox("School Attended", ["GP", "MS"],
                                  index=0 if current_s.get('school', 'GP') == 'GP' else 1,
                                  format_func=lambda x: "Gabriel Pereira (GP)" if x == "GP" else "Mousinho da Silveira (MS)")
            sex = st.selectbox("Gender", ["F", "M"],
                               index=0 if current_s.get('sex', 'F') == 'F' else 1,
                               format_func=lambda x: "Female" if x == "F" else "Male")
            age = st.slider("Age (Years)", 15, 22, int(current_s.get('age', 17)))
            address = st.selectbox("Residential Area", ["U", "R"],
                                   index=0 if current_s.get('address', 'U') == 'U' else 1,
                                   format_func=lambda x: "Urban / City (U)" if x == "U" else "Rural (R)")
        with col_d2:
            famsize = st.selectbox("Family Size", ["GT3", "LE3"],
                                   index=0 if current_s.get('famsize', 'GT3') == 'GT3' else 1,
                                   format_func=lambda x: "More than 3 members (> 3)" if x == "GT3" else "3 or fewer members (≤ 3)")
            pstatus = st.selectbox("Parent Living Arrangement", ["T", "A"],
                                   index=0 if current_s.get('Pstatus', 'T') == 'T' else 1,
                                   format_func=lambda x: "Living Together (T)" if x == "T" else "Apart / Separated (A)")
            higher = st.selectbox("Aspirations to Attend University / Higher Ed", ["yes", "no"],
                                  index=0 if current_s.get('higher', 'yes') == 'yes' else 1,
                                  format_func=lambda x: "Yes (Plans for Higher Education)" if x == "yes" else "No")
            internet = st.selectbox("Home Internet Access", ["yes", "no"],
                                    index=0 if current_s.get('internet', 'yes') == 'yes' else 1,
                                    format_func=lambda x: "Yes (Has Internet at Home)" if x == "yes" else "No")

    with tab_life:
        st.markdown("##### Family Support & Social Lifestyle Habits")
        col_l1, col_l2 = st.columns(2)
        with col_l1:
            medu = st.selectbox("Mother's Education Level", [0, 1, 2, 3, 4],
                                index=int(current_s.get('Medu', 2)),
                                format_func=lambda x: {0: "None", 1: "Primary (4th grade)", 2: "5th to 9th grade", 3: "Secondary Education", 4: "Higher Education / University"}[x])
            fedu = st.selectbox("Father's Education Level", [0, 1, 2, 3, 4],
                                index=int(current_s.get('Fedu', 2)),
                                format_func=lambda x: {0: "None", 1: "Primary (4th grade)", 2: "5th to 9th grade", 3: "Secondary Education", 4: "Higher Education / University"}[x])
            mjob = st.selectbox("Mother's Occupation", ["teacher", "health", "services", "at_home", "other"],
                                index=["teacher", "health", "services", "at_home", "other"].index(current_s.get('Mjob', 'other')))
            fjob = st.selectbox("Father's Occupation", ["teacher", "health", "services", "at_home", "other"],
                                index=["teacher", "health", "services", "at_home", "other"].index(current_s.get('Fjob', 'other')))
            reason = st.selectbox("Reason for Choosing School", ["home", "reputation", "course", "other"],
                                  index=["home", "reputation", "course", "other"].index(current_s.get('reason', 'course')))
            guardian = st.selectbox("Primary Legal Guardian", ["mother", "father", "other"],
                                    index=["mother", "father", "other"].index(current_s.get('guardian', 'mother')))
            famsup = st.selectbox("Family Educational Support", ["yes", "no"],
                                  index=0 if current_s.get('famsup', 'yes') == 'yes' else 1)
        with col_l2:
            activities = st.selectbox("Extracurricular Activities", ["yes", "no"],
                                      index=0 if current_s.get('activities', 'no') == 'yes' else 1)
            nursery = st.selectbox("Attended Nursery / Preschool", ["yes", "no"],
                                   index=0 if current_s.get('nursery', 'yes') == 'yes' else 1)
            romantic = st.selectbox("In a Romantic Relationship", ["yes", "no"],
                                    index=0 if current_s.get('romantic', 'no') == 'yes' else 1)
            traveltime = st.selectbox("Commute Time to School", [1, 2, 3, 4],
                                      index=int(current_s.get('traveltime', 1)) - 1,
                                      format_func=lambda x: {1: "< 15 minutes", 2: "15 to 30 minutes", 3: "30 to 60 minutes", 4: "> 60 minutes"}[x])
            famrel = st.slider("Family Relationship Quality [1=Very Poor to 5=Excellent]", 1, 5, int(current_s.get('famrel', 4)))
            freetime = st.slider("Free Time After Classes [1=Very Low to 5=Very High]", 1, 5, int(current_s.get('freetime', 3)))
            goout = st.slider("Going Out with Friends [1=Rarely to 5=Very Frequent]", 1, 5, int(current_s.get('goout', 3)))
            dalc = st.slider("Weekday Alcohol Intake [1=None/Very Low to 5=High]", 1, 5, int(current_s.get('Dalc', 1)))
            walc = st.slider("Weekend Alcohol Intake [1=None/Very Low to 5=High]", 1, 5, int(current_s.get('Walc', 1)))
            health = st.slider("Current Health Status [1=Very Poor to 5=Very Good]", 1, 5, int(current_s.get('health', 4)))

    # Save form inputs into session state
    st.session_state['active_student'] = {
        'school': school, 'sex': sex, 'age': age, 'address': address, 'famsize': famsize,
        'Pstatus': pstatus, 'Medu': medu, 'Fedu': fedu, 'Mjob': mjob, 'Fjob': fjob,
        'reason': reason, 'guardian': guardian, 'traveltime': traveltime, 'studytime': studytime,
        'failures': failures, 'schoolsup': schoolsup, 'famsup': famsup, 'paid': paid,
        'activities': activities, 'nursery': nursery, 'higher': higher, 'internet': internet,
        'romantic': romantic, 'famrel': famrel, 'freetime': freetime, 'goout': goout,
        'Dalc': dalc, 'Walc': walc, 'health': health, 'absences': absences, 'G1': g1, 'G2': g2
    }

    st.markdown("---")
    predict_clicked = st.button("🚀 Predict Final Grade (G3)", type="primary", use_container_width=True)

    if predict_clicked or 'last_reg_pred' in st.session_state:
        input_df = pd.DataFrame([st.session_state['active_student']])
        X_proc = preprocessor.transform(input_df)

        pred_lr = float(reg_models['Linear Regression'].predict(X_proc)[0])
        pred_sgd = float(reg_models['Gradient Descent (SGD)'].predict(X_proc)[0])

        pred_lr = max(0.0, min(20.0, pred_lr))
        pred_sgd = max(0.0, min(20.0, pred_sgd))
        avg_score = (pred_lr + pred_sgd) / 2.0
        st.session_state['last_reg_pred'] = avg_score

        st.markdown("### 🎯 Predicted Final Examination Performance")

        res1, res2, res3 = st.columns(3)
        l_code, l_desc, l_badge = get_letter_grade(avg_score)

        with res1:
            st.markdown(f"""
            <div class="saas-card" style="border-left: 5px solid #3B82F6;">
                <div class="kpi-title">Linear Regression Model</div>
                <div class="kpi-value">{pred_lr:.2f} <span style="font-size: 16px; color: #64748B;">/ 20</span></div>
                <div class="kpi-sub">Standard Ordinary Least Squares</div>
            </div>
            """, unsafe_allow_html=True)

        with res2:
            st.markdown(f"""
            <div class="saas-card" style="border-left: 5px solid #8B5CF6;">
                <div class="kpi-title">Gradient Descent (SGD)</div>
                <div class="kpi-value">{pred_sgd:.2f} <span style="font-size: 16px; color: #64748B;">/ 20</span></div>
                <div class="kpi-sub">Iteratively Optimized Model</div>
            </div>
            """, unsafe_allow_html=True)

        with res3:
            border_c = "#10B981" if avg_score >= 10 else "#EF4444"
            st.markdown(f"""
            <div class="saas-card" style="border-left: 5px solid {border_c};">
                <div class="kpi-title">Consensus Predicted Grade</div>
                <div class="kpi-value">{avg_score:.2f} <span style="font-size: 16px; color: #64748B;">/ 20</span></div>
                <div class="kpi-sub">
                    <span class="badge-pill {l_badge}">{l_code} ({l_desc})</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Graphical Comparison
        st.markdown("##### 📊 Grade Progression Trajectory")
        fig, ax = plt.subplots(figsize=(8, 2.6))
        categories = ['Term 1 (G1)', 'Term 2 (G2)', 'AI Predicted Final (G3)']
        scores = [st.session_state['active_student']['G1'], st.session_state['active_student']['G2'], avg_score]
        colors = ['#94A3B8', '#64748B', '#10B981' if avg_score >= 10 else '#EF4444']

        bars = ax.barh(categories, scores, color=colors, height=0.5)
        ax.set_xlim(0, 20)
        ax.axvline(10, color='#EF4444', linestyle='--', linewidth=1.5, label='Passing Threshold (10 Marks)')

        for bar in bars:
            w = bar.get_width()
            ax.text(w + 0.3, bar.get_y() + bar.get_height()/2, f'{w:.1f} / 20', va='center', fontweight='bold', fontsize=10)

        ax.set_xlabel("Score (Scale: 0 to 20)", fontsize=10)
        ax.legend(loc='lower right', frameon=True)
        sns.despine()
        st.pyplot(fig)
        plt.close(fig)

        st.markdown(f"""
        <div class="callout-box">
            <b>💡 How to understand this result:</b><br>
            A score of <b>10.0 or above</b> means the student is on track to pass. A score below 10 indicates academic risk.
            The student's Term 1 score was <b>{st.session_state['active_student']['G1']}</b>, Term 2 was <b>{st.session_state['active_student']['G2']}</b>, 
            and the model predicts a final outcome of <b>{avg_score:.2f}</b>.
        </div>
        """, unsafe_allow_html=True)

    # Next Step Guidance Card
    st.markdown("""
    <div class="next-step-box">
        <div>
            <div class="next-step-title">👉 Ready for Step 2?</div>
            <p class="next-step-desc">Evaluate the student's risk probability across 7 diverse AI classification models.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("👉 Proceed to Step 2: Analyze Academic Failure Risk", type="primary", use_container_width=True):
        set_page(2)


# =========================================================
# PAGE 3: ACADEMIC RISK CLASSIFICATION & ENSEMBLES
# =========================================================
elif st.session_state['page_index'] == 2:
    st.markdown("""
    <div class="guide-box">
        <div class="guide-title">
            <span>📍 Step 2 of 5</span> • <span>Academic Failure Risk Classification</span>
        </div>
        <p class="guide-desc">
            <b>What to do here:</b> We evaluate whether the student is <b>At Risk of Failure (Final Grade < 10)</b> 
            using <b>7 different AI algorithms</b> (4 individual models + 3 ensemble architectures).
        </p>
    </div>
    """, unsafe_allow_html=True)

    render_active_student_summary()

    input_df = pd.DataFrame([st.session_state['active_student']])
    X_proc = preprocessor.transform(input_df)

    clf_preds = {}
    clf_probs = {}

    for name, model in clf_models.items():
        pred = model.predict(X_proc)[0]
        prob = model.predict_proba(X_proc)[0][1] if hasattr(model, 'predict_proba') else (1.0 if pred == 1 else 0.0)
        clf_preds[name] = int(pred)
        clf_probs[name] = float(prob)

    risk_votes = sum(clf_preds.values())
    avg_risk_prob = np.mean(list(clf_probs.values()))
    stacking_pred = clf_preds['Stacking Classifier']
    stacking_prob = clf_probs['Stacking Classifier']

    st.markdown("### ⚠️ Risk Assessment Consensus")
    rc1, rc2, rc3 = st.columns(3)

    with rc1:
        is_risk = avg_risk_prob >= 0.5
        badge_style = "badge-risk" if is_risk else "badge-pass"
        status_text = "AT RISK (Needs Support)" if is_risk else "NOT AT RISK (Passing)"
        border_c = "#EF4444" if is_risk else "#10B981"
        st.markdown(f"""
        <div class="saas-card" style="border-left: 5px solid {border_c};">
            <div class="kpi-title">Multi-Model Consensus</div>
            <div class="kpi-value" style="font-size: 22px;">{status_text}</div>
            <div class="kpi-sub">{risk_votes} of 7 AI models flagged this student as at-risk</div>
        </div>
        """, unsafe_allow_html=True)

    with rc2:
        st.markdown(f"""
        <div class="saas-card" style="border-left: 5px solid #F59E0B;">
            <div class="kpi-title">Average Risk Probability</div>
            <div class="kpi-value">{avg_risk_prob * 100:.1f}%</div>
            <div class="kpi-sub">Likelihood of scoring below 10 marks</div>
        </div>
        """, unsafe_allow_html=True)

    with rc3:
        st_border = "#EF4444" if stacking_pred == 1 else "#10B981"
        st.markdown(f"""
        <div class="saas-card" style="border-left: 5px solid {st_border};">
            <div class="kpi-title">Top Model Verdict (Stacking)</div>
            <div class="kpi-value" style="font-size: 22px;">{'At Risk' if stacking_pred == 1 else 'Not At Risk'}</div>
            <div class="kpi-sub">Confidence: <b>{stacking_prob * 100:.1f}%</b> failure probability</div>
        </div>
        """, unsafe_allow_html=True)

    # Visual Probability Bar Chart
    st.markdown("##### 📊 Model-by-Model Failure Risk Probability")
    fig, ax = plt.subplots(figsize=(9, 3.6))
    model_names = list(clf_probs.keys())
    probs = [clf_probs[m] * 100 for m in model_names]
    bar_colors = ['#EF4444' if p >= 50 else '#10B981' for p in probs]

    bars = ax.bar(model_names, probs, color=bar_colors, edgecolor='none', width=0.55)
    ax.axhline(50, color='#EF4444', linestyle='--', linewidth=1.5, label='Risk Threshold (50%)')
    ax.set_ylabel("Failure Risk Probability (%)", fontsize=10)
    ax.set_ylim(0, 105)
    plt.xticks(rotation=20, ha='right', fontsize=9.5)

    for bar in bars:
        h = bar.get_height()
        ax.annotate(f'{h:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.legend(frameon=True)
    sns.despine()
    st.pyplot(fig)
    plt.close(fig)

    # Detailed Table of Predictions
    st.markdown("##### 🔬 Individual vs. Ensemble Breakdown")
    rows = []
    for name, pred in clf_preds.items():
        is_ens = name in ['Voting Classifier', 'AdaBoost', 'Stacking Classifier']
        p = clf_probs[name]
        status = "⚠️ At Risk" if pred == 1 else "✅ Not At Risk"
        rows.append({
            "AI Algorithm": name,
            "Architecture": "Ensemble (Multi-Model)" if is_ens else "Individual Classifier",
            "Classification Result": status,
            "Failure Probability": f"{p * 100:.1f}%",
            "Confidence Level": "High Confidence" if (p > 0.75 or p < 0.25) else "Moderate / Borderline"
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.markdown("""
    <div class="callout-box">
        <b>💡 Why use multiple models?</b><br>
        In high-stakes educational decision-making, relying on a single model can overlook vulnerable students. 
        Ensemble methods (like Stacking and AdaBoost) combine the individual strengths of SVM, Trees, and Neighbors 
        to produce the most reliable prediction.
    </div>
    """, unsafe_allow_html=True)

    # Next Step Guidance
    st.markdown("""
    <div class="next-step-box">
        <div>
            <div class="next-step-title">👉 Ready for Step 3?</div>
            <p class="next-step-desc">Inspect exact scientific evaluation benchmarks (Accuracy, Precision, Recall, F1) on real test data.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("👉 Proceed to Step 3: Compare Model Benchmarks & Metrics", type="primary", use_container_width=True):
        set_page(3)


# =========================================================
# PAGE 4: MODEL COMPARISON & BENCHMARKS
# =========================================================
elif st.session_state['page_index'] == 3:
    st.markdown("""
    <div class="guide-box">
        <div class="guide-title">
            <span>📍 Step 3 of 5</span> • <span>Objective AI Model Benchmarks (Real Test Data)</span>
        </div>
        <p class="guide-desc">
            <b>What to do here:</b> Review and compare actual empirical performance metrics calculated on 
            <b>79 unseen student test records</b> with zero data leakage. All numbers represent real test data.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Beginner Friendly Glossary
    with st.expander("📘 Plain-English Metric Guide (Click to expand)", expanded=False):
        g1, g2 = st.columns(2)
        with g1:
            st.markdown("""
            - **Accuracy**: Percentage of test students correctly classified as passing or failing.
            - **Recall (Sensitivity)**: Critical in education! Measures what percentage of *actual struggling students* were successfully caught.
            """)
        with g2:
            st.markdown("""
            - **Precision**: Measures how trustworthy the alarm is when the system flags a student as at-risk.
            - **RMSE (Root Mean Squared Error)**: Average difference between predicted grade and real grade in marks (lower is better).
            """)

    tab_c1, tab_c2, tab_c3 = st.tabs([
        "🏆 Classification & Ensemble Benchmarks",
        "📈 Grade Prediction (Regression) Benchmarks",
        "🔲 Interactive Confusion Matrix Explorer"
    ])

    with tab_c1:
        st.markdown("##### 7-Model Classification Benchmark Leaderboard")
        clf_metrics = eval_results['classification']
        table_rows = []
        for m, res in clf_metrics.items():
            table_rows.append({
                "Algorithm": m,
                "Type": res.get('Category', 'Individual'),
                "Accuracy (%)": f"{res['Accuracy']*100:.2f}%",
                "Recall / Catch Rate (%)": f"{res['Recall']*100:.2f}%",
                "Precision (%)": f"{res['Precision']*100:.2f}%",
                "F1-Score": f"{res['F1-Score']:.3f}"
            })
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True)

        # Bar chart comparison
        fig, ax = plt.subplots(figsize=(10, 4.2))
        models = list(clf_metrics.keys())
        accs = [clf_metrics[m]['Accuracy'] * 100 for m in models]
        f1s = [clf_metrics[m]['F1-Score'] * 100 for m in models]

        x = np.arange(len(models))
        width = 0.35

        ax.bar(x - width/2, accs, width, label='Accuracy (%)', color='#3B82F6')
        ax.bar(x + width/2, f1s, width, label='F1-Score (%)', color='#F59E0B')

        ax.set_ylabel('Score (%)', fontsize=10)
        ax.set_title('Accuracy vs F1-Score Across Individual & Ensemble Models', fontweight='bold', pad=12)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=20, ha='right', fontsize=9.5)
        ax.set_ylim(50, 100)
        ax.legend(frameon=True)
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        sns.despine()

        for i in x:
            ax.text(i - width/2, accs[i] + 1, f"{accs[i]:.1f}%", ha='center', fontsize=8, fontweight='bold')
            ax.text(i + width/2, f1s[i] + 1, f"{f1s[i]:.1f}%", ha='center', fontsize=8, fontweight='bold')

        st.pyplot(fig)
        plt.close(fig)

        st.info("💡 **Key Takeaway:** Stacking Classifier achieved the highest overall accuracy (**92.41%**) and recall (**92.31%**), catching 24 out of 26 struggling test students.")

    with tab_c2:
        st.markdown("##### Continuous Grade Prediction (Regression) Results")
        reg_metrics = eval_results['regression']
        reg_df = pd.DataFrame([
            {
                "Model": m,
                "MAE (Average Error)": f"{res['MAE']:.3f} marks",
                "RMSE (Root Mean Squared Error)": f"{res['RMSE']:.3f} marks",
                "R² Score (Goodness of Fit)": f"{res['R2']:.3f}"
            }
            for m, res in reg_metrics.items()
        ])
        st.dataframe(reg_df, use_container_width=True)

        col_rb1, col_rb2 = st.columns(2)
        with col_rb1:
            fig, ax = plt.subplots(figsize=(5, 3))
            r2_scores = [res['R2'] for res in reg_metrics.values()]
            ax.bar(list(reg_metrics.keys()), r2_scores, color=['#3B82F6', '#8B5CF6'], width=0.45)
            ax.set_ylim(0, 1.0)
            ax.set_ylabel("R² Score (Higher is Better)")
            ax.set_title("R² Score (Variance Explained)", fontweight='bold')
            for i, v in enumerate(r2_scores):
                ax.text(i, v + 0.02, f"{v:.3f}", ha='center', fontweight='bold')
            sns.despine()
            st.pyplot(fig)
            plt.close(fig)

        with col_rb2:
            fig, ax = plt.subplots(figsize=(5, 3))
            rmse_scores = [res['RMSE'] for res in reg_metrics.values()]
            ax.bar(list(reg_metrics.keys()), rmse_scores, color=['#10B981', '#F59E0B'], width=0.45)
            ax.set_ylim(0, 3.0)
            ax.set_ylabel("RMSE in Marks (Lower is Better)")
            ax.set_title("RMSE (Prediction Error)", fontweight='bold')
            for i, v in enumerate(rmse_scores):
                ax.text(i, v + 0.08, f"{v:.3f}", ha='center', fontweight='bold')
            sns.despine()
            st.pyplot(fig)
            plt.close(fig)

    with tab_c3:
        st.markdown("##### Confusion Matrix Inspection")
        selected_model = st.selectbox("Choose Model to Inspect:", list(clf_models.keys()), index=6)

        cm = np.array(clf_metrics[selected_model]['Confusion_Matrix'])
        tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]

        c_col1, c_col2 = st.columns([1.2, 1])
        with c_col1:
            fig, ax = plt.subplots(figsize=(5, 3.8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                        xticklabels=['Passed (0)', 'At Risk (1)'],
                        yticklabels=['Passed (0)', 'At Risk (1)'], ax=ax,
                        annot_kws={'fontsize': 14, 'weight': 'bold'})
            ax.set_xlabel('AI Predicted Label', fontweight='bold')
            ax.set_ylabel('Actual Ground Truth', fontweight='bold')
            ax.set_title(f'Confusion Matrix: {selected_model}', fontweight='bold')
            st.pyplot(fig)
            plt.close(fig)

        with c_col2:
            st.markdown(f"""
            <div class="saas-card">
                <div style="font-weight: 700; color: #1E293B; margin-bottom: 8px;">Breakdown for {selected_model}:</div>
                <div style="font-size: 13px; line-height: 1.8; color: #475569;">
                    • <b>{tp} At-Risk students caught</b> (True Positives)<br>
                    • <b>{tn} Passing students confirmed</b> (True Negatives)<br>
                    • <b>{fp} False alarms</b> (Passing student flagged)<br>
                    • <b>{fn} Missed at-risk cases</b> (Dangerous Type II Error)<br>
                    • <b>Recall / Safety Rate:</b> <code>{tp/(tp+fn)*100:.1f}%</code>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Next Step Guidance
    st.markdown("""
    <div class="next-step-box">
        <div>
            <div class="next-step-title">👉 Ready for Step 4?</div>
            <p class="next-step-desc">Explore unsupervised student behavioral archetypes discovered by K-Means clustering.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("👉 Proceed to Step 4: Explore Student Behavioral Groups", type="primary", use_container_width=True):
        set_page(4)


# =========================================================
# PAGE 5: STUDENT CLUSTERING (BEHAVIORAL PERSONAS)
# =========================================================
elif st.session_state['page_index'] == 4:
    st.markdown("""
    <div class="guide-box">
        <div class="guide-title">
            <span>📍 Step 4 of 5</span> • <span>Unsupervised Student Behavioral Clustering (K-Means)</span>
        </div>
        <p class="guide-desc">
            <b>What to do here:</b> Without using grades as labels, K-Means clustering uncovered <b>3 natural archetypes</b> 
            of students based on their lifestyle, study time, alcohol habits, and attendance patterns.
        </p>
    </div>
    """, unsafe_allow_html=True)

    render_active_student_summary()

    c_meta = eval_results['clustering']
    personas = c_meta['personas']
    profiles = c_meta['profiles']

    st.markdown("### 👥 The 3 Discovered Student Archetypes")
    p1, p2, p3 = st.columns(3)

    with p1:
        st.markdown(f"""
        <div class="saas-card" style="border-top: 4px solid #EF4444; height: 100%;">
            <span class="badge-pill badge-risk">Cluster 0 Persona</span>
            <div style="font-weight: 800; font-size: 17px; color: #991B1B; margin: 8px 0 4px 0;">{personas[0]['name']}</div>
            <p style="font-size: 13px; color: #475569; margin-bottom: 12px;">{personas[0]['desc']}</p>
            <div style="font-size: 12.5px; line-height: 1.7; background: #FEF2F2; padding: 10px; border-radius: 8px;">
                <b>Avg Final Grade:</b> {profiles[0]['G3']:.1f} / 20<br>
                <b>Avg Study Time:</b> ~{profiles[0]['studytime']:.1f} (2-5 hrs/wk)<br>
                <b>Avg Failures:</b> {profiles[0]['failures']:.2f}<br>
                <b>Avg Absences:</b> {profiles[0]['absences']:.1f} days
            </div>
        </div>
        """, unsafe_allow_html=True)

    with p2:
        st.markdown(f"""
        <div class="saas-card" style="border-top: 4px solid #F59E0B; height: 100%;">
            <span class="badge-pill badge-warn">Cluster 1 Persona</span>
            <div style="font-weight: 800; font-size: 17px; color: #B45309; margin: 8px 0 4px 0;">{personas[1]['name']}</div>
            <p style="font-size: 13px; color: #475569; margin-bottom: 12px;">{personas[1]['desc']}</p>
            <div style="font-size: 12.5px; line-height: 1.7; background: #FFFBEB; padding: 10px; border-radius: 8px;">
                <b>Avg Final Grade:</b> {profiles[1]['G3']:.1f} / 20<br>
                <b>Avg Absences:</b> {profiles[1]['absences']:.1f} days (Highest!)<br>
                <b>Weekend Alcohol:</b> {profiles[1]['Walc']:.1f} / 5 (Highest!)<br>
                <b>Socializing (Go Out):</b> {profiles[1]['goout']:.1f} / 5
            </div>
        </div>
        """, unsafe_allow_html=True)

    with p3:
        st.markdown(f"""
        <div class="saas-card" style="border-top: 4px solid #10B981; height: 100%;">
            <span class="badge-pill badge-pass">Cluster 2 Persona</span>
            <div style="font-weight: 800; font-size: 17px; color: #065F46; margin: 8px 0 4px 0;">{personas[2]['name']}</div>
            <p style="font-size: 13px; color: #475569; margin-bottom: 12px;">{personas[2]['desc']}</p>
            <div style="font-size: 12.5px; line-height: 1.7; background: #ECFDF5; padding: 10px; border-radius: 8px;">
                <b>Avg Final Grade:</b> {profiles[2]['G3']:.1f} / 20 (Distinction)<br>
                <b>Avg Failures:</b> {profiles[2]['failures']:.2f} (Near Zero)<br>
                <b>Avg Study Time:</b> Consistent discipline<br>
                <b>Alcohol Intake:</b> Low
            </div>
        </div>
        """, unsafe_allow_html=True)

    # 2D PCA Cluster Map
    st.markdown("---")
    st.markdown("### 🗺️ 2D PCA Map of Student Behavioral Segments")
    if df_clustered is not None:
        fig, ax = plt.subplots(figsize=(9, 4.4))
        palette = {0: '#EF4444', 1: '#F59E0B', 2: '#10B981'}
        sns.scatterplot(
            data=df_clustered,
            x='pca_1', y='pca_2',
            hue='cluster',
            palette=palette,
            s=55, alpha=0.85, ax=ax
        )
        ax.set_title("Student Segments Visualized via Principal Component Analysis (PCA)", fontweight='bold', pad=12)
        ax.set_xlabel("Principal Component 1 (Academic Attainment & Consistency)", fontsize=10)
        ax.set_ylabel("Principal Component 2 (Social, Absenteeism & Alcohol Factors)", fontsize=10)
        ax.legend(title="Archetype", labels=[personas[0]['name'], personas[1]['name'], personas[2]['name']], frameon=True)
        sns.despine()
        st.pyplot(fig)
        plt.close(fig)

    # Which cluster does the current student belong to?
    st.markdown("### 📍 Which Group Does the Current Student Match?")
    cl_student = st.session_state['active_student']
    est_g3 = (cl_student['G1'] + cl_student['G2']) / 2.0
    cl_features = ['studytime', 'failures', 'absences', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'G1', 'G2', 'G3']
    cl_df = pd.DataFrame([{
        'studytime': cl_student['studytime'],
        'failures': cl_student['failures'],
        'absences': cl_student['absences'],
        'freetime': cl_student['freetime'],
        'goout': cl_student['goout'],
        'Dalc': cl_student['Dalc'],
        'Walc': cl_student['Walc'],
        'health': cl_student['health'],
        'G1': cl_student['G1'],
        'G2': cl_student['G2'],
        'G3': est_g3
    }], columns=cl_features)

    scaled_cl = clustering_artifacts['scaler'].transform(cl_df)
    assigned_cluster = int(clustering_artifacts['model'].predict(scaled_cl)[0])

    badge_map = {0: "badge-risk", 1: "badge-warn", 2: "badge-pass"}
    st.markdown(f"""
    <div class="saas-card" style="border: 2px solid #C7D2FE;">
        <span class="badge-pill {badge_map[assigned_cluster]}">Assigned Archetype: Cluster {assigned_cluster}</span>
        <div style="font-size: 20px; font-weight: 800; color: #1E1B4B; margin: 8px 0;">{personas[assigned_cluster]['name']}</div>
        <p style="color: #475569; font-size: 14px; margin: 0;">{personas[assigned_cluster]['desc']}</p>
    </div>
    """, unsafe_allow_html=True)

    # Next Step Guidance
    st.markdown("""
    <div class="next-step-box">
        <div>
            <div class="next-step-title">👉 Ready for the Final Step?</div>
            <p class="next-step-desc">Synthesize all predictions into a concrete, personalized pedagogical action plan & weekly timetable.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("👉 Proceed to Step 5: Generate Study Action Plan & Timetable", type="primary", use_container_width=True):
        set_page(5)


# =========================================================
# PAGE 6: ACTION PLAN & STUDY RECOMMENDATIONS
# =========================================================
elif st.session_state['page_index'] == 5:
    st.markdown("""
    <div class="guide-box">
        <div class="guide-title">
            <span>📍 Step 5 of 5</span> • <span>Rule-Based Expert Study Recommendations</span>
        </div>
        <p class="guide-desc">
            <b>What to do here:</b> Our expert pedagogical engine evaluates the student's predicted grade, failure risk, 
            attendance, and habits to build an individualized remediation plan and structured weekly study schedule.
        </p>
    </div>
    """, unsafe_allow_html=True)

    render_active_student_summary()

    # Pre-generate predictions for expert engine
    input_df = pd.DataFrame([st.session_state['active_student']])
    X_proc = preprocessor.transform(input_df)
    pred_g3 = float(reg_models['Linear Regression'].predict(X_proc)[0])
    pred_risk = int(clf_models['Stacking Classifier'].predict(X_proc)[0])

    dossier = generate_recommendations(
        student_data=st.session_state['active_student'],
        predicted_g3=pred_g3,
        is_at_risk=(pred_risk == 1)
    )

    # Urgency Banner
    urgency_border = "#EF4444" if dossier['urgency_level'] == 'danger' else ("#F59E0B" if dossier['urgency_level'] == 'warning' else "#10B981")
    st.markdown(f"""
    <div class="saas-card" style="border-left: 6px solid {urgency_border};">
        <div class="kpi-title">Pedagogical Intervention Status</div>
        <div class="kpi-value" style="font-size: 22px;">{dossier['urgency']}</div>
        <div class="kpi-sub">
            Target Predicted Grade: <b>{dossier['predicted_g3']:.1f} / 20</b> • 
            Risk Status: <b>{'At Risk of Course Failure' if dossier['is_at_risk'] else 'Not At Risk (Passing)'}</b>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Side-by-side Strengths vs Warning Flags
    sw1, sw2 = st.columns(2)
    with sw1:
        st.markdown("##### ✅ Protective Factors & Strengths")
        for s in dossier['strengths']:
            st.success(f"• {s}")
    with sw2:
        st.markdown("##### ⚠️ Risk Factors & Warning Flags")
        for w in dossier['warnings']:
            st.error(f"• {w}")

    st.markdown("---")
    st.markdown("### 🎯 Recommended Intervention Plan")

    act_col1, act_col2 = st.columns(2)
    with act_col1:
        st.markdown("##### 📚 Academic Remediation Actions")
        for a in dossier['academic_actions']:
            st.markdown(f"""
            <div class="saas-card" style="padding: 14px 16px; margin-bottom: 10px;">
                <span class="badge-pill {'badge-risk' if a['priority']=='Critical' else 'badge-warn'}">{a['priority']} Priority</span>
                <div style="font-weight: 700; color: #1E293B; margin-top: 4px;">{a['title']}</div>
                <div style="font-size: 13px; color: #475569; margin-top: 2px;">{a['detail']}</div>
            </div>
            """, unsafe_allow_html=True)

        if dossier['attendance_actions']:
            st.markdown("##### ⏰ Attendance & Punctuality Actions")
            for a in dossier['attendance_actions']:
                st.markdown(f"""
                <div class="saas-card" style="padding: 14px 16px; margin-bottom: 10px;">
                    <span class="badge-pill badge-risk">{a['priority']} Priority</span>
                    <div style="font-weight: 700; color: #1E293B; margin-top: 4px;">{a['title']}</div>
                    <div style="font-size: 13px; color: #475569; margin-top: 2px;">{a['detail']}</div>
                </div>
                """, unsafe_allow_html=True)

    with act_col2:
        st.markdown("##### 🧘 Lifestyle, Wellness & Support Actions")
        if dossier['lifestyle_actions']:
            for a in dossier['lifestyle_actions']:
                st.markdown(f"""
                <div class="saas-card" style="padding: 14px 16px; margin-bottom: 10px;">
                    <span class="badge-pill badge-warn">{a['priority']} Priority</span>
                    <div style="font-weight: 700; color: #1E293B; margin-top: 4px;">{a['title']}</div>
                    <div style="font-size: 13px; color: #475569; margin-top: 2px;">{a['detail']}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Current lifestyle, sleep, and social habits are well-balanced.")

        if dossier['support_actions']:
            st.markdown("##### 🤝 Institutional & Community Support")
            for a in dossier['support_actions']:
                st.markdown(f"""
                <div class="saas-card" style="padding: 14px 16px; margin-bottom: 10px;">
                    <span class="badge-pill badge-info">{a['priority']} Priority</span>
                    <div style="font-weight: 700; color: #1E293B; margin-top: 4px;">{a['title']}</div>
                    <div style="font-size: 13px; color: #475569; margin-top: 2px;">{a['detail']}</div>
                </div>
                """, unsafe_allow_html=True)

    # Weekly Study Schedule
    st.markdown("---")
    st.markdown("### 🗓️ Recommended Weekly Timetable Allocation")
    sched_df = pd.DataFrame([
        {"Day": day, "Activity / Study Focus": desc}
        for day, desc in dossier['weekly_schedule'].items()
    ])
    st.dataframe(sched_df, use_container_width=True)

    # Downloadable Text Report
    report_text = f"""========================================================================
EDUPREDICT AI - STUDENT ACADEMIC INTERVENTION & RECOMMENDATION DOSSIER
========================================================================
Generated For: Active Student Assessment Profile
Intervention Urgency : {dossier['urgency']}
Predicted Final Grade: {dossier['predicted_g3']} / 20 (Portuguese 0-20 Scale)
Academic Risk Status : {'At Risk of Course Failure' if dossier['is_at_risk'] else 'Not At Risk (Passing)'}

1. IDENTIFIED STRENGTHS & PROTECTIVE FACTORS:
{chr(10).join(['  • ' + s for s in dossier['strengths']])}

2. CRITICAL WARNING FLAGS:
{chr(10).join(['  • ' + w for w in dossier['warnings']])}

3. ACADEMIC REMEDIATION PLAN:
{chr(10).join([f"  • [{a['priority']} Priority] {a['title']}: {a['detail']}" for a in dossier['academic_actions']])}

4. ATTENDANCE & LIFESTYLE INTERVENTIONS:
{chr(10).join([f"  • [{a['priority']} Priority] {a['title']}: {a['detail']}" for a in (dossier['attendance_actions'] + dossier['lifestyle_actions'])])}

5. RECOMMENDED WEEKLY TIMETABLE:
{chr(10).join([f"  • {day}: {desc}" for day, desc in dossier['weekly_schedule'].items()])}
========================================================================
System: Student Success Predictor (CRISP-DM AI Pipeline)
Dataset: UCI Student Performance (Cortez & Silva)
========================================================================
"""
    st.download_button(
        label="📥 Download Official Academic Intervention Dossier (.txt)",
        data=report_text,
        file_name="student_academic_intervention_report.txt",
        mime="text/plain",
        use_container_width=True
    )

    # Reset Journey Action
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 20px 0;">
        <h4 style="color: #1E293B;">🎉 Student Assessment Completed!</h4>
        <p style="color: #64748B; font-size: 14px;">Want to assess another student profile or test a different scenario?</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("🔄 Start New Student Assessment", type="secondary", use_container_width=True):
        set_page(1)
