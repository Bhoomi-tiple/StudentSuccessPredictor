"""
Student Success Predictor - Professional EdTech Web Application
================================================================
A modern, intuitive AI decision-support platform for student academic performance,
early failure risk detection, and personalized study recommendations.

Primary User Journey:
Home -> Enter Student Details -> Generate Student Report -> View Results -> Download Report (PDF/CSV/TXT)
"""

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Import rule-based expert recommendation engine
from expert_system import generate_recommendations

# Import report generation engine (PDF, CSV, TXT)
from report_generator import (
    generate_pdf_report,
    generate_csv_report,
    generate_txt_report
)

# ---------------------------------------------------------
# Page Configuration & Modern SaaS Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="EduPredict AI | Student Academic Success Platform",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Modern, Clean, Professional Interface
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1160px;
    }

    /* Hero Banner */
    .hero-banner {
        background: linear-gradient(135deg, #1E1B4B 0%, #312E81 55%, #4338CA 100%);
        color: white;
        padding: 36px 40px;
        border-radius: 16px;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(49, 46, 129, 0.25);
    }
    .hero-pill {
        display: inline-block;
        background: rgba(255, 255, 255, 0.16);
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
        line-height: 1.25;
    }
    .hero-subtitle {
        font-size: 15px;
        color: #C7D2FE;
        line-height: 1.6;
        margin: 0 0 20px 0;
        max-width: 760px;
    }

    /* SaaS Card Container */
    .saas-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 22px 24px;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.04);
        margin-bottom: 20px;
    }

    /* KPI Metric Card */
    .kpi-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 16px 18px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.03);
        border-top: 4px solid #4F46E5;
    }
    .kpi-title {
        color: #64748B;
        font-size: 11.5px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 4px;
    }
    .kpi-value {
        color: #0F172A;
        font-size: 26px;
        font-weight: 800;
        line-height: 1.2;
        margin-bottom: 2px;
    }
    .kpi-sub {
        color: #64748B;
        font-size: 12px;
    }

    /* Status Badges */
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

    /* Callout & Guide Box */
    .guide-box {
        background-color: #F8FAFC;
        border-left: 4px solid #4F46E5;
        border-top: 1px solid #E2E8F0;
        border-right: 1px solid #E2E8F0;
        border-bottom: 1px solid #E2E8F0;
        border-radius: 0 12px 12px 0;
        padding: 14px 18px;
        margin-bottom: 20px;
    }
    .guide-title {
        color: #1E293B;
        font-size: 13.5px;
        font-weight: 700;
        margin-bottom: 4px;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .guide-desc {
        color: #475569;
        font-size: 13px;
        margin: 0;
        line-height: 1.5;
    }

    /* Download Box */
    .download-card {
        background: linear-gradient(135deg, #F8FAFC 0%, #EEF2FF 100%);
        border: 1px solid #C7D2FE;
        border-radius: 14px;
        padding: 20px 24px;
        margin-top: 24px;
    }

    div.stButton > button:first-child {
        font-weight: 600;
        border-radius: 8px;
        padding: 0.55rem 1.25rem;
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
    """Loads all persisted models, preprocessors, and evaluation metrics."""
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
# Default Preset Data & Session State Management
# ---------------------------------------------------------
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
    "Average Student": {
        'school': 'GP', 'sex': 'M', 'age': 17, 'address': 'U', 'famsize': 'LE3',
        'Pstatus': 'T', 'Medu': 2, 'Fedu': 2, 'Mjob': 'services', 'Fjob': 'services',
        'reason': 'home', 'guardian': 'mother', 'traveltime': 2, 'studytime': 2,
        'failures': 1, 'schoolsup': 'no', 'famsup': 'yes', 'paid': 'no',
        'activities': 'yes', 'nursery': 'yes', 'higher': 'yes', 'internet': 'yes',
        'romantic': 'yes', 'famrel': 3, 'freetime': 4, 'goout': 5, 'Dalc': 2,
        'Walc': 4, 'health': 4, 'absences': 10, 'G1': 11, 'G2': 11
    }
}

if 'active_student' not in st.session_state:
    st.session_state['active_student'] = PRESETS["Average Student"].copy()

if 'active_page' not in st.session_state:
    st.session_state['active_page'] = "🏠 Home"

if 'report_generated' not in st.session_state:
    st.session_state['report_generated'] = False

def set_page(name):
    st.session_state['active_page'] = name
    st.rerun()

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
# Sidebar Navigation: Simple, Balanced, Uncluttered
# ---------------------------------------------------------
st.sidebar.markdown("""
<div style="padding: 10px 0 16px 0;">
    <div style="font-size: 20px; font-weight: 800; color: #1E1B4B; display: flex; align-items: center; gap: 8px;">
        <span>🎓</span> EduPredict AI
    </div>
    <div style="font-size: 12px; color: #64748B; margin-top: 2px;">Student Academic Success Platform</div>
</div>
""", unsafe_allow_html=True)

nav_options = [
    "🏠 Home",
    "📋 Student Assessment & Report",
    "🔍 Model Comparison & Benchmarks",
    "👥 Student Behavioral Groups"
]

selected_page = st.sidebar.radio(
    "Navigation Menu:",
    nav_options,
    index=nav_options.index(st.session_state['active_page']) if st.session_state['active_page'] in nav_options else 0,
    key="sidebar_radio"
)

if selected_page != st.session_state['active_page']:
    st.session_state['active_page'] = selected_page
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("##### ⚡ Quick-Fill Presets")
st.sidebar.caption("Load a profile instantly with 1-click:")

sp1, sp2 = st.sidebar.columns(2)
with sp1:
    if st.button("👩‍🎓 High Achiever", use_container_width=True):
        st.session_state['active_student'] = PRESETS["High Achiever"].copy()
        st.session_state['report_generated'] = False
        st.session_state['active_page'] = "📋 Student Assessment & Report"
        st.rerun()
with sp2:
    if st.button("⚠️ At-Risk", use_container_width=True):
        st.session_state['active_student'] = PRESETS["At-Risk Student"].copy()
        st.session_state['report_generated'] = False
        st.session_state['active_page'] = "📋 Student Assessment & Report"
        st.rerun()

if st.sidebar.button("⚖️ Average Student", use_container_width=True):
    st.session_state['active_student'] = PRESETS["Average Student"].copy()
    st.session_state['report_generated'] = False
    st.session_state['active_page'] = "📋 Student Assessment & Report"
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="font-size: 11px; color: #94A3B8; line-height: 1.5;">
    <b>Primary Flow:</b> Details ➔ Report ➔ Download<br/>
    <b>Benchmark:</b> UCI Student Performance<br/>
    <b>Stacking Accuracy:</b> 92.41% (79 test students)
</div>
""", unsafe_allow_html=True)


# =========================================================
# PAGE 1: HOME (CLEAR EXPLANATION & PROMINENT GET STARTED)
# =========================================================
if st.session_state['active_page'] == "🏠 Home":
    st.markdown("""
    <div class="hero-banner">
        <span class="hero-pill">✨ Educational Decision Support System</span>
        <h1 class="hero-title">Predict Student Success Early.<br/>Prevent Academic Failure.</h1>
        <p class="hero-subtitle">
            An easy-to-use AI platform that forecasts course grades, identifies students at risk of failure, 
            and generates comprehensive, downloadable pedagogical intervention reports in seconds.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Primary Call to Action
    st.markdown("""
    <div class="saas-card" style="border: 2px solid #C7D2FE; background: linear-gradient(to right, #FFFFFF, #F8FAFC);">
        <h3 style="margin: 0 0 6px 0; color: #1E1B4B; font-size: 20px;">🚀 Ready to Assess a Student?</h3>
        <p style="margin: 0 0 16px 0; color: #475569; font-size: 14px;">
            Enter student details (or pick a 1-click test preset) and generate a complete student success report with instant PDF, CSV, and TXT downloads.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🚀 Get Started — Generate Student Report", type="primary", use_container_width=True):
        set_page("📋 Student Assessment & Report")

    st.markdown("### 🧭 How It Works (Simple 3-Step Journey)")
    col_s1, col_s2, col_s3 = st.columns(3)

    with col_s1:
        st.markdown("""
        <div class="saas-card" style="height: 100%;">
            <div style="font-size: 28px; margin-bottom: 8px;">1️⃣</div>
            <div style="font-weight: 700; color: #1E293B; font-size: 16px; margin-bottom: 6px;">Enter Student Details</div>
            <p style="color: #64748B; font-size: 13px; line-height: 1.5; margin: 0;">
                Input past term marks (G1 & G2), weekly study hours, attendance records, and lifestyle factors.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col_s2:
        st.markdown("""
        <div class="saas-card" style="height: 100%;">
            <div style="font-size: 28px; margin-bottom: 8px;">2️⃣</div>
            <div style="font-weight: 700; color: #1E293B; font-size: 16px; margin-bottom: 6px;">Click Generate Report</div>
            <p style="color: #64748B; font-size: 13px; line-height: 1.5; margin: 0;">
                Our machine learning models forecast final score (0–20 scale) and assess academic failure risk.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col_s3:
        st.markdown("""
        <div class="saas-card" style="height: 100%;">
            <div style="font-size: 28px; margin-bottom: 8px;">3️⃣</div>
            <div style="font-weight: 700; color: #1E293B; font-size: 16px; margin-bottom: 6px;">Download Complete Report</div>
            <p style="color: #64748B; font-size: 13px; line-height: 1.5; margin: 0;">
                Receive personalized study advice and download the complete report as <b>PDF, CSV, or TXT</b>.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📊 Actual Model Evaluation Benchmarks (Test Set = 79 Students)")

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown("""
        <div class="kpi-card" style="border-top-color: #3B82F6;">
            <div class="kpi-title">Stacking Classifier Accuracy</div>
            <div class="kpi-value">92.41%</div>
            <div class="kpi-sub">Evaluated on 79 unseen students</div>
        </div>
        """, unsafe_allow_html=True)

    with k2:
        st.markdown("""
        <div class="kpi-card" style="border-top-color: #10B981;">
            <div class="kpi-title">Stacking Recall (Catch Rate)</div>
            <div class="kpi-value">92.31%</div>
            <div class="kpi-sub">24 of 26 at-risk students caught</div>
        </div>
        """, unsafe_allow_html=True)

    with k3:
        st.markdown("""
        <div class="kpi-card" style="border-top-color: #8B5CF6;">
            <div class="kpi-title">Stacking F1-Score</div>
            <div class="kpi-value">0.889</div>
            <div class="kpi-sub">Harmonic mean of precision & recall</div>
        </div>
        """, unsafe_allow_html=True)

    with k4:
        st.markdown("""
        <div class="kpi-card" style="border-top-color: #F59E0B;">
            <div class="kpi-title">SGD Regression R² & RMSE</div>
            <div class="kpi-value">0.805</div>
            <div class="kpi-sub">RMSE = 2.034 marks (0-20 scale)</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="guide-box" style="margin-top: 16px;">
        <div class="guide-title">
            <span>ℹ️ Scientific Transparency & Limitations</span>
        </div>
        <p class="guide-desc">
            Classification accuracy (92.41%) and recall (92.31%) were achieved by the Stacking Ensemble on an independent, 
            unseen test set of 79 students. These benchmarks demonstrate strong statistical reliability on benchmark data, 
            but do not guarantee 100% future certainty. Continuous grade prediction achieved an R² of 0.805 with an average 
            error margin (RMSE) of 2.034 marks. This tool is built to assist educators with early warnings, not to replace human evaluation.
        </p>
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# PAGE 2: STUDENT ASSESSMENT & REPORT (PRIMARY WORKFLOW)
# =========================================================
elif st.session_state['active_page'] == "📋 Student Assessment & Report":
    st.markdown("## 📋 Student Assessment & Report Generator")
    st.markdown("Enter student information below or select a quick-preset, then click **Generate Student Report**.")

    # Quick Presets Buttons right on page
    st.markdown("##### ⚡ Quick-Fill Presets:")
    q1, q2, q3 = st.columns(3)
    with q1:
        if st.button("👩‍🎓 Load High Achiever (16/20)", key="btn_p1", use_container_width=True):
            st.session_state['active_student'] = PRESETS["High Achiever"].copy()
            st.session_state['report_generated'] = False
            st.rerun()
    with q2:
        if st.button("⚠️ Load At-Risk Student (6/20)", key="btn_p2", use_container_width=True):
            st.session_state['active_student'] = PRESETS["At-Risk Student"].copy()
            st.session_state['report_generated'] = False
            st.rerun()
    with q3:
        if st.button("⚖️ Load Average Student (11/20)", key="btn_p3", use_container_width=True):
            st.session_state['active_student'] = PRESETS["Average Student"].copy()
            st.session_state['report_generated'] = False
            st.rerun()

    cur_s = st.session_state['active_student']

    # Organized Form Tabs
    st.markdown("### 📝 Student Details Form")
    tab1, tab2, tab3 = st.tabs([
        "🎓 1. Academic Performance & Attendance",
        "👤 2. Student Background & Goals",
        "🏡 3. Family Environment & Habits"
    ])

    with tab1:
        st.markdown("##### Key Academic Indicators (Strongest Predictors)")
        c1, c2 = st.columns(2)
        with c1:
            g1 = st.slider("Term 1 Exam Grade (G1) [0 to 20]", 0, 20, int(cur_s.get('G1', 11)),
                           help="Portuguese standard scale: 0-9 Fail, 10-20 Pass")
            g2 = st.slider("Term 2 Exam Grade (G2) [0 to 20]", 0, 20, int(cur_s.get('G2', 11)),
                           help="Second term score is highly predictive of final outcome")
            studytime = st.selectbox(
                "Weekly Study Time",
                options=[1, 2, 3, 4],
                index=[1, 2, 3, 4].index(cur_s.get('studytime', 2)),
                format_func=lambda x: {1: "< 2 hours/week", 2: "2 to 5 hours/week", 3: "5 to 10 hours/week", 4: "> 10 hours/week"}[x]
            )
        with c2:
            failures = st.slider("Past Course Failures", 0, 4, int(cur_s.get('failures', 0)),
                                 help="Number of times student previously failed this subject")
            absences = st.slider("School Absences (Days Missed)", 0, 93, int(cur_s.get('absences', 4)),
                                 help="Total absent days in the academic year")
            schoolsup = st.selectbox("School Support / Remedial Classes", ["yes", "no"],
                                     index=0 if cur_s.get('schoolsup', 'no') == 'yes' else 1,
                                     format_func=lambda x: "Yes (Receives School Tutoring)" if x == "yes" else "No")
            paid = st.selectbox("Extra Paid Mathematics Tutoring", ["yes", "no"],
                                index=0 if cur_s.get('paid', 'no') == 'yes' else 1,
                                format_func=lambda x: "Yes (Private Tutoring)" if x == "yes" else "No")

    with tab2:
        st.markdown("##### Demographics & Aspirations")
        c3, c4 = st.columns(2)
        with c3:
            school = st.selectbox("School Attended", ["GP", "MS"],
                                  index=0 if cur_s.get('school', 'GP') == 'GP' else 1,
                                  format_func=lambda x: "Gabriel Pereira (GP)" if x == "GP" else "Mousinho da Silveira (MS)")
            sex = st.selectbox("Gender", ["F", "M"],
                               index=0 if cur_s.get('sex', 'F') == 'F' else 1,
                               format_func=lambda x: "Female" if x == "F" else "Male")
            age = st.slider("Age (Years)", 15, 22, int(cur_s.get('age', 17)))
            address = st.selectbox("Home Residential Area", ["U", "R"],
                                   index=0 if cur_s.get('address', 'U') == 'U' else 1,
                                   format_func=lambda x: "Urban / City (U)" if x == "U" else "Rural (R)")
        with c4:
            famsize = st.selectbox("Family Size", ["GT3", "LE3"],
                                   index=0 if cur_s.get('famsize', 'GT3') == 'GT3' else 1,
                                   format_func=lambda x: "More than 3 members (> 3)" if x == "GT3" else "3 or fewer members (≤ 3)")
            pstatus = st.selectbox("Parent Living Arrangement", ["T", "A"],
                                   index=0 if cur_s.get('Pstatus', 'T') == 'T' else 1,
                                   format_func=lambda x: "Living Together (T)" if x == "T" else "Apart / Separated (A)")
            higher = st.selectbox("Plans to Attend Higher Education / University", ["yes", "no"],
                                  index=0 if cur_s.get('higher', 'yes') == 'yes' else 1)
            internet = st.selectbox("Home Internet Access", ["yes", "no"],
                                    index=0 if cur_s.get('internet', 'yes') == 'yes' else 1)

    with tab3:
        st.markdown("##### Family Support & Lifestyle")
        c5, c6 = st.columns(2)
        with c5:
            medu = st.selectbox("Mother's Education Level", [0, 1, 2, 3, 4],
                                index=int(cur_s.get('Medu', 2)),
                                format_func=lambda x: {0: "None", 1: "Primary (4th grade)", 2: "5th-9th grade", 3: "Secondary Education", 4: "Higher Education / University"}[x])
            fedu = st.selectbox("Father's Education Level", [0, 1, 2, 3, 4],
                                index=int(cur_s.get('Fedu', 2)),
                                format_func=lambda x: {0: "None", 1: "Primary (4th grade)", 2: "5th-9th grade", 3: "Secondary Education", 4: "Higher Education / University"}[x])
            mjob = st.selectbox("Mother's Job", ["teacher", "health", "services", "at_home", "other"],
                                index=["teacher", "health", "services", "at_home", "other"].index(cur_s.get('Mjob', 'other')))
            fjob = st.selectbox("Father's Job", ["teacher", "health", "services", "at_home", "other"],
                                index=["teacher", "health", "services", "at_home", "other"].index(cur_s.get('Fjob', 'other')))
            reason = st.selectbox("Reason for Choosing School", ["home", "reputation", "course", "other"],
                                  index=["home", "reputation", "course", "other"].index(cur_s.get('reason', 'course')))
            guardian = st.selectbox("Primary Guardian", ["mother", "father", "other"],
                                    index=["mother", "father", "other"].index(cur_s.get('guardian', 'mother')))
            famsup = st.selectbox("Family Educational Support", ["yes", "no"],
                                  index=0 if cur_s.get('famsup', 'yes') == 'yes' else 1)
        with c6:
            activities = st.selectbox("Extracurricular Activities", ["yes", "no"],
                                      index=0 if cur_s.get('activities', 'no') == 'yes' else 1)
            nursery = st.selectbox("Attended Nursery / Kindergarten", ["yes", "no"],
                                   index=0 if cur_s.get('nursery', 'yes') == 'yes' else 1)
            romantic = st.selectbox("In a Romantic Relationship", ["yes", "no"],
                                    index=0 if cur_s.get('romantic', 'no') == 'yes' else 1)
            traveltime = st.selectbox("Commute Time to School", [1, 2, 3, 4],
                                      index=int(cur_s.get('traveltime', 1)) - 1,
                                      format_func=lambda x: {1: "< 15 minutes", 2: "15 to 30 minutes", 3: "30 to 60 minutes", 4: "> 60 minutes"}[x])
            famrel = st.slider("Quality of Family Relationships [1-5]", 1, 5, int(cur_s.get('famrel', 4)))
            freetime = st.slider("Free Time After School [1-5]", 1, 5, int(cur_s.get('freetime', 3)))
            goout = st.slider("Going Out with Friends [1-5]", 1, 5, int(cur_s.get('goout', 3)))
            dalc = st.slider("Weekday Alcohol Intake [1-5]", 1, 5, int(cur_s.get('Dalc', 1)))
            walc = st.slider("Weekend Alcohol Intake [1-5]", 1, 5, int(cur_s.get('Walc', 1)))
            health = st.slider("Current Health Status [1-5]", 1, 5, int(cur_s.get('health', 4)))

    # Update active student in session state
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
    generate_clicked = st.button("📊 Generate Complete Student Report", type="primary", use_container_width=True)

    if generate_clicked or st.session_state.get('report_generated', False):
        st.session_state['report_generated'] = True

        # Run Prediction Pipeline
        active_student = st.session_state['active_student']
        input_df = pd.DataFrame([active_student])
        X_proc = preprocessor.transform(input_df)

        pred_lr = float(reg_models['Linear Regression'].predict(X_proc)[0])
        pred_sgd = float(reg_models['Gradient Descent (SGD)'].predict(X_proc)[0])

        pred_lr = max(0.0, min(20.0, pred_lr))
        pred_sgd = max(0.0, min(20.0, pred_sgd))
        predicted_g3 = (pred_lr + pred_sgd) / 2.0

        # Classification predictions across all 7 models
        clf_preds = {}
        clf_probs = {}
        for name, model in clf_models.items():
            p = model.predict(X_proc)[0]
            prob = model.predict_proba(X_proc)[0][1] if hasattr(model, 'predict_proba') else (1.0 if p == 1 else 0.0)
            clf_preds[name] = int(p)
            clf_probs[name] = float(prob)

        stacking_pred = clf_preds['Stacking Classifier']
        stacking_prob = clf_probs['Stacking Classifier']
        avg_risk_prob = np.mean(list(clf_probs.values()))
        is_at_risk = (stacking_pred == 1)

        l_code, l_desc, l_badge = get_letter_grade(predicted_g3)

        prediction_results = {
            'predicted_g3': predicted_g3,
            'letter_grade': l_code,
            'letter_desc': l_desc,
            'is_at_risk': is_at_risk,
            'risk_probability': stacking_prob,
            'avg_risk_prob': avg_risk_prob,
            'pred_lr': pred_lr,
            'pred_sgd': pred_sgd,
            'clf_preds': clf_preds,
            'clf_probs': clf_probs
        }

        # Generate rule-based recommendations
        recommendations = generate_recommendations(
            student_data=active_student,
            predicted_g3=predicted_g3,
            is_at_risk=is_at_risk
        )

        # ---------------------------------------------------------
        # DISPLAY RESULTS
        # ---------------------------------------------------------
        st.markdown("## 🎯 Generated Student Assessment Results")

        res_col1, res_col2 = st.columns(2)

        with res_col1:
            st.markdown(f"""
            <div class="saas-card" style="border-left: 6px solid #4F46E5;">
                <div class="kpi-title">Forecasted Final Examination Grade (G3)</div>
                <div class="kpi-value">{predicted_g3:.2f} <span style="font-size: 16px; color: #64748B;">/ 20.0</span></div>
                <div class="kpi-sub" style="margin-top: 6px;">
                    <span class="badge-pill {l_badge}">{l_code} ({l_desc})</span> • 
                    Linear: {pred_lr:.1f} • SGD: {pred_sgd:.1f}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with res_col2:
            risk_border = "#EF4444" if is_at_risk else "#10B981"
            risk_badge = "badge-risk" if is_at_risk else "badge-pass"
            risk_title = "AT RISK OF FAILURE (< 10 Marks)" if is_at_risk else "NOT AT RISK (Passing Track)"
            st.markdown(f"""
            <div class="saas-card" style="border-left: 6px solid {risk_border};">
                <div class="kpi-title">Academic Risk Classification (Stacking Ensemble)</div>
                <div class="kpi-value" style="font-size: 22px;">{risk_title}</div>
                <div class="kpi-sub" style="margin-top: 6px;">
                    <span class="badge-pill {risk_badge}">Failure Probability: {stacking_prob * 100:.1f}%</span> • 
                    Intervention: {recommendations.get('urgency', 'MODERATE')}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Visual Score Trajectory
        st.markdown("##### 📈 Grade Progression & Passing Line")
        fig, ax = plt.subplots(figsize=(8, 2.2))
        cats = ['Term 1 (G1)', 'Term 2 (G2)', 'AI Forecasted (G3)']
        scores = [active_student['G1'], active_student['G2'], predicted_g3]
        bar_colors = ['#94A3B8', '#64748B', '#10B981' if predicted_g3 >= 10 else '#EF4444']

        bars = ax.barh(cats, scores, color=bar_colors, height=0.45)
        ax.set_xlim(0, 20)
        ax.axvline(10, color='#EF4444', linestyle='--', linewidth=1.5, label='Passing Threshold (10 Marks)')

        for b in bars:
            w = b.get_width()
            ax.text(w + 0.3, b.get_y() + b.get_height()/2, f'{w:.1f} / 20', va='center', fontweight='bold', fontsize=9.5)

        ax.set_xlabel("Marks (Scale 0 to 20)", fontsize=9.5)
        ax.legend(loc='lower right', frameon=True)
        sns.despine()
        st.pyplot(fig)
        plt.close(fig)

        # Explanation Box
        if is_at_risk:
            explanation = (
                f"The student is classified <b>At Risk of Academic Failure</b> with a predicted score of <b>{predicted_g3:.1f}/20</b>. "
                f"Primary contributing factors include past academic grades (Term 1: {active_student['G1']}, Term 2: {active_student['G2']}), "
                f"accumulated absences ({active_student['absences']} days), and historical course failures ({active_student['failures']})."
            )
        else:
            explanation = (
                f"The student is on a <b>Passing Academic Track</b> with a predicted score of <b>{predicted_g3:.1f}/20</b> ({l_desc}). "
                f"Consistent historical performance (Term 1: {active_student['G1']}, Term 2: {active_student['G2']}) and manageable absences "
                f"({active_student['absences']} days) support continued academic success."
            )

        st.markdown(f"""
        <div class="saas-card" style="background: #F8FAFC;">
            <div style="font-weight: 700; color: #1E293B; margin-bottom: 4px;">💡 Explanation of Prediction:</div>
            <div style="font-size: 13.5px; color: #334155; line-height: 1.6;">{explanation}</div>
        </div>
        """, unsafe_allow_html=True)

        # Strengths & Warnings
        st.markdown("### 🔍 Key Diagnostic Insights")
        sw1, sw2 = st.columns(2)
        with sw1:
            st.markdown("##### ✅ Protective Factors & Strengths")
            for s in recommendations.get('strengths', []):
                st.success(f"• {s}")
        with sw2:
            st.markdown("##### ⚠️ Risk Factors & Critical Warnings")
            for w in recommendations.get('warnings', []):
                st.error(f"• {w}")

        # Action Items
        st.markdown("### 🎯 Personalized Pedagogical Recommendations")
        act_col1, act_col2 = st.columns(2)
        with act_col1:
            st.markdown("##### 📚 Academic Remediation Plan")
            for a in recommendations.get('academic_actions', []):
                st.markdown(f"""
                <div class="saas-card" style="padding: 12px 14px; margin-bottom: 8px;">
                    <span class="badge-pill {'badge-risk' if a['priority']=='Critical' else 'badge-warn'}">{a['priority']} Priority</span>
                    <div style="font-weight: 700; color: #1E293B; margin-top: 3px;">{a['title']}</div>
                    <div style="font-size: 12.5px; color: #475569; margin-top: 2px;">{a['detail']}</div>
                </div>
                """, unsafe_allow_html=True)

            if recommendations.get('attendance_actions'):
                st.markdown("##### ⏰ Attendance & Punctuality Plan")
                for a in recommendations.get('attendance_actions', []):
                    st.markdown(f"""
                    <div class="saas-card" style="padding: 12px 14px; margin-bottom: 8px;">
                        <span class="badge-pill badge-risk">{a['priority']} Priority</span>
                        <div style="font-weight: 700; color: #1E293B; margin-top: 3px;">{a['title']}</div>
                        <div style="font-size: 12.5px; color: #475569; margin-top: 2px;">{a['detail']}</div>
                    </div>
                    """, unsafe_allow_html=True)

        with act_col2:
            st.markdown("##### 🧘 Lifestyle, Wellness & Support")
            if recommendations.get('lifestyle_actions'):
                for a in recommendations.get('lifestyle_actions', []):
                    st.markdown(f"""
                    <div class="saas-card" style="padding: 12px 14px; margin-bottom: 8px;">
                        <span class="badge-pill badge-warn">{a['priority']} Priority</span>
                        <div style="font-weight: 700; color: #1E293B; margin-top: 3px;">{a['title']}</div>
                        <div style="font-size: 12.5px; color: #475569; margin-top: 2px;">{a['detail']}</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("Current lifestyle and social habits are balanced.")

            if recommendations.get('support_actions'):
                st.markdown("##### 🤝 Institutional & Community Support")
                for a in recommendations.get('support_actions', []):
                    st.markdown(f"""
                    <div class="saas-card" style="padding: 12px 14px; margin-bottom: 8px;">
                        <span class="badge-pill badge-info">{a['priority']} Priority</span>
                        <div style="font-weight: 700; color: #1E293B; margin-top: 3px;">{a['title']}</div>
                        <div style="font-size: 12.5px; color: #475569; margin-top: 2px;">{a['detail']}</div>
                    </div>
                    """, unsafe_allow_html=True)

        # Weekly Timetable
        st.markdown("##### 🗓️ Recommended Weekly Timetable Allocation")
        sched_df = pd.DataFrame([
            {"Day": day, "Prescribed Focus": desc}
            for day, desc in recommendations.get('weekly_schedule', {}).items()
        ])
        st.dataframe(sched_df, use_container_width=True)

        # ---------------------------------------------------------
        # THREE WORKING DOWNLOAD OPTIONS (PDF, CSV, TXT)
        # ---------------------------------------------------------
        st.markdown("---")
        st.markdown("### 📥 Download Student Report (3 Export Formats)")
        st.caption("All downloads contain the real student's details, exact predictions, recommendations, and model information:")

        # Generate the three export formats in memory
        pdf_bytes = generate_pdf_report(active_student, prediction_results, recommendations)
        csv_string = generate_csv_report(active_student, prediction_results, recommendations)
        txt_string = generate_txt_report(active_student, prediction_results, recommendations)

        d1, d2, d3 = st.columns(3)

        with d1:
            st.download_button(
                label="📄 Download Official PDF Report",
                data=pdf_bytes,
                file_name=f"student_academic_report_{active_student.get('sex', 'S')}_{int(predicted_g3)}.pdf",
                mime="application/pdf",
                use_container_width=True
            )

        with d2:
            st.download_button(
                label="📊 Download Report Data (CSV)",
                data=csv_string,
                file_name=f"student_assessment_data_{int(predicted_g3)}.csv",
                mime="text/csv",
                use_container_width=True
            )

        with d3:
            st.download_button(
                label="📝 Download Text Dossier (TXT)",
                data=txt_string,
                file_name=f"student_intervention_dossier_{int(predicted_g3)}.txt",
                mime="text/plain",
                use_container_width=True
            )

        # Model Transparency & Limitations Note
        st.markdown("""
        <div class="guide-box" style="margin-top: 24px;">
            <div class="guide-title">
                <span>🛡️ Model Performance & Accuracy Disclosure</span>
            </div>
            <p class="guide-desc">
                • <b>Stacking Classifier:</b> 92.41% Accuracy, 92.31% Recall, 0.889 F1-score (evaluated on 79 unseen test students).<br/>
                • <b>SGD Regression:</b> R² = 0.805, RMSE = 2.034 marks on a 0-20 scale.<br/>
                • <b>Notice:</b> Test set accuracy does not guarantee identical future performance on differing curricula. 
                Regression R² reflects continuous grade variance explained, not classification accuracy. 
                This assessment is designed to support educators and counselors with early warnings.
            </p>
        </div>
        """, unsafe_allow_html=True)


# =========================================================
# PAGE 3: MODEL COMPARISON & BENCHMARKS (SECONDARY SECTION)
# =========================================================
elif st.session_state['active_page'] == "🔍 Model Comparison & Benchmarks":
    st.markdown("## 🔍 Machine Learning Model Benchmarks & Comparison")
    st.markdown("Scientific evaluation results measured on the **unseen test partition of 79 students** with zero data leakage.")

    tab_m1, tab_m2, tab_m3 = st.tabs([
        "🏆 Classification & Ensemble Models",
        "📈 Grade Prediction Regression Models",
        "🔲 Confusion Matrix Visualizer"
    ])

    with tab_m1:
        st.markdown("##### 7 Classification Models Leaderboard")
        clf_metrics = eval_results['classification']
        rows = []
        for m, res in clf_metrics.items():
            rows.append({
                "Algorithm": m,
                "Type": res.get('Category', 'Individual'),
                "Accuracy (%)": f"{res['Accuracy']*100:.2f}%",
                "Recall / Catch Rate (%)": f"{res['Recall']*100:.2f}%",
                "Precision (%)": f"{res['Precision']*100:.2f}%",
                "F1-Score": f"{res['F1-Score']:.3f}"
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        fig, ax = plt.subplots(figsize=(9, 3.8))
        models = list(clf_metrics.keys())
        accs = [clf_metrics[m]['Accuracy'] * 100 for m in models]
        f1s = [clf_metrics[m]['F1-Score'] * 100 for m in models]
        x = np.arange(len(models))
        w = 0.35

        ax.bar(x - w/2, accs, w, label='Accuracy (%)', color='#3B82F6')
        ax.bar(x + w/2, f1s, w, label='F1-Score (%)', color='#F59E0B')
        ax.set_ylabel('Score (%)', fontsize=9.5)
        ax.set_title('Accuracy vs F1-Score Across Classifiers & Ensembles', fontweight='bold', pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=20, ha='right', fontsize=9)
        ax.set_ylim(50, 100)
        ax.legend(frameon=True)
        sns.despine()
        st.pyplot(fig)
        plt.close(fig)

    with tab_m2:
        st.markdown("##### Continuous Grade Prediction (Regression) Results")
        reg_metrics = eval_results['regression']
        reg_df = pd.DataFrame([
            {
                "Model": m,
                "MAE (Average Error)": f"{res['MAE']:.3f} marks",
                "RMSE (Root Mean Squared Error)": f"{res['RMSE']:.3f} marks",
                "R² Score (Variance Explained)": f"{res['R2']:.3f}"
            }
            for m, res in reg_metrics.items()
        ])
        st.dataframe(reg_df, use_container_width=True)

        c_r1, c_r2 = st.columns(2)
        with c_r1:
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

        with c_r2:
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

    with tab_m3:
        st.markdown("##### Confusion Matrix Inspection")
        selected_m = st.selectbox("Select Model to Inspect:", list(clf_models.keys()), index=6)
        cm = np.array(clf_metrics[selected_m]['Confusion_Matrix'])
        tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]

        c_cm1, c_cm2 = st.columns([1.2, 1])
        with c_cm1:
            fig, ax = plt.subplots(figsize=(5, 3.8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                        xticklabels=['Passed (0)', 'At Risk (1)'],
                        yticklabels=['Passed (0)', 'At Risk (1)'], ax=ax,
                        annot_kws={'fontsize': 14, 'weight': 'bold'})
            ax.set_xlabel('Predicted Label', fontweight='bold')
            ax.set_ylabel('Actual Ground Truth', fontweight='bold')
            ax.set_title(f'Confusion Matrix: {selected_m}', fontweight='bold')
            st.pyplot(fig)
            plt.close(fig)

        with c_cm2:
            st.markdown(f"""
            <div class="saas-card">
                <div style="font-weight: 700; color: #1E293B; margin-bottom: 8px;">Breakdown for {selected_m}:</div>
                <div style="font-size: 13px; line-height: 1.8; color: #475569;">
                    • <b>{tp} At-Risk students caught</b> (True Positives)<br/>
                    • <b>{tn} Passing students confirmed</b> (True Negatives)<br/>
                    • <b>{fp} False alarms</b> (Passing student flagged)<br/>
                    • <b>{fn} Missed at-risk cases</b> (False Negatives)<br/>
                    • <b>Recall / Safety Rate:</b> <code>{tp/(tp+fn)*100:.1f}%</code>
                </div>
            </div>
            """, unsafe_allow_html=True)


# =========================================================
# PAGE 4: STUDENT CLUSTERING (SECONDARY SECTION)
# =========================================================
elif st.session_state['active_page'] == "👥 Student Behavioral Groups":
    st.markdown("## 👥 Student Behavioral Archetypes (K-Means Clustering)")
    st.markdown("Unsupervised segmentation discovering 3 natural student learning personas based on study habits, social factors, and attendance.")

    c_meta = eval_results['clustering']
    personas = c_meta['personas']
    profiles = c_meta['profiles']

    col_k1, col_k2, col_k3 = st.columns(3)
    with col_k1:
        st.markdown(f"""
        <div class="kpi-card" style="border-top-color: #3B82F6;">
            <div class="kpi-title">Number of Clusters (K)</div>
            <div class="kpi-value">{c_meta['optimal_k']}</div>
            <div class="kpi-sub">Validated by Silhouette analysis</div>
        </div>
        """, unsafe_allow_html=True)
    with col_k2:
        st.markdown(f"""
        <div class="kpi-card" style="border-top-color: #10B981;">
            <div class="kpi-title">Silhouette Score</div>
            <div class="kpi-value">{c_meta['silhouette']:.3f}</div>
            <div class="kpi-sub">Cluster separation metric</div>
        </div>
        """, unsafe_allow_html=True)
    with col_k3:
        st.markdown(f"""
        <div class="kpi-card" style="border-top-color: #F59E0B;">
            <div class="kpi-title">Davies-Bouldin Index</div>
            <div class="kpi-value">{c_meta['davies_bouldin']:.3f}</div>
            <div class="kpi-sub">Lower indicates tighter clusters</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 👥 The 3 Discovered Student Personas")
    p1, p2, p3 = st.columns(3)

    with p1:
        st.markdown(f"""
        <div class="saas-card" style="border-top: 4px solid #EF4444; height: 100%;">
            <span class="badge-pill badge-risk">Cluster 0</span>
            <div style="font-weight: 800; font-size: 16px; color: #991B1B; margin: 8px 0 4px 0;">{personas[0]['name']}</div>
            <p style="font-size: 12.5px; color: #475569; margin-bottom: 12px;">{personas[0]['desc']}</p>
            <div style="font-size: 12px; line-height: 1.6; background: #FEF2F2; padding: 10px; border-radius: 8px;">
                <b>Avg Final Grade:</b> {profiles[0]['G3']:.1f} / 20<br/>
                <b>Avg Failures:</b> {profiles[0]['failures']:.2f}<br/>
                <b>Avg Absences:</b> {profiles[0]['absences']:.1f} days
            </div>
        </div>
        """, unsafe_allow_html=True)

    with p2:
        st.markdown(f"""
        <div class="saas-card" style="border-top: 4px solid #F59E0B; height: 100%;">
            <span class="badge-pill badge-warn">Cluster 1</span>
            <div style="font-weight: 800; font-size: 16px; color: #B45309; margin: 8px 0 4px 0;">{personas[1]['name']}</div>
            <p style="font-size: 12.5px; color: #475569; margin-bottom: 12px;">{personas[1]['desc']}</p>
            <div style="font-size: 12px; line-height: 1.6; background: #FFFBEB; padding: 10px; border-radius: 8px;">
                <b>Avg Final Grade:</b> {profiles[1]['G3']:.1f} / 20<br/>
                <b>Avg Absences:</b> {profiles[1]['absences']:.1f} days (Highest!)<br/>
                <b>Weekend Alcohol:</b> {profiles[1]['Walc']:.1f} / 5
            </div>
        </div>
        """, unsafe_allow_html=True)

    with p3:
        st.markdown(f"""
        <div class="saas-card" style="border-top: 4px solid #10B981; height: 100%;">
            <span class="badge-pill badge-pass">Cluster 2</span>
            <div style="font-weight: 800; font-size: 16px; color: #065F46; margin: 8px 0 4px 0;">{personas[2]['name']}</div>
            <p style="font-size: 12.5px; color: #475569; margin-bottom: 12px;">{personas[2]['desc']}</p>
            <div style="font-size: 12px; line-height: 1.6; background: #ECFDF5; padding: 10px; border-radius: 8px;">
                <b>Avg Final Grade:</b> {profiles[2]['G3']:.1f} / 20 (Distinction)<br/>
                <b>Avg Failures:</b> {profiles[2]['failures']:.2f} (Near Zero)<br/>
                <b>Study Discipline:</b> High & Consistent
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🗺️ 2D PCA Map of Student Behavioral Segments")
    if df_clustered is not None:
        fig, ax = plt.subplots(figsize=(9, 4.2))
        palette = {0: '#EF4444', 1: '#F59E0B', 2: '#10B981'}
        sns.scatterplot(
            data=df_clustered,
            x='pca_1', y='pca_2',
            hue='cluster',
            palette=palette,
            s=50, alpha=0.85, ax=ax
        )
        ax.set_title("Student Behavioral Clusters (PCA Space)", fontweight='bold', pad=10)
        ax.set_xlabel("Principal Component 1 (Attainment & Consistency)", fontsize=9.5)
        ax.set_ylabel("Principal Component 2 (Social, Absences & Alcohol)", fontsize=9.5)
        ax.legend(title="Persona", labels=[personas[0]['name'], personas[1]['name'], personas[2]['name']], frameon=True)
        sns.despine()
        st.pyplot(fig)
        plt.close(fig)
