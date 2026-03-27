"""
app.py  –  Entry point for the CSE Resume Intelligence Streamlit app.

Run with:
    streamlit run app.py

Prerequisites:
    1. Copy structured_resumes.csv  →  resume_app/data/structured_resumes.csv
    2. Run `python train_models.py` to generate the two .pkl files in models/
    3. pip install streamlit pandas numpy plotly pdfplumber scikit-learn joblib
"""

import os
import sys

import streamlit as st

# Make sure imports from sibling packages resolve correctly regardless of cwd
APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

# ---------------------------------------------------------------------------
# Page layout & global style
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="CSE Resume Intelligence",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Minimal custom CSS: clean, professional
st.markdown(
    """
    <style>
        /* Remove default top padding */
        .block-container { padding-top: 1.5rem; }

        /* Sidebar nav label */
        .css-1d391kg { background: #0e1117; }

        /* Metric cards */
        [data-testid="metric-container"] {
            background-color: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 8px;
            padding: 12px 16px;
        }

        /* Section dividers */
        hr { border-color: rgba(255, 255, 255, 0.08); }

        /* Hide Streamlit branding */
        #MainMenu, footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## CSE Resume Intelligence")
    st.caption("ML-Powered Resume Analysis")
    st.divider()

    page = st.radio(
        "Navigation",
        options=[
            "Home - Market Dashboard",
            "Resume Analyzer",
            "Model Comparison",
        ],
        index=0,
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("Dataset: 3,200 CSE resumes - 16 roles")
    st.caption("Algorithms: Random Forest, Gradient Boosting, SVM, Logistic Regression, KNN")

# ---------------------------------------------------------------------------
# Route to the selected page
# ---------------------------------------------------------------------------
if page == "Home - Market Dashboard":
    from views.home import render
    render()
elif page == "Resume Analyzer":
    from views.analyzer import render
    render()
else:
    from views.model_comparison import render
    render()
