"""
pages/home.py
Market Intelligence Dashboard – CSE Focused
"""

import os
import re

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter

# Skill category taxonomy (mirrors utils/features.py)
SKILL_CATEGORIES = {
    "Programming":     ["python", "java", "c++", "javascript", "c#", "php",
                        "swift", "kotlin", "scala", "r", "matlab"],
    "Data Science":    ["machine learning", "deep learning", "nlp", "tableau",
                        "tensorflow", "keras", "pytorch", "pandas", "numpy",
                        "scikit-learn", "spark", "hadoop"],
    "Web Development": ["html", "css", "javascript", "react", "angular",
                        "node.js", "django", "flask", "php"],
    "Core CS":         ["c++", "java", "sql", "linux", "git", "mysql",
                        "postgresql", "mongodb", "c#"],
    "Cloud / DevOps":  ["aws", "docker", "kubernetes"],
}

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "structured_resumes.csv")


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading dataset…")
def load_data() -> pd.DataFrame:
    path = os.path.abspath(DATA_PATH)
    if not os.path.exists(path):
        st.error(
            f"Dataset not found at `{path}`.\n"
            "Copy `structured_resumes.csv` into `resume_app/data/`."
        )
        st.stop()
    df = pd.read_csv(path)
    # Derived columns
    df["num_skills"] = df["skills"].apply(
        lambda x: len([s.strip() for s in str(x).split(",") if s.strip()])
        if pd.notna(x) and str(x).strip() else 0
    )
    df["has_internship"] = df["internships"].notna().map(
        {True: "With Internship", False: "No Internship"}
    )
    return df


def _parse_skills_flat(df: pd.DataFrame) -> list[str]:
    raw = df["skills"].dropna().str.split(",").tolist()
    return [s.strip().lower() for sublist in raw for s in sublist if s.strip()]


def _categorise_skill(skill: str) -> str | None:
    for cat, members in SKILL_CATEGORIES.items():
        if skill in [m.lower() for m in members]:
            return cat
    return None


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------

def _chart_top_skills(all_skills: list[str]):
    """Horizontal bar chart – Top 15 skills."""
    counts = Counter(all_skills)
    top15 = pd.DataFrame(counts.most_common(15), columns=["Skill", "Count"])
    top15 = top15.sort_values("Count", ascending=True)          # ascending so largest bar at top

    fig = px.bar(
        top15,
        x="Count",
        y="Skill",
        orientation="h",
        color="Count",
        color_continuous_scale="teal",
        title="Top 15 Most Common Skills among CSE Students",
        labels={"Count": "Frequency", "Skill": ""},
        height=480,
    )
    fig.update_layout(
        coloraxis_showscale=False,
        title_font_size=16,
        margin=dict(l=20, r=20, t=50, b=20),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(tickfont=dict(size=13)),
    )
    fig.update_traces(marker_line_width=0)
    return fig


def _chart_internship_skill(df: pd.DataFrame):
    """Box plot – Internship vs Skill Count Distribution."""
    fig = px.box(
        df,
        x="has_internship",
        y="num_skills",
        color="has_internship",
        points="outliers",
        color_discrete_map={
            "With Internship": "#2ecc71",
            "No Internship":   "#e74c3c",
        },
        title="Internship vs Skill Count Distribution",
        labels={"has_internship": "", "num_skills": "Number of Skills"},
        height=420,
    )
    fig.update_layout(
        showlegend=False,
        title_font_size=16,
        margin=dict(l=20, r=20, t=50, b=20),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _chart_skill_categories(all_skills: list[str]):
    """Horizontal bar chart – Skill category % distribution."""
    cat_counts: dict[str, int] = {c: 0 for c in SKILL_CATEGORIES}
    for skill in all_skills:
        cat = _categorise_skill(skill)
        if cat:
            cat_counts[cat] += 1
    total = sum(cat_counts.values()) or 1
    cat_df = pd.DataFrame(
        [{"Category": c, "Percentage": (v / total) * 100} for c, v in cat_counts.items()]
    ).sort_values("Percentage", ascending=False)

    fig = px.bar(
        cat_df,
        x="Category",
        y="Percentage",
        color="Category",
        color_discrete_sequence=px.colors.qualitative.Set2,
        title="Skill Category Distribution (% of All Skill Occurrences)",
        labels={"Percentage": "% of Total", "Category": ""},
        height=400,
        text=cat_df["Percentage"].map("{:.1f}%".format),
    )
    fig.update_traces(textposition="outside", marker_line_width=0)
    fig.update_layout(
        showlegend=False,
        title_font_size=16,
        margin=dict(l=20, r=20, t=60, b=20),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _chart_domain_distribution(df: pd.DataFrame):
    """Treemap of resume counts per domain."""
    domain_counts = df["domain"].value_counts().reset_index()
    domain_counts.columns = ["Domain", "Count"]
    fig = px.treemap(
        domain_counts,
        path=["Domain"],
        values="Count",
        color="Count",
        color_continuous_scale="Blues",
        title="Resume Distribution Across Domains",
        height=420,
    )
    fig.update_layout(
        title_font_size=16,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------

def render():
    # Header
    st.markdown(
        "<h1 style='text-align:center; letter-spacing:1px;'>"
        "CSE Resume Market Intelligence Dashboard"
        "</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align:center; color:#888; margin-top:-10px;'>"
        "Insights from 3,200 CSE student resumes"
        "</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    df         = load_data()
    all_skills = _parse_skills_flat(df)

    # ── Summary KPIs ──────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Resumes",    f"{len(df):,}")
    k2.metric("Unique Domains",   f"{df['domain'].nunique()}")
    k3.metric("Avg Skills / Resume", f"{df['num_skills'].mean():.1f}")
    k4.metric("With Internship",  f"{df['internships'].notna().sum():,}")

    st.divider()

    # ── A: Top 15 Skills ─────────────────────────────────────────────────
    st.subheader("A · Top 15 Most Common Skills")
    st.plotly_chart(_chart_top_skills(all_skills), width='stretch')

    st.divider()

    # ── B & C side-by-side ───────────────────────────────────────────────
    col_b, col_c = st.columns(2, gap="large")
    with col_b:
        st.subheader("B · Internship vs Skill Count")
        st.plotly_chart(_chart_internship_skill(df), width='stretch')
    with col_c:
        st.subheader("C · Skill Category Distribution")
        st.plotly_chart(_chart_skill_categories(all_skills), width='stretch')

    st.divider()

    # ── Domain distribution ───────────────────────────────────────────────
    st.subheader("Domain Distribution")
    st.plotly_chart(_chart_domain_distribution(df), width='stretch')
