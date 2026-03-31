"""
views/analyzer.py
Resume Analyzer with ML-first role prediction + rule-based guardrail diagnostics.
"""

import os
import sys

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.features import FEATURE_COLUMNS, extract_features
from utils.model_loader import (
    get_feature_importances,
    load_classification_model,
    load_label_encoder,
    load_model_comparison,
    load_regression_model,
    load_threshold_config,
)
from utils.parser import parse_resume
from utils.role_recommender import (
    DEFAULT_MATCH_THRESHOLDS,
    ROLE_PROFILES,
    recommend_roles,
    role_match_diagnostics,
    skill_gap_for_role,
)


def _score_color(score: float) -> str:
    if score < 40:
        return "#e74c3c"
    if score < 70:
        return "#f39c12"
    return "#27ae60"


def _score_label(score: float) -> str:
    if score < 40:
        return "Needs Improvement"
    if score < 70:
        return "Developing"
    return "Strong"


def _readiness_score_breakdown(feature_df) -> tuple[float, dict[str, float]]:
    """Compute the exact transparent score formula used to build readiness targets."""
    row = feature_df.iloc[0]

    c_skills = min(float(row["num_skills"]) * 3.0, 30.0)
    c_projects = min(float(row["num_projects"]) * 7.0, 21.0)
    c_certs = min(float(row["certifications_count"]) * 5.0, 15.0)
    c_intern = float(row["internship_flag"]) * 14.0
    c_diversity = float(row["skill_diversity"]) * 7.0

    bonus_flags = (
        float(row["has_python"]) +
        float(row["has_ml"]) +
        float(row["has_web"]) +
        float(row["has_cloud"]) +
        float(row["has_database"]) +
        float(row["has_nlp"]) +
        float(row["has_devops"]) +
        float(row["has_mobile"]) +
        float(row["has_security"]) +
        float(row["has_design"]) +
        min(float(row["certifications_count"]), 3.0)
    )

    total = min(round(c_skills + c_projects + c_certs + c_intern + c_diversity + bonus_flags, 2), 100.0)
    breakdown = {
        "skills_component": c_skills,
        "projects_component": c_projects,
        "certifications_component": c_certs,
        "internship_component": c_intern,
        "diversity_component": c_diversity,
        "bonus_component": bonus_flags,
    }
    return total, breakdown


def _gauge_chart(score: float) -> go.Figure:
    color = _score_color(score)
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"suffix": " / 100", "font": {"size": 32}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 40], "color": "rgba(231,76,60,0.12)"},
                    {"range": [40, 70], "color": "rgba(243,156,18,0.12)"},
                    {"range": [70, 100], "color": "rgba(39,174,96,0.12)"},
                ],
                "threshold": {
                    "line": {"color": color, "width": 3},
                    "thickness": 0.75,
                    "value": score,
                },
            },
        )
    )
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=20, b=20), paper_bgcolor="rgba(0,0,0,0)")
    return fig


def _feature_importance_chart(importances: dict[str, float]) -> go.Figure:
    top = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:8]
    labels = [n.replace("_", " ").title() for n, _ in top]
    values = [v for _, v in top]

    fig = px.bar(
        x=values,
        y=labels,
        orientation="h",
        color=values,
        color_continuous_scale="teal",
        labels={"x": "Importance", "y": ""},
        title="Feature Importance (Readiness Model)",
        height=340,
    )
    fig.update_layout(
        coloraxis_showscale=False,
        margin=dict(l=10, r=10, t=50, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(autorange="reversed"),
    )
    fig.update_traces(marker_line_width=0)
    return fig


def _role_confidence_chart(recs: list[dict]) -> go.Figure:
    roles = [r["role"] for r in recs]
    confs = [r["confidence"] for r in recs]
    colors = [_score_color(c) for c in confs]

    fig = go.Figure(
        go.Bar(
            x=confs,
            y=roles,
            orientation="h",
            marker_color=colors,
            text=[f"{c:.1f}%" for c in confs],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Skill-Based Role Diagnostics",
        xaxis=dict(title="Role Match Confidence %", range=[0, 100]),
        yaxis=dict(autorange="reversed"),
        margin=dict(l=10, r=20, t=50, b=10),
        height=320,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    return fig


def render():
    st.markdown("<h1 style='letter-spacing:1px;'>Resume Analyzer</h1>", unsafe_allow_html=True)
    st.markdown(
        "Upload your PDF resume. Final role prediction is **ML-first** with "
        "guardrail thresholds from skill-coverage diagnostics."
    )
    st.divider()

    uploaded = st.file_uploader("Upload Resume (PDF)", type=["pdf"], help="Only text-based PDF files are supported.")
    col_btn, _ = st.columns([1, 4])
    with col_btn:
        analyze = st.button("Analyze Resume", type="primary", disabled=(uploaded is None))

    if not uploaded:
        st.info("Upload a PDF resume to get started.")
        return
    if not analyze:
        return

    with st.spinner("Parsing resume..."):
        parsed = parse_resume(uploaded.read())

    raw_text = parsed["raw_text"]
    user_skills = parsed["skills"]
    explicit_skills = parsed.get("skills_explicit", [])
    inferred_skills = parsed.get("skills_inferred", [])
    skills_section_found = parsed.get("skills_section_found", False)
    has_projects = parsed["projects"] is not None
    has_certs = parsed["certifications"] is not None
    has_internship = (parsed["internships"] is not None) or (parsed.get("experience") is not None)

    if not raw_text.strip():
        st.error("Could not extract text. Please upload a text-based PDF (not a scanned image).")
        return

    with st.spinner("Extracting features..."):
        X = extract_features(parsed)

    comparison = load_model_comparison()
    thresholds_cfg = load_threshold_config()
    reg_model = load_regression_model()
    clf_model = load_classification_model()
    le = load_label_encoder()

    best_reg_name = comparison.get("best_reg_name", "Unknown")
    best_clf_name = comparison.get("best_clf_name", "Unknown")
    best_clf_acc = comparison.get("classification", {}).get("test_accuracy", 0.0)

    ml_min_probability = float(thresholds_cfg.get("ml_min_probability", 0.35))  # Lower threshold for engineered features
    role_thresholds = {**DEFAULT_MATCH_THRESHOLDS, **thresholds_cfg.get("role_match_thresholds", {})}

    X_cls = X.copy()

    with st.spinner("Running ML predictions..."):
        readiness_score_ml = float(np.clip(reg_model.predict(X)[0], 0, 100))
        readiness_score, readiness_breakdown = _readiness_score_breakdown(X)

        proba = clf_model.predict_proba(X_cls)[0]
        all_idx_desc = np.argsort(proba)[::-1]
        ranked_roles = [(le.inverse_transform([i])[0], float(proba[i])) for i in all_idx_desc]
        top3_roles = ranked_roles[:3]

        ml_role = top3_roles[0][0]
        ml_confidence = top3_roles[0][1]

    # Guardrail diagnostics for all roles (SECONDARY validation/explainability)
    role_diag_map = {
        role: role_match_diagnostics(role, user_skills, raw_text, thresholds=role_thresholds)
        for role in ROLE_PROFILES.keys()
    }
    ml_diag = role_diag_map[ml_role]
    # Always compute diagnostics list for charts/explainability so it is defined in all paths.
    recommendations = recommend_roles(user_skills, raw_text, top_n=5, thresholds=role_thresholds)

    # ML-FIRST DECISION LOGIC
    # Final role is always derived from ML prediction.
    # Rule-based diagnostics are used only for explainability, not for overriding ML class.
    final_role = None
    final_prob = 0.0
    final_diag = None
    final_pass = False
    abstained = False

    if ml_confidence >= ml_min_probability:
        # ML prediction is strong enough on its own
        final_role = ml_role
        final_prob = ml_confidence
        final_diag = ml_diag
        final_pass = True
    else:
        # ML confidence below threshold. Keep ML class as provisional instead of overriding.
        final_role = ml_role
        final_prob = ml_confidence
        final_diag = ml_diag
        final_pass = False
        abstained = False

    st.divider()
    st.subheader("ML-Powered Analysis Results")
    st.markdown(
        f"> **Regression:** {best_reg_name} &nbsp;|&nbsp; "
        f"**Classifier:** {best_clf_name} (Test Accuracy: {best_clf_acc:.1%})"
    )

    col_score, col_role = st.columns(2, gap="large")

    with col_score:
        st.markdown("#### Resume Readiness Score")
        st.plotly_chart(_gauge_chart(readiness_score), width="stretch")
        st.markdown(
            f"<div style='text-align:center; font-size:18px; font-weight:600; color:{_score_color(readiness_score)};'>"
            f"{_score_label(readiness_score)}</div>",
            unsafe_allow_html=True,
        )
        st.caption(
            f"Transparent formula score: {readiness_score:.1f}/100 | "
            f"ML estimate ({best_reg_name}): {readiness_score_ml:.1f}/100"
        )

    with col_role:
        st.markdown("#### Final Job Role Decision")

        if abstained:
            st.error("❌ Low Confidence - Cannot Recommend")
            st.markdown("The ML model and skill diagnostics did not provide sufficient confidence for a strong recommendation.")
            st.markdown(f"- **Top ML prediction:** {ml_role} ({ml_confidence * 100:.1f}%)")
            st.markdown(f"- **ML threshold:** {ml_min_probability * 100:.1f}%")
            st.markdown(f"- **Skill coverage:** Below target thresholds")
        else:
            if final_pass:
                st.success(f"**{final_role}**")
                st.caption("Confidence status: HIGH (passes ML threshold)")
            else:
                st.warning(f"**{final_role}** (provisional - below confidence threshold)")
                st.caption("Confidence status: LOW (model is not sure) - treat this as a tentative suggestion.")
            if final_role in ROLE_PROFILES:
                st.caption(ROLE_PROFILES[final_role]["description"])
            st.markdown(f"**ML Confidence:** {final_prob * 100:.1f}% (Threshold: {ml_min_probability * 100:.1f}%)")

        st.markdown("**Top 3 ML Predictions:**")
        for idx, (role, prob) in enumerate(top3_roles, start=1):
            st.markdown(f"- #{idx} {role} ({prob * 100:.1f}%)")

    st.divider()

    st.subheader("Prediction Explainability")
    e1, e2, e3, e4 = st.columns(4)
    active_prob = final_prob if final_role else ml_confidence
    active_diag = final_diag if final_diag else ml_diag
    e1.metric("Model Probability", f"{active_prob * 100:.1f}%", f"Threshold {ml_min_probability * 100:.1f}%")
    e2.metric("Required Coverage", f"{active_diag['required_coverage']:.1f}%")
    e3.metric("Overall Coverage", f"{active_diag['overall_coverage']:.1f}%")
    e4.metric("Matched Skills", f"{active_diag['matched_count']}")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Threshold Checks**")
        prob_pass = active_prob >= ml_min_probability
        rule_pass = bool(active_diag["eligible"])
        st.markdown(f"- ML probability threshold: {'PASS' if prob_pass else 'FAIL'}")
        st.markdown(f"- Role eligibility threshold: {'PASS' if rule_pass else 'FAIL'}")
        st.markdown(f"- Required effective hits: {active_diag['required_effective_hits']} (need >= {active_diag['min_required_hits_needed']})")
    with c2:
        st.markdown("**Matched Required Skills**")
        if active_diag["required_hits"]:
            st.markdown("- " + "\n- ".join([s.title() for s in active_diag["required_hits"]]))
        else:
            st.caption("None")

    st.divider()

    with st.expander("How Resume Readiness Score Is Calculated", expanded=False):
        st.markdown("The displayed readiness score is computed using this exact formula:")
        st.code(
            "score = min(num_skills*3, 30) + min(num_projects*7, 21) + "
            "min(certifications*5, 15) + internship_flag*14 + "
            "skill_diversity*7 + (10 skill/domain flags + min(certifications,3))",
            language=None,
        )
        st.markdown(f"- Skills component: **{readiness_breakdown['skills_component']:.1f}** / 30")
        st.markdown(f"- Projects component: **{readiness_breakdown['projects_component']:.1f}** / 21")
        st.markdown(f"- Certifications component: **{readiness_breakdown['certifications_component']:.1f}** / 15")
        st.markdown(f"- Internship component: **{readiness_breakdown['internship_component']:.1f}** / 14")
        st.markdown(f"- Skill diversity component: **{readiness_breakdown['diversity_component']:.1f}** / 7")
        st.markdown(f"- Bonus flags component: **{readiness_breakdown['bonus_component']:.1f}**")
        st.markdown(f"- **Final transparent score: {readiness_score:.1f} / 100**")
        st.caption(
            f"ML readiness estimate ({best_reg_name}) is shown for comparison: {readiness_score_ml:.1f} / 100."
        )

    st.divider()

    st.subheader("Skill-Based Role Diagnostics")
    st.plotly_chart(_role_confidence_chart(recommendations), width="stretch")

    if abstained:
        st.warning("No role met confidence and coverage thresholds. Improve role-specific skill coverage.")
    else:
        # Only show skill gap for recommended roles (not abstain cases)
        gap_role = final_role
        st.subheader(f"Skill Gap: {gap_role}")
        gap = skill_gap_for_role(gap_role, user_skills)

        gd1, gd2, gd3 = st.columns(3, gap="large")
        with gd1:
            st.markdown("**Your Detected Skills**")
            if user_skills:
                for s in sorted(user_skills):
                    st.markdown(f"- {s.title()}")
            else:
                st.caption("No skills detected.")

            if skills_section_found:
                st.caption("Source: explicit Skills section (used for prediction).")
                if inferred_skills:
                    st.caption("Also mentioned elsewhere in resume (not primary list):")
                    for s in sorted(inferred_skills):
                        st.markdown(f"- {s.title()}")
            else:
                st.caption("No dedicated Skills section found; extracted from full resume text.")

        with gd2:
            st.markdown("**Matched Skills**")
            if gap["matched"]:
                for s in gap["matched"]:
                    st.markdown(f"- {s.title()}")
            else:
                st.caption("No matches found.")

        with gd3:
            st.markdown("**Skills to Acquire**")
            if gap["missing"]:
                for s in gap["missing"][:10]:
                    st.markdown(f"- {s.title()}")
            else:
                st.success("You have all key skills for this role.")

    st.divider()

    st.subheader("Feature Importance")
    imps = get_feature_importances(reg_model, FEATURE_COLUMNS)
    if imps:
        st.plotly_chart(_feature_importance_chart(imps), width="stretch")
    else:
        st.info("Feature importance not available for this model type.")

    st.divider()

    with st.expander("Parsed Resume Summary", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Skills Found", len(user_skills))
        c2.metric("Projects Section", "Yes" if has_projects else "No")
        c3.metric("Certifications", "Yes" if has_certs else "No")
        c4.metric("Internship", "Yes" if has_internship else "No")
        st.markdown(f"- Skills section detected: **{'Yes' if skills_section_found else 'No'}**")
        if explicit_skills:
            st.markdown("- Explicit skills extracted:")
            st.markdown("- " + "\n- ".join([s.title() for s in sorted(explicit_skills)]))
        if inferred_skills:
            st.markdown("- Mentioned outside Skills section (informational):")
            st.markdown("- " + "\n- ".join([s.title() for s in sorted(inferred_skills)]))
        st.caption("Raw text preview (first 800 chars):")
        st.code(raw_text[:800], language=None)

    with st.expander("Feature Vector (Model Input)", expanded=False):
        st.dataframe(X, width=900)
        st.caption("Engineered numeric features fed to the ML pipeline.")
