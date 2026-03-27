"""
pages/model_comparison.py
Model Comparison Dashboard - Shows all ML algorithms trained and their metrics.
Displays accuracy, F1, confusion matrices, classification report, etc.
"""

import os
import sys

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # noqa: E402

from utils.model_loader import load_model_comparison


def render():
    st.markdown(
        "<h1 style='letter-spacing:1px;'>Model Comparison Dashboard</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "Comparison of **5 ML algorithms** trained on the CSE resume dataset. "
        "Both regression (Readiness Score) and classification (Job Role) tasks."
    )
    st.divider()

    comparison = load_model_comparison()
    if not comparison:
        st.error("No model comparison data found. Run `python train_models.py` first.")
        return

    reg_data = comparison.get("regression", {})
    clf_data = comparison.get("classification", {})
    ds_data  = comparison.get("dataset", {})

    # -- Dataset Info ---------------------------------------------------------
    st.subheader("Dataset Overview")
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Total Resumes",   f"{ds_data.get('total_rows', 0):,}")
    d2.metric("Job Roles",       ds_data.get("num_roles", 0))
    d3.metric("Features",        ds_data.get("num_features", 0))
    d4.metric("Train / Test",    f"{ds_data.get('train_size', 0)} / {ds_data.get('test_size', 0)}")

    st.divider()

    # ========================================================================
    # REGRESSION COMPARISON
    # ========================================================================
    st.subheader("Regression Models (Readiness Score)")
    st.markdown(f"**Best Model:** {reg_data.get('best_model', '?')} (MAE = {reg_data.get('best_mae', 0):.4f})")

    reg_results = reg_data.get("results", {})
    if reg_results:
        # Table
        raw_df = pd.DataFrame(reg_results).T
        raw_df.index.name = "Algorithm"
        raw_df = raw_df.reset_index()

        # Coalesce aliases from different training versions into one stable schema.
        reg_df = pd.DataFrame({
            "Algorithm": raw_df["Algorithm"],
            "MAE": raw_df["val_mae"] if "val_mae" in raw_df.columns else raw_df.get("mae", np.nan),
            "R2 Score": raw_df["val_r2"] if "val_r2" in raw_df.columns else raw_df.get("r2_score", np.nan),
            "CV MAE": raw_df.get("cv_mae", np.nan),
            "Train Time (s)": raw_df.get("train_time_sec", np.nan),
        }).sort_values("MAE")

        st.dataframe(
            reg_df.style.highlight_min(subset=["MAE", "CV MAE"], color="#27ae6030")
                        .highlight_max(subset=["R2 Score"], color="#27ae6030")
                        .format({"MAE": "{:.4f}", "R2 Score": "{:.4f}", "CV MAE": "{:.4f}", "Train Time (s)": "{:.2f}"}),
            width=800,
            hide_index=True,
        )

        # Bar chart comparison
        col_r1, col_r2 = st.columns(2)

        with col_r1:
            fig_mae = px.bar(
                reg_df, x="Algorithm", y="MAE",
                color="MAE", color_continuous_scale="RdYlGn_r",
                title="MAE Comparison (lower is better)",
                height=350,
            )
            fig_mae.update_layout(
                coloraxis_showscale=False,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_mae, width="stretch")

        with col_r2:
            fig_r2 = px.bar(
                reg_df, x="Algorithm", y="R2 Score",
                color="R2 Score", color_continuous_scale="RdYlGn",
                title="R2 Score Comparison (higher is better)",
                height=350,
            )
            fig_r2.update_layout(
                coloraxis_showscale=False,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_r2, width="stretch")

    st.divider()

    # ========================================================================
    # CLASSIFICATION COMPARISON
    # ========================================================================
    st.subheader("Classification Models (Job Role Prediction)")
    st.markdown(
        f"**Best Model:** {clf_data.get('best_model', '?')} "
        f"(Accuracy = {clf_data.get('best_accuracy', 0):.2%})"
    )

    clf_results = clf_data.get("results", {})
    if clf_results:
        # Table
        rows = []
        for name, metrics in clf_results.items():
            rows.append({
                "Algorithm": name,
                "Accuracy": metrics["accuracy"],
                "Precision": metrics["precision"],
                "Recall": metrics["recall"],
                "F1 Score": metrics["f1_score"],
                "CV Accuracy": metrics["cv_accuracy"],
                "Train Time (s)": metrics["train_time_sec"],
            })
        clf_df = pd.DataFrame(rows).sort_values("Accuracy", ascending=False)

        st.dataframe(
            clf_df.style.highlight_max(
                subset=["Accuracy", "Precision", "Recall", "F1 Score", "CV Accuracy"],
                color="#27ae6030"
            ).format({
                "Accuracy": "{:.2%}", "Precision": "{:.2%}", "Recall": "{:.2%}",
                "F1 Score": "{:.2%}", "CV Accuracy": "{:.2%}", "Train Time (s)": "{:.2f}"
            }),
            width=900,
            hide_index=True,
        )

        # Grouped bar chart: Accuracy, Precision, Recall, F1
        metrics_long = []
        for _, row in clf_df.iterrows():
            for metric in ["Accuracy", "Precision", "Recall", "F1 Score"]:
                metrics_long.append({
                    "Algorithm": row["Algorithm"],
                    "Metric": metric,
                    "Value": row[metric] * 100,
                })
        metrics_long_df = pd.DataFrame(metrics_long)

        fig_clf = px.bar(
            metrics_long_df,
            x="Algorithm", y="Value", color="Metric",
            barmode="group",
            title="Classification Metrics Comparison",
            labels={"Value": "Percentage (%)"},
            height=400,
            color_discrete_sequence=["#3498db", "#2ecc71", "#f39c12", "#e74c3c"],
        )
        fig_clf.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_clf, width="stretch")

        # Confusion Matrix for best model
        st.subheader("Confusion Matrix (Best Classifier)")
        role_labels = clf_data.get("role_labels", [])
        best_name = clf_data.get("best_model", "")
        best_cm = clf_results.get(best_name, {}).get("confusion_matrix")

        if best_cm and role_labels:
            cm_array = np.array(best_cm)
            fig_cm = px.imshow(
                cm_array,
                x=role_labels, y=role_labels,
                color_continuous_scale="Blues",
                labels=dict(x="Predicted", y="Actual", color="Count"),
                title=f"Confusion Matrix - {best_name}",
                height=600,
                aspect="auto",
            )
            fig_cm.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(tickangle=45),
            )
            st.plotly_chart(fig_cm, width="stretch")

        # Classification Report
        report = clf_data.get("classification_report", {})
        if report:
            st.subheader("Per-Class Classification Report")
            report_rows = []
            for cls_name, metrics in report.items():
                if cls_name in ("accuracy", "macro avg", "weighted avg"):
                    continue
                if isinstance(metrics, dict):
                    report_rows.append({
                        "Role": cls_name,
                        "Precision": metrics.get("precision", 0),
                        "Recall": metrics.get("recall", 0),
                        "F1-Score": metrics.get("f1-score", 0),
                        "Support": int(metrics.get("support", 0)),
                    })

            if report_rows:
                report_df = pd.DataFrame(report_rows)
                st.dataframe(
                    report_df.style.format({
                        "Precision": "{:.2%}", "Recall": "{:.2%}", "F1-Score": "{:.2%}"
                    }).background_gradient(subset=["F1-Score"], cmap="RdYlGn"),
                    width=800,
                    hide_index=True,
                )

                # Add summary rows
                for avg_type in ("macro avg", "weighted avg"):
                    if avg_type in report:
                        m = report[avg_type]
                        st.markdown(
                            f"**{avg_type.title()}:** "
                            f"Precision={m['precision']:.2%} | "
                            f"Recall={m['recall']:.2%} | "
                            f"F1={m['f1-score']:.2%}"
                        )

    st.divider()

    # Training time comparison
    st.subheader("Training Time Comparison")
    time_data = []
    for name, metrics in reg_results.items():
        time_data.append({"Algorithm": name, "Task": "Regression", "Time (s)": metrics["train_time_sec"]})
    for name, metrics in clf_results.items():
        time_data.append({"Algorithm": name, "Task": "Classification", "Time (s)": metrics["train_time_sec"]})

    time_df = pd.DataFrame(time_data)
    fig_time = px.bar(
        time_df, x="Algorithm", y="Time (s)", color="Task",
        barmode="group", title="Training Time by Algorithm",
        height=350,
        color_discrete_sequence=["#3498db", "#e74c3c"],
    )
    fig_time.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_time, width="stretch")

    st.divider()

    # Key findings
    st.subheader("Key Findings")
    st.markdown(f"""
- **5 algorithms** were trained and compared for each task (10 models total)
- **Best Regression:** {reg_data.get('best_model', '?')} achieved the lowest MAE of {reg_data.get('best_mae', 0):.4f}
- **Best Classification:** {clf_data.get('best_model', '?')} achieved {clf_data.get('best_accuracy', 0):.2%} accuracy
- **Cross-validation** (5-fold) was used to validate generalization
- **StandardScaler** was applied for SVM, KNN, and Logistic Regression
- **{ds_data.get('num_features', 0)} features** were engineered from resume text
- Dataset is **balanced** ({ds_data.get('total_rows', 0)} resumes across {ds_data.get('num_roles', 0)} roles)
""")
