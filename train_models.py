"""
train_models.py
Train and compare ML models for:
1) Regression: readiness score
2) Classification: job role from text + engineered features

Outputs:
- models/regression_model.pkl
- models/classification_model.pkl (calibrated)
- models/label_encoder.pkl
- models/feature_names.pkl
- models/model_comparison.json
- models/thresholds_config.json
"""

import ast
import json
import os
import time

import joblib
import numpy as np
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC, SVR


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "data", "structured_resumes.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

FEATURES = [
    "num_skills",
    "num_projects",
    "certifications_count",
    "internship_flag",
    "skill_diversity",
    "has_python",
    "has_ml",
    "has_web",
    "has_cloud",
    "has_database",
    "has_nlp",
    "has_devops",
    "has_mobile",
    "has_security",
    "has_design",
]


def safe_parse_list(val) -> list:
    if pd.isna(val):
        return []
    if isinstance(val, list):
        return val
    val = str(val).strip()
    if val.startswith("["):
        try:
            parsed = ast.literal_eval(val)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except Exception:
            pass
    return [x.strip() for x in val.split(",") if x.strip()]


def keyword_flag(text, keywords: list) -> int:
    if pd.isna(text):
        return 0
    txt = str(text).lower()
    return 1 if any(k in txt for k in keywords) else 0


def compute_readiness_score(row) -> float:
    score = min(row["num_skills"] * 3, 30)
    score += min(row["num_projects"] * 7, 21)
    score += min(row["certifications_count"] * 5, 15)
    score += row["internship_flag"] * 14
    score += row["skill_diversity"] * 7
    score += (
        row["has_python"] + row["has_ml"] + row["has_web"] + row["has_cloud"] +
        row["has_database"] + row["has_nlp"] + row["has_devops"] + row["has_mobile"] +
        row["has_security"] + row["has_design"] + min(row["certifications_count"], 3)
    )
    return min(round(float(score), 2), 100)


def make_reg_model(name: str):
    if name == "Random Forest":
        return RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1)
    if name == "Gradient Boosting":
        return GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42)
    if name == "Ridge Regression":
        return Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=1.0))])
    if name == "SVR":
        return Pipeline([("scale", StandardScaler()), ("model", SVR(kernel="rbf", C=10.0))])
    if name == "KNN Regressor":
        return Pipeline([("scale", StandardScaler()), ("model", KNeighborsRegressor(n_neighbors=7))])
    raise ValueError(name)


def make_clf_model(name: str):
    # Use ONLY engineered numeric features (no TF-IDF text)
    # This is more robust and interpretable
    numeric_preprocessor = ColumnTransformer(
        transformers=[("num", "passthrough", FEATURES)],
        remainder="drop",
    )

    if name == "Random Forest":
        return Pipeline([
            ("prep", numeric_preprocessor),
            ("model", RandomForestClassifier(n_estimators=250, max_depth=15, random_state=42, n_jobs=-1)),
        ])
    if name == "Logistic Regression":
        return Pipeline([
            ("prep", numeric_preprocessor),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=1500, random_state=42, C=0.5)),
        ])
    if name == "Gradient Boosting":
        return Pipeline([
            ("prep", numeric_preprocessor),
            ("model", RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1)),
        ])
    if name == "SVM":
        return Pipeline([
            ("prep", numeric_preprocessor),
            ("scale", StandardScaler()),
            ("model", SVC(kernel="rbf", C=5.0, probability=True, random_state=42)),
        ])
    if name == "KNN":
        return Pipeline([
            ("prep", numeric_preprocessor),
            ("scale", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=11)),
        ])
    raise ValueError(name)


def top_k_accuracy(y_true, proba, k: int) -> float:
    topk = np.argsort(proba, axis=1)[:, -k:]
    hits = [int(y in topk_row) for y, topk_row in zip(y_true, topk)]
    return float(np.mean(hits)) if hits else 0.0


def tune_probability_threshold(y_true, proba):
    best = {"threshold": 0.25, "score": -1.0, "coverage": 0.0, "accepted_f1": 0.0}
    y_pred = np.argmax(proba, axis=1)
    conf = np.max(proba, axis=1)

    for t in np.arange(0.15, 0.60, 0.03):
        accepted = conf >= t
        if accepted.sum() < 10:
            continue
        coverage = float(np.mean(accepted))
        f1 = f1_score(y_true[accepted], y_pred[accepted], average="weighted", zero_division=0)
        score = (0.8 * f1) + (0.2 * coverage)  # Prioritize F1-score
        if score > best["score"]:
            best = {
                "threshold": round(float(t), 2),
                "score": float(score),
                "coverage": coverage,
                "accepted_f1": float(f1),
            }
    # Reasonable bounds for engineered-features-only classifier
    best["threshold"] = max(0.20, min(0.50, float(best["threshold"])))
    return best


def main():
    print("=" * 72)
    print(" CSE Resume Intelligence - Final ML Training Pipeline")
    print("=" * 72)

    df = pd.read_csv(CSV_PATH)
    if "job_role" not in df.columns:
        raise ValueError("Dataset missing 'job_role' column")

    df = df.dropna(subset=["job_role", "raw_text"]).copy()
    print(f"Rows: {len(df)} | Roles: {df['job_role'].nunique()}")

    df["skills_list"] = df["skills"].apply(safe_parse_list)
    df["projects_list"] = df["projects"].apply(safe_parse_list)
    df["certs_list"] = df["certifications"].apply(safe_parse_list)
    df["intern_list"] = df["internships"].apply(safe_parse_list)

    df["num_skills"] = df["skills_list"].apply(len)
    df["num_projects"] = df["projects_list"].apply(len)
    df["certifications_count"] = df["certs_list"].apply(len)
    df["internship_flag"] = df["intern_list"].apply(lambda x: 1 if x else 0)
    df["skill_diversity"] = df["skills_list"].apply(lambda x: len(set(s.lower() for s in x)) / max(len(x), 1))

    kw_map = {
        "has_python": ["python"],
        "has_ml": ["machine learning", "deep learning", "tensorflow", "pytorch", "xgboost"],
        "has_web": ["html", "css", "javascript", "react", "next.js", "vue", "angular", "tailwind"],
        "has_cloud": ["aws", "azure", "gcp", "cloud", "terraform", "kubernetes"],
        "has_database": ["sql", "mysql", "mongodb", "postgresql", "database", "snowflake", "redis"],
        "has_nlp": ["nlp", "natural language", "transformers", "bert", "langchain", "llm", "hugging face", "spacy"],
        "has_devops": ["docker", "kubernetes", "ci/cd", "devops", "ansible", "terraform", "jenkins", "github actions"],
        "has_mobile": ["flutter", "react native", "android", "ios", "swift", "kotlin", "swiftui", "jetpack"],
        "has_security": ["penetration testing", "ethical hacking", "owasp", "siem", "metasploit", "cybersecurity", "zero trust"],
        "has_design": ["figma", "adobe xd", "wireframing", "prototyping", "user research", "ux", "ui design", "usability"],
    }
    for col, keywords in kw_map.items():
        df[col] = df["raw_text"].apply(lambda t, kw=keywords: keyword_flag(t, kw))

    df["readiness_score"] = df.apply(compute_readiness_score, axis=1)

    # Targets and inputs
    y_reg = df["readiness_score"].values
    le = LabelEncoder()
    y_cls = le.fit_transform(df["job_role"])

    X_reg = df[FEATURES].copy()
    X_cls = df[["raw_text", *FEATURES]].copy()

    # Split train/val/test with stratification for classification.
    Xc_train_val, Xc_test, yc_train_val, yc_test = train_test_split(
        X_cls, y_cls, test_size=0.15, random_state=42, stratify=y_cls
    )
    Xc_train, Xc_val, yc_train, yc_val = train_test_split(
        Xc_train_val, yc_train_val, test_size=0.1765, random_state=42, stratify=yc_train_val
    )

    Xr_train_val, Xr_test, yr_train_val, yr_test = train_test_split(
        X_reg, y_reg, test_size=0.15, random_state=42
    )
    Xr_train, Xr_val, yr_train, yr_val = train_test_split(
        Xr_train_val, yr_train_val, test_size=0.1765, random_state=42
    )

    # ---------------- Regression model selection on validation ----------------
    reg_names = ["Random Forest", "Gradient Boosting", "Ridge Regression", "SVR", "KNN Regressor"]
    reg_results = {}
    best_reg_name = None
    best_reg_model = None
    best_reg_val_mae = float("inf")

    for name in reg_names:
        model = make_reg_model(name)
        t0 = time.time()
        model.fit(Xr_train, yr_train)
        val_pred = model.predict(Xr_val)
        elapsed = time.time() - t0

        val_mae = mean_absolute_error(yr_val, val_pred)
        val_r2 = r2_score(yr_val, val_pred)
        cv = cross_val_score(model, Xr_train, yr_train, cv=3, scoring="neg_mean_absolute_error")

        reg_results[name] = {
            "val_mae": round(float(val_mae), 4),
            "val_r2": round(float(val_r2), 4),
            "cv_mae": round(float(-cv.mean()), 4),
            "train_time_sec": round(float(elapsed), 2),
            # Backward-compatible keys used by dashboard
            "mae": round(float(val_mae), 4),
            "r2_score": round(float(val_r2), 4),
        }

        if val_mae < best_reg_val_mae:
            best_reg_val_mae = val_mae
            best_reg_name = name
            best_reg_model = model

    # Final regression test metrics (selected by validation)
    yr_test_pred = best_reg_model.predict(Xr_test)
    reg_test_mae = mean_absolute_error(yr_test, yr_test_pred)
    reg_test_r2 = r2_score(yr_test, yr_test_pred)

    # ---------------- Classification model selection on validation ----------------
    clf_names = ["Random Forest", "Logistic Regression", "SVM", "KNN"]
    clf_results = {}
    all_clf_models = {}
    best_clf_name = None
    best_clf_model = None
    best_clf_val_f1 = -1.0

    for name in clf_names:
        model = make_clf_model(name)
        t0 = time.time()
        model.fit(Xc_train, yc_train)
        val_pred = model.predict(Xc_val)
        val_proba = model.predict_proba(Xc_val)
        elapsed = time.time() - t0

        val_acc = accuracy_score(yc_val, val_pred)
        val_prec = precision_score(yc_val, val_pred, average="weighted", zero_division=0)
        val_rec = recall_score(yc_val, val_pred, average="weighted", zero_division=0)
        val_f1 = f1_score(yc_val, val_pred, average="weighted", zero_division=0)
        val_top3 = top_k_accuracy(yc_val, val_proba, 3)
        cv = cross_val_score(model, Xc_train, yc_train, cv=3, scoring="accuracy")

        clf_results[name] = {
            "val_accuracy": round(float(val_acc), 4),
            "val_precision": round(float(val_prec), 4),
            "val_recall": round(float(val_rec), 4),
            "val_f1_score": round(float(val_f1), 4),
            "val_top3_accuracy": round(float(val_top3), 4),
            "cv_accuracy": round(float(cv.mean()), 4),
            "train_time_sec": round(float(elapsed), 2),
            "confusion_matrix": confusion_matrix(yc_val, val_pred).tolist(),
            # Backward-compatible keys used by dashboard
            "accuracy": round(float(val_acc), 4),
            "precision": round(float(val_prec), 4),
            "recall": round(float(val_rec), 4),
            "f1_score": round(float(val_f1), 4),
        }

        all_clf_models[name] = model

        if val_f1 > best_clf_val_f1:
            best_clf_val_f1 = val_f1
            best_clf_name = name
            best_clf_model = model

    # Calibrate probabilities on training data (no prefit mode in current sklearn).
    calibrated_clf = CalibratedClassifierCV(
        estimator=make_clf_model(best_clf_name),
        method="sigmoid",
        cv=3,
    )
    calibrated_clf.fit(Xc_train, yc_train)

    # Threshold tuning on validation calibrated probabilities.
    val_cal_proba = calibrated_clf.predict_proba(Xc_val)
    threshold_choice = tune_probability_threshold(yc_val, val_cal_proba)

    # Final classification test metrics.
    test_pred = calibrated_clf.predict(Xc_test)
    test_proba = calibrated_clf.predict_proba(Xc_test)

    test_acc = accuracy_score(yc_test, test_pred)
    test_prec = precision_score(yc_test, test_pred, average="weighted", zero_division=0)
    test_rec = recall_score(yc_test, test_pred, average="weighted", zero_division=0)
    test_f1 = f1_score(yc_test, test_pred, average="weighted", zero_division=0)
    test_top3 = top_k_accuracy(yc_test, test_proba, 3)

    report = classification_report(yc_test, test_pred, target_names=le.classes_, output_dict=True, zero_division=0)
    cm = confusion_matrix(yc_test, test_pred).tolist()

    # Save models
    joblib.dump(best_reg_model, os.path.join(MODELS_DIR, "regression_model.pkl"))
    joblib.dump(calibrated_clf, os.path.join(MODELS_DIR, "classification_model.pkl"))
    joblib.dump(le, os.path.join(MODELS_DIR, "label_encoder.pkl"))
    joblib.dump(FEATURES, os.path.join(MODELS_DIR, "feature_names.pkl"))

    for name, model in all_clf_models.items():
        safe_name = name.lower().replace(" ", "_")
        joblib.dump(model, os.path.join(MODELS_DIR, f"clf_{safe_name}.pkl"))

    # Save comparison metrics
    comparison = {
        "regression": {
            "results": reg_results,
            "best_model": best_reg_name,
            "best_mae": round(float(best_reg_val_mae), 4),
            "best_val_mae": round(float(best_reg_val_mae), 4),
            "test_mae": round(float(reg_test_mae), 4),
            "test_r2": round(float(reg_test_r2), 4),
        },
        "classification": {
            "results": clf_results,
            "best_model": best_clf_name,
            "best_accuracy": round(float(test_acc), 4),
            "test_accuracy": round(float(test_acc), 4),
            "test_precision": round(float(test_prec), 4),
            "test_recall": round(float(test_rec), 4),
            "test_f1_score": round(float(test_f1), 4),
            "test_top1_accuracy": round(float(test_acc), 4),
            "test_top3_accuracy": round(float(test_top3), 4),
            "classification_report": report,
            "confusion_matrix": cm,
            "role_labels": list(le.classes_),
            "probability_calibration": {
                "method": "sigmoid",
                "base_model": best_clf_name,
            },
        },
        "dataset": {
            "total_rows": int(len(df)),
            "num_roles": int(df["job_role"].nunique()),
            "num_features": int(len(FEATURES)),
            "feature_names": FEATURES,
            "train_size": int(len(Xc_train)),
            "val_size": int(len(Xc_val)),
            "test_size": int(len(Xc_test)),
        },
        "classification_input_mode": "text_plus_numeric",
        "classification_input_columns": ["raw_text", *FEATURES],
        "best_clf_name": best_clf_name,
        "best_reg_name": best_reg_name,
    }

    with open(os.path.join(MODELS_DIR, "model_comparison.json"), "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    thresholds_config = {
        "ml_min_probability": threshold_choice["threshold"],
        "threshold_tuning": {
            "coverage": round(threshold_choice["coverage"], 4),
            "accepted_f1": round(threshold_choice["accepted_f1"], 4),
            "selection_score": round(threshold_choice["score"], 4),
            "source": "validation_split",
        },
        "role_match_thresholds": {
            "min_required_coverage": 0.30,
            "min_required_hits": 2,
            "min_overall_coverage": 0.20,
            "text_hit_weight_required": 0.40,
            "text_hit_weight_preferred": 0.30,
        },
    }

    with open(os.path.join(MODELS_DIR, "thresholds_config.json"), "w", encoding="utf-8") as f:
        json.dump(thresholds_config, f, indent=2)

    print("=" * 72)
    print(f"Best regression model: {best_reg_name}")
    print(f"Best classifier model: {best_clf_name} (calibrated)")
    print(f"Test Accuracy: {test_acc:.4f} | Test Top-3: {test_top3:.4f}")
    print(f"Saved thresholds_config.json with ml_min_probability={threshold_choice['threshold']}")
    print("=" * 72)


if __name__ == "__main__":
    main()
