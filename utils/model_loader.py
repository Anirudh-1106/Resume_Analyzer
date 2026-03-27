"""
utils/model_loader.py
Loads trained models and configuration artifacts.
"""

import json
import os
import subprocess
import sys

import joblib
import streamlit as st

_MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def _model_path(name: str) -> str:
    return os.path.abspath(os.path.join(_MODELS_DIR, name))


def _train_models_with_current_python() -> None:
    """Rebuild artifacts using the same interpreter running Streamlit."""
    train_script = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "train_models.py"))
    subprocess.run([sys.executable, train_script], check=True, cwd=os.path.dirname(train_script))


def _safe_joblib_load(path: str, artifact_name: str):
    try:
        return joblib.load(path)
    except Exception as exc:
        # Common incompatibility signatures when pickle was produced by another sklearn build/version.
        msg = str(exc)
        mismatch_hint = (
            "__pyx_unpickle_" in msg
            or "InconsistentVersionWarning" in msg
            or "Can't get attribute" in msg
        )
        if not mismatch_hint:
            raise

        st.warning(
            f"Detected incompatible {artifact_name}. Rebuilding models with current Python environment..."
        )
        _train_models_with_current_python()
        return joblib.load(path)


@st.cache_resource(show_spinner="Loading regression model...")
def load_regression_model():
    path = _model_path("regression_model.pkl")
    if not os.path.exists(path):
        _train_models_with_current_python()
    return _safe_joblib_load(path, "regression_model.pkl")


@st.cache_resource(show_spinner="Loading classification model...")
def load_classification_model():
    path = _model_path("classification_model.pkl")
    if not os.path.exists(path):
        _train_models_with_current_python()
    return _safe_joblib_load(path, "classification_model.pkl")


@st.cache_resource(show_spinner="Loading label encoder...")
def load_label_encoder():
    path = _model_path("label_encoder.pkl")
    if not os.path.exists(path):
        _train_models_with_current_python()
    return _safe_joblib_load(path, "label_encoder.pkl")


@st.cache_data(show_spinner=False)
def load_model_comparison() -> dict:
    path = _model_path("model_comparison.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_threshold_config() -> dict:
    path = _model_path("thresholds_config.json")
    if not os.path.exists(path):
        return {
            "ml_min_probability": 0.55,
            "role_match_thresholds": {
                "min_required_coverage": 0.30,
                "min_required_hits": 2,
                "min_overall_coverage": 0.20,
                "text_hit_weight_required": 0.40,
                "text_hit_weight_preferred": 0.30,
            },
        }
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_feature_importances(model, feature_names: list[str]) -> dict[str, float]:
    if hasattr(model, "feature_importances_"):
        return dict(zip(feature_names, model.feature_importances_))
    if hasattr(model, "coef_"):
        import numpy as np

        coefs = np.abs(model.coef_).mean(axis=0) if model.coef_.ndim > 1 else np.abs(model.coef_)
        return dict(zip(feature_names, coefs))
    return {}
