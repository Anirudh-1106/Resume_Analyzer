"""Verify models and configuration."""
from utils.model_loader import load_regression_model, load_classification_model, load_threshold_config
import json

reg = load_regression_model()
clf = load_classification_model()
cfg = load_threshold_config()

print("✓ Models loaded successfully")
print(f"  Regression: {type(reg).__name__}")
print(f"  Classifier: {type(clf).__name__}")
print(f"  ML threshold: {cfg.get('ml_min_probability')}")
print(f"\n✓ App ready to run!")
