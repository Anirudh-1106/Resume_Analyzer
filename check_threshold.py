"""Check validation probabilities to understand threshold selection."""
import json
import joblib
import numpy as np

# Load the config to see what threshold was chosen
config = json.load(open('models/thresholds_config.json'))
print("Threshold config:")
for key, val in config.items():
    print(f"  {key}: {val}")

# Load model to understand its default behavior
clf = joblib.load('models/classification_model.pkl')
le = joblib.load('models/label_encoder.pkl')

print(f"\nModel type: {type(clf)}")
print(f"Base estimator: {type(clf.base_estimator_)}")
print(f"Number of classes: {len(le.classes_)}")
