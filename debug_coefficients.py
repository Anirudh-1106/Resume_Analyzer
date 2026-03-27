"""Debug script to analyze why Cybersecurity Engineer got high probability."""
import json
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load model and label encoder
clf = joblib.load('models/classification_model.pkl')
le = joblib.load('models/label_encoder.pkl')

# Get class indices
roles = le.classes_
print("=" * 80)
print("CLASSIFIER COEFFICIENTS ANALYSIS (What features push toward each role)")
print("=" * 80)

# Extract coefficients from the base estimator (before calibration)
base_clf = clf.base_estimator_
coefs = base_clf.coef_  # Shape: (n_classes, n_features)

print(f"\nModel: {type(base_clf).__name__}")
print(f"Number of roles (classes): {len(roles)}")
print(f"Number of features: {coefs.shape[1]}")

# Find which role has highest coefficient for which features
print("\n" + "=" * 80)
print("MOST INFLUENTIAL FEATURES PER ROLE (Top 5)")
print("=" * 80)

for role_idx, role in enumerate(roles):
    role_coefs = coefs[role_idx]
    top_indices = np.argsort(role_coefs)[-5:][::-1]
    bottom_indices = np.argsort(role_coefs)[:5]
    
    print(f"\n{role}:")
    print(f"  Intercept (base probability): {base_clf.intercept_[role_idx]:.4f}")
    print(f"  Top 5 POSITIVE features (push toward this role):")
    for idx in top_indices:
        print(f"    - Feature {idx}: +{role_coefs[idx]:.4f}")

print("\n" + "=" * 80)
print("CYBERSECURITY ENGINEER SPECIFICALLY")
print("=" * 80)
cyber_idx = np.where(roles == "Cybersecurity Engineer")[0][0]
cyber_coefs = coefs[cyber_idx]
print(f"\nTop 10 features pushing toward Cybersecurity Engineer:")
for i in np.argsort(cyber_coefs)[::-1][:10]:
    print(f"  Feature {i}: {cyber_coefs[i]:.4f}")

print(f"\nIntercept (base bias): {base_clf.intercept_[cyber_idx]:.4f}")
print(f"\nTop features for DATA SCIENTIST (for comparison):")
ds_idx = np.where(roles == "Data Scientist")[0][0]
ds_coefs = coefs[ds_idx]
for i in np.argsort(ds_coefs)[::-1][:10]:
    print(f"  Feature {i}: {ds_coefs[i]:.4f}")
