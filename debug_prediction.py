"""
Debug: See what the classifier actually receives and predicts.
"""
import joblib
import numpy as np
from utils.features import extract_features

# Load models
clf = joblib.load('models/classification_model.pkl')
le = joblib.load('models/label_encoder.pkl')
cfg = joblib.load('models/feature_names.pkl')

# Simulate user's resume with data science skills
# This is what they said they have: data science ml ai
test_resume = {
    "raw_text": "Data Science Machine Learning AI Deep Learning Python Pandas NumPy Scikit-learn TensorFlow PyTorch Algorithms Statistics",
    "skills": ["data science", "machine learning", "ai", "python", "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch"],
    "projects": "Built ML prediction models",
    "certifications": "ML Specialization",
    "internships": "Data Science Internship"
}

print("=" * 80)
print("YOUR RESUME INPUT")
print("=" * 80)
print(f"Raw text: {test_resume['raw_text']}")
print(f"Skills: {test_resume['skills']}")

# Extract features
numeric_features = extract_features(test_resume)
print(f"\nNumeric features: {numeric_features.to_dict('records')[0]}")

# Build the full input as the classifier would receive it
X_for_clf = numeric_features.copy()
X_for_clf.insert(0, "raw_text", test_resume["raw_text"])

print(f"\nFull input shape: {X_for_clf.shape}")
print(f"Columns sent to classifier: {X_for_clf.columns.tolist()}")

# Make prediction
print("\n" + "=" * 80)
print("MODEL PREDICTION")
print("=" * 80)

proba = clf.predict_proba(X_for_clf)[0]
roles = le.classes_
ranked = sorted([(roles[i], proba[i]) for i in range(len(roles))], key=lambda x: x[1], reverse=True)

print(f"\nAll predictions (sorted by probability):")
for role, prob in ranked:
    print(f"  {role}: {prob*100:.2f}%")

print(f"\nTop prediction: {ranked[0][0]} ({ranked[0][1]*100:.2f}%)")
print(f"Expected prediction: Data Scientist or AI Engineer (>50%)")

if ranked[0][0] not in ["Data Scientist", "AI Engineer", "Machine Learning Engineer"]:
    print(f"\n⚠️  PROBLEM: Got {ranked[0][0]} instead of a Data Science/ML role!")
