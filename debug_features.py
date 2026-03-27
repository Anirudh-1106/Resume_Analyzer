"""
Debug: Trace feature extraction for a test resume with data science skills.
"""
import pandas as pd
from utils.features import extract_features

# Simulate a resume with data science/ML/AI skills
test_resume = {
    "raw_text": "Data Science Machine Learning AI Deep Learning Python Pandas NumPy Scikit-learn TensorFlow PyTorch",
    "skills": ["data science", "machine learning", "ai", "python", "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch"],
    "projects": "Built ML models for prediction",
    "certifications": "ML Certification",
    "internships": "ML Intern"
}

print("=" * 80)
print("TEST RESUME")
print("=" * 80)
print(f"Raw text: {test_resume['raw_text']}")
print(f"Skills: {test_resume['skills']}")

features = extract_features(test_resume)
print("\n" + "=" * 80)
print("EXTRACTED FEATURES")
print("=" * 80)
print(f"Feature names: {features.columns.tolist()}")
print(f"\nFeature values:")
for col in features.columns:
    val = features[col].values[0]
    print(f"  {col}: {val}")

# Now let's check what a cybersecurity resume looks like
cyber_resume = {
    "raw_text": "Cybersecurity Penetration Testing Ethical Hacking Linux Bash Python",
    "skills": ["cybersecurity", "penetration testing", "ethical hacking", "linux", "bash", "python"],
    "projects": "Security audits",
    "certifications": "CEH",
    "internships": "Security Intern"
}

print("\n" + "=" * 80)
print("CYBER RESUME (FOR COMPARISON)")
print("=" * 80)
print(f"Raw text: {cyber_resume['raw_text']}")
print(f"Skills: {cyber_resume['skills']}")

cyber_features = extract_features(cyber_resume)
print(f"\nFeature values:")
for col in cyber_features.columns:
    val = cyber_features[col].values[0]
    print(f"  {col}: {val}")

print("\n" + "=" * 80)
print("FEATURE DIFFERENCES")
print("=" * 80)
for col in features.columns:
    ds_val = features[col].values[0]
    cy_val = cyber_features[col].values[0]
    if ds_val != cy_val:
        print(f"  {col}: DS={ds_val}, Cyber={cy_val}")
