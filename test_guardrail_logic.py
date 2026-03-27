"""Test the guardrail-first recommendation logic."""
from utils.parser import SKILLS_LIST
from utils.role_recommender import recommend_roles, role_match_diagnostics, DEFAULT_MATCH_THRESHOLDS, ROLE_PROFILES
import json

# Test with user's data science skills
test_input = {
    "raw_text": "Data Science Machine Learning AI Deep Learning Python TensorFlow PyTorch Statistics",
    "skills": ["data science", "machine learning", "ai", "python", "tensorflow", "pytorch", "statistics"],
    "projects": "Built ML models",
    "certifications": "ML Cert",
    "internships": "ML Intern"
}

print("=" * 80)
print("TESTING GUARDRAIL-FIRST RECOMMENDATION")
print("=" * 80)
print(f"\nInput skills: {test_input['skills']}")

# Get guardrail recommendations (PRIMARY source)
recs = recommend_roles(test_input['skills'], test_input['raw_text'], top_n=5)
print(f"\nGuardrail recommendations (PRIMARY):")
for rec in recs[:3]:
    print(f"  {rec['role']}: {rec['confidence']:.1f}% confidence")

# Check if first recommendation is strong
if recs and recs[0]['confidence'] >= 30:
    print(f"\n✓ Strong guardrail match: {recs[0]['role']} ({recs[0]['confidence']:.1f}%)")
else:
    print(f"\n⚠ Weak guardrail recommendations (max: {recs[0]['confidence']:.1f}%)")

# Also check diagnostics for context
cfg = json.load(open('models/thresholds_config.json'))
thresholds = {**DEFAULT_MATCH_THRESHOLDS, **cfg.get('role_match_thresholds', {})}

print(f"\n" + "=" * 80)
print("DETAILED DIAGNOSTICS FOR TOP 3 ROLES")
print("=" * 80)

for role in ["Data Scientist", "AI Engineer", "Machine Learning Engineer"]:
    diag = role_match_diagnostics(role, test_input['skills'], test_input['raw_text'], thresholds)
    print(f"\n{role}:")
    print(f"  Confidence: {diag['confidence']:.1f}%")
    print(f"  Eligible: {diag['eligible']}")
    print(f"  Required coverage: {diag['required_coverage']:.1f}%")
    print(f"  Required hits: {len(diag['required_hits'])} / {len(ROLE_PROFILES[role]['required'])}")
    if diag['required_hits']:
        print(f"    - Matched: {', '.join(diag['required_hits'])}")
