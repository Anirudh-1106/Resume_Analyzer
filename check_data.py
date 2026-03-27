"""Check if training data is properly labeled."""
import pandas as pd

df = pd.read_csv('data/structured_resumes.csv')
print("=" * 80)
print("TRAINING DATA SAMPLES")
print("=" * 80)

# Show a Data Scientist resume
ds_samples = df[df['job_role'] == 'Data Scientist'].head(2)
print("\nData Scientist samples:")
for idx, row in ds_samples.iterrows():
    print(f"\n- {row['job_role']}")
    skills_str = str(row['skills'])[:80]
    print(f"  Skills: {skills_str}...")
    text_str = str(row['raw_text'])[:100]
    print(f"  Text: {text_str}...")

# Show a Computer Vision Engineer resume  
cv_samples = df[df['job_role'] == 'Computer Vision Engineer'].head(2)
print("\n\nComputer Vision Engineer samples:")
for idx, row in cv_samples.iterrows():
    print(f"\n- {row['job_role']}")
    skills_str = str(row['skills'])[:80]
    print(f"  Skills: {skills_str}...")
    text_str = str(row['raw_text'])[:100]
    print(f"  Text: {text_str}...")
