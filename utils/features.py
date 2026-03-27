"""
utils/features.py
Converts a parsed resume dict into a numeric feature vector
that matches the schema used during model training.
"""

import re
import ast
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Skill category taxonomy (used for Home page EDA)
# ---------------------------------------------------------------------------
SKILL_CATEGORIES = {
    "Programming":     ["python", "java", "c++", "javascript", "c#", "php",
                        "swift", "kotlin", "scala", "r", "matlab"],
    "Data Science":    ["machine learning", "deep learning", "nlp", "tableau",
                        "tensorflow", "keras", "pytorch", "pandas", "numpy",
                        "scikit-learn", "spark", "hadoop"],
    "Web Development": ["html", "css", "javascript", "react", "angular",
                        "node.js", "django", "flask", "php"],
    "Core CS":         ["c++", "java", "sql", "linux", "git", "mysql",
                        "postgresql", "mongodb", "c#"],
    "Cloud / DevOps":  ["aws", "docker", "kubernetes"],
}
ALL_CATEGORIES = list(SKILL_CATEGORIES.keys())

# ---------------------------------------------------------------------------
# Feature columns — must stay in sync with train_models.py FEATURES list
# ---------------------------------------------------------------------------
FEATURE_COLUMNS = [
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


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_parse_list(val) -> list:
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


def _keyword_flag(text: str, keywords: list[str]) -> int:
    if not text:
        return 0
    text_lower = str(text).lower()
    return 1 if any(k in text_lower for k in keywords) else 0


def _skill_diversity(skills: list[str]) -> float:
    if not skills:
        return 0.0
    return len(set(s.lower() for s in skills)) / len(skills)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_skill_categories(skills: list[str]) -> dict[str, list[str]]:
    """Map a list of user skills to their category buckets."""
    result = {cat: [] for cat in ALL_CATEGORIES}
    for skill in skills:
        for cat, members in SKILL_CATEGORIES.items():
            if skill.lower() in [m.lower() for m in members]:
                result[cat].append(skill)
    return result


def extract_features(parsed: dict) -> pd.DataFrame:
    """
    Convert a parsed resume dict → single-row DataFrame matching FEATURE_COLUMNS.

    Parameters
    ----------
    parsed : dict  –  Output of utils.parser.parse_resume().

    Returns
    -------
    pd.DataFrame with columns matching FEATURE_COLUMNS.
    """
    skills      = parsed.get("skills") or []
    projects    = parsed.get("projects")
    certs       = parsed.get("certifications")
    internships = parsed.get("internships")
    raw_text    = parsed.get("raw_text") or ""

    skills_list = skills if isinstance(skills, list) else _safe_parse_list(skills)
    proj_list   = _safe_parse_list(projects)
    cert_list   = _safe_parse_list(certs)
    intern_list = _safe_parse_list(internships)

    row = {
        "num_skills":           len(skills_list),
        "num_projects":         len(proj_list),
        "certifications_count": len(cert_list),
        "internship_flag":      1 if intern_list else 0,
        "skill_diversity":      _skill_diversity(skills_list),
        "has_python":           _keyword_flag(raw_text, ["python"]),
        "has_ml":               _keyword_flag(raw_text, ["machine learning", "deep learning", "tensorflow", "pytorch", "xgboost"]),
        "has_web":              _keyword_flag(raw_text, ["html", "css", "javascript", "react", "next.js", "vue", "angular", "tailwind"]),
        "has_cloud":            _keyword_flag(raw_text, ["aws", "azure", "gcp", "cloud", "terraform", "kubernetes"]),
        "has_database":         _keyword_flag(raw_text, ["sql", "mysql", "mongodb", "postgresql", "database", "snowflake", "redis"]),
        "has_nlp":              _keyword_flag(raw_text, ["nlp", "natural language", "transformers", "bert", "langchain", "llm", "hugging face", "spacy"]),
        "has_devops":           _keyword_flag(raw_text, ["docker", "kubernetes", "ci/cd", "devops", "ansible", "terraform", "jenkins", "github actions"]),
        "has_mobile":           _keyword_flag(raw_text, ["flutter", "react native", "android", "ios", "swift", "kotlin", "swiftui", "jetpack"]),
        "has_security":         _keyword_flag(raw_text, ["penetration testing", "ethical hacking", "owasp", "siem", "metasploit", "cybersecurity", "zero trust"]),
        "has_design":           _keyword_flag(raw_text, ["figma", "adobe xd", "wireframing", "prototyping", "user research", "ux", "ui design", "usability"]),
    }

    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def category_distribution(all_skills_flat: list[str]) -> dict[str, float]:
    """Return % distribution of skills across the five categories."""
    counts = {cat: 0 for cat in ALL_CATEGORIES}
    for skill in all_skills_flat:
        for cat, members in SKILL_CATEGORIES.items():
            if skill.strip().lower() in [m.lower() for m in members]:
                counts[cat] += 1
    total = sum(counts.values()) or 1
    return {cat: (v / total) * 100 for cat, v in counts.items()}

