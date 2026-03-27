"""
utils/role_recommender.py
Skill-based role diagnostics and recommendations.
Used as guardrail/explainability alongside ML predictions.
"""

from __future__ import annotations

from math import ceil


ROLE_PROFILES: dict[str, dict] = {
    "Data Scientist": {
        "required": ["python", "machine learning", "deep learning", "pandas", "numpy", "scikit-learn", "statistics", "data analysis"],
        "preferred": ["tensorflow", "pytorch", "keras", "r", "tableau", "sql", "spark", "mlflow", "xgboost", "plotly"],
        "description": "Builds ML models and extracts actionable insights from data.",
    },
    "Machine Learning Engineer": {
        "required": ["python", "machine learning", "deep learning", "tensorflow", "pytorch", "scikit-learn", "docker", "mlflow"],
        "preferred": ["kubernetes", "aws", "kafka", "airflow", "onnx", "fastapi", "redis", "xgboost", "ci/cd"],
        "description": "Builds and deploys production-grade ML systems at scale.",
    },
    "AI Engineer": {
        "required": ["python", "deep learning", "nlp", "transformers", "pytorch", "langchain", "openai api", "hugging face", "llm"],
        "preferred": ["rag", "stable diffusion", "lora", "rlhf", "generative ai", "fastapi", "docker", "aws", "onnx", "computer vision"],
        "description": "Builds LLM-powered applications, RAG pipelines, and AI agents.",
    },
    "NLP Engineer": {
        "required": ["python", "nlp", "transformers", "pytorch", "hugging face", "deep learning", "machine learning"],
        "preferred": ["langchain", "openai api", "llm", "spacy", "tensorflow", "text classification", "named entity recognition", "docker", "aws"],
        "description": "Specialises in natural language processing and text understanding systems.",
    },
    "Computer Vision Engineer": {
        "required": ["python", "computer vision", "deep learning", "pytorch", "tensorflow", "opencv", "numpy"],
        "preferred": ["yolo", "image segmentation", "object detection", "aws", "onnx", "mlflow", "cuda", "docker", "scikit-learn"],
        "description": "Builds vision systems for object detection, segmentation, and image generation.",
    },
    "Data Engineer": {
        "required": ["python", "sql", "spark", "kafka", "airflow", "docker", "postgresql", "aws"],
        "preferred": ["dbt", "snowflake", "databricks", "delta lake", "kubernetes", "terraform", "redis", "elasticsearch", "mongodb", "scala"],
        "description": "Builds scalable data pipelines and data lakehouse architectures.",
    },
    "Business Analyst": {
        "required": ["sql", "excel", "tableau", "data analysis", "communication", "leadership"],
        "preferred": ["power bi", "python", "pandas", "statistics", "r", "agile", "jira", "plotly", "user research"],
        "description": "Bridges business requirements with data-driven technical solutions.",
    },
    "Software Engineer": {
        "required": ["python", "java", "c++", "git", "sql", "rest api", "docker"],
        "preferred": ["kubernetes", "aws", "redis", "mongodb", "go", "spring boot", "fastapi", "django", "microservices", "kafka"],
        "description": "Designs and implements scalable software systems and microservices.",
    },
    "Frontend Developer": {
        "required": ["javascript", "typescript", "react", "html", "css", "next.js", "git", "tailwind"],
        "preferred": ["vue.js", "angular", "graphql", "vite", "webpack", "figma", "rest api", "websockets", "redux"],
        "description": "Builds performant, accessible user interfaces using modern JS frameworks.",
    },
    "Full Stack Developer": {
        "required": ["javascript", "typescript", "react", "node.js", "python", "sql", "mongodb", "docker", "rest api"],
        "preferred": ["next.js", "graphql", "aws", "redis", "kubernetes", "fastapi", "postgresql", "tailwind", "websockets", "kafka"],
        "description": "Develops end-to-end web applications across frontend and backend.",
    },
    "Backend Developer": {
        "required": ["python", "java", "sql", "rest api", "docker", "postgresql", "redis", "linux", "git"],
        "preferred": ["go", "rust", "kafka", "kubernetes", "aws", "mongodb", "elasticsearch", "spring boot", "fastapi", "grpc"],
        "description": "Designs high-performance server-side APIs and distributed systems.",
    },
    "DevOps Engineer": {
        "required": ["docker", "kubernetes", "aws", "terraform", "linux", "git", "ci/cd", "python", "bash"],
        "preferred": ["azure", "gcp", "helm", "argocd", "prometheus", "grafana", "ansible", "github actions", "jenkins"],
        "description": "Builds CI/CD pipelines and manages cloud infrastructure at scale.",
    },
    "Cloud Engineer": {
        "required": ["aws", "azure", "terraform", "docker", "kubernetes", "linux", "python", "git"],
        "preferred": ["gcp", "ansible", "helm", "prometheus", "grafana", "lambda", "s3", "ci/cd", "bash", "networking"],
        "description": "Architects and manages scalable multi-cloud infrastructure.",
    },
    "Cybersecurity Engineer": {
        "required": ["python", "linux", "penetration testing", "ethical hacking", "owasp", "siem", "bash"],
        "preferred": ["metasploit", "zero trust", "devsecops", "docker", "aws", "cryptography", "networking", "wireshark", "kali linux"],
        "description": "Secures applications and infrastructure against cyber threats.",
    },
    "Mobile App Developer": {
        "required": ["flutter", "dart", "react native", "javascript", "firebase", "android", "ios", "git"],
        "preferred": ["swift", "kotlin", "swiftui", "jetpack compose", "sql", "rest api", "aws", "typescript", "redux"],
        "description": "Builds cross-platform and native mobile apps for iOS and Android.",
    },
    "UI/UX Designer": {
        "required": ["figma", "adobe xd", "wireframing", "prototyping", "user research", "design systems", "usability testing"],
        "preferred": ["html", "css", "javascript", "react", "framer", "data analysis", "statistics", "communication", "leadership"],
        "description": "Creates user-centred designs through research, wireframing, and prototyping.",
    },
}

_DEFAULT_ROLE = "Software Engineer"

DEFAULT_MATCH_THRESHOLDS = {
    "min_required_coverage": 0.30,
    "min_required_hits": 2,
    "min_overall_coverage": 0.20,
    "text_hit_weight_required": 0.40,
    "text_hit_weight_preferred": 0.30,
}

ROLE_TOP_SKILLS: dict[str, list[str]] = {
    role: (data["required"] + data["preferred"])[:10]
    for role, data in ROLE_PROFILES.items()
}


def _normalize_skills(skills: list[str]) -> set[str]:
    return {str(s).strip().lower() for s in (skills or []) if str(s).strip()}


def role_match_diagnostics(
    role: str,
    skills: list[str],
    raw_text: str = "",
    thresholds: dict | None = None,
) -> dict:
    """Return detailed match diagnostics for a role."""
    if role not in ROLE_PROFILES:
        return {
            "role": role,
            "eligible": False,
            "required_coverage": 0.0,
            "preferred_coverage": 0.0,
            "overall_coverage": 0.0,
            "required_hits": [],
            "preferred_hits": [],
            "required_text_hits": [],
            "preferred_text_hits": [],
            "required_effective_hits": 0.0,
            "preferred_effective_hits": 0.0,
            "min_required_hits_needed": 0,
            "matched_count": 0,
            "score": 0.0,
            "confidence": 0.0,
            "description": "",
        }

    cfg = {**DEFAULT_MATCH_THRESHOLDS, **(thresholds or {})}
    profile = ROLE_PROFILES[role]
    required = [s.lower() for s in profile["required"]]
    preferred = [s.lower() for s in profile["preferred"]]

    skill_set = _normalize_skills(skills)
    text_lower = (raw_text or "").lower()

    req_hits = [s for s in required if s in skill_set]
    pref_hits = [s for s in preferred if s in skill_set]
    req_text = [s for s in required if s not in skill_set and s in text_lower]
    pref_text = [s for s in preferred if s not in skill_set and s in text_lower]

    req_total = max(len(required), 1)
    pref_total = max(len(preferred), 1)
    all_total = req_total + pref_total

    req_effective = len(req_hits) + cfg["text_hit_weight_required"] * len(req_text)
    pref_effective = len(pref_hits) + cfg["text_hit_weight_preferred"] * len(pref_text)

    req_coverage = req_effective / req_total
    pref_coverage = pref_effective / pref_total
    overall_coverage = (req_effective + pref_effective) / all_total

    min_required_dynamic = max(
        int(cfg["min_required_hits"]),
        ceil(cfg["min_required_coverage"] * req_total),
    )

    eligible = (
        req_effective >= min_required_dynamic
        and req_coverage >= cfg["min_required_coverage"]
        and overall_coverage >= cfg["min_overall_coverage"]
    )

    confidence = (req_coverage * 0.75 + pref_coverage * 0.25) * 100
    if not eligible:
        confidence *= 0.2

    score = confidence

    return {
        "role": role,
        "eligible": bool(eligible),
        "required_coverage": round(req_coverage * 100, 1),
        "preferred_coverage": round(pref_coverage * 100, 1),
        "overall_coverage": round(overall_coverage * 100, 1),
        "required_hits": req_hits,
        "preferred_hits": pref_hits,
        "required_text_hits": req_text,
        "preferred_text_hits": pref_text,
        "required_effective_hits": round(req_effective, 2),
        "preferred_effective_hits": round(pref_effective, 2),
        "min_required_hits_needed": int(min_required_dynamic),
        "matched_count": len(set(req_hits + pref_hits + req_text + pref_text)),
        "score": round(score, 3),
        "confidence": round(min(max(confidence, 0.0), 100.0), 1),
        "description": profile["description"],
    }


def recommend_roles(
    skills: list[str],
    raw_text: str = "",
    top_n: int = 5,
    thresholds: dict | None = None,
) -> list[dict]:
    """Recommend top roles by strict coverage-based diagnostics."""
    diagnostics = [
        role_match_diagnostics(role, skills, raw_text, thresholds)
        for role in ROLE_PROFILES
    ]

    diagnostics.sort(key=lambda x: x["score"], reverse=True)

    meaningful = [d for d in diagnostics if d["eligible"] and d["confidence"] >= 20]
    if meaningful:
        return [
            {
                "role": d["role"],
                "score": d["score"],
                "confidence": d["confidence"],
                "description": d["description"],
                "matched": sorted(set(d["required_hits"] + d["preferred_hits"])),
            }
            for d in meaningful[:top_n]
        ]

    # Fallback for sparse resumes (still low confidence).
    fallback = diagnostics[0] if diagnostics else {
        "role": _DEFAULT_ROLE,
        "description": ROLE_PROFILES[_DEFAULT_ROLE]["description"],
    }

    return [{
        "role": fallback.get("role", _DEFAULT_ROLE),
        "score": float(fallback.get("score", 0.0)),
        "confidence": float(min(fallback.get("confidence", 0.0), 15.0)),
        "description": fallback.get("description", ROLE_PROFILES[_DEFAULT_ROLE]["description"]),
        "matched": sorted(set(fallback.get("required_hits", []) + fallback.get("preferred_hits", []))),
    }]


def top_role(skills: list[str], raw_text: str = "", thresholds: dict | None = None) -> dict:
    return recommend_roles(skills, raw_text, top_n=1, thresholds=thresholds)[0]


def skill_gap_for_role(role: str, user_skills: list[str]) -> dict:
    if role not in ROLE_PROFILES:
        return {"matched": [], "missing": [], "match_pct": 0.0}

    profile = ROLE_PROFILES[role]
    target = set(s.lower() for s in profile["required"] + profile["preferred"])
    user_set = _normalize_skills(user_skills)

    matched = sorted(target & user_set)
    missing = sorted(target - user_set)
    match_pct = len(matched) / len(target) * 100 if target else 0.0

    return {
        "matched": matched,
        "missing": missing,
        "match_pct": round(match_pct, 1),
    }
