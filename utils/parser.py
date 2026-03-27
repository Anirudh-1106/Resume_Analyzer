"""
utils/parser.py
Handles PDF text extraction and raw resume field parsing.
"""

import io
import logging
import re
import warnings

import pdfplumber

# Suppress noisy pdfminer warnings from malformed font descriptors.
logging.getLogger("pdfminer").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*FontBBox.*")


SKILLS_LIST = [
    # Languages
    "python", "java", "c++", "c#", "c", "javascript", "typescript",
    "r", "scala", "go", "rust", "php", "swift", "kotlin",
    "dart", "matlab", "sql",

    # AI / ML / DL
    "machine learning", "deep learning", "nlp", "computer vision",
    "artificial intelligence", "ai", "data science",
    "tensorflow", "pytorch", "keras", "scikit-learn", "xgboost",
    "lightgbm", "hugging face", "transformers", "langchain",
    "openai api", "stable diffusion", "onnx", "mlflow",
    "generative ai", "llm", "rag", "lora", "rlhf",
    "spacy", "opencv", "yolo", "cuda",
    "object detection", "image segmentation", "text classification",
    "named entity recognition",

    # Data & Analytics
    "pandas", "numpy", "spark", "hadoop", "kafka", "airflow",
    "dbt", "snowflake", "databricks", "delta lake", "excel",
    "tableau", "power bi", "matplotlib", "seaborn", "plotly",
    "statistics", "data analysis",

    # CS Fundamentals
    "data structures", "algorithms", "dsa", "operating systems",
    "computer networks", "oops", "system design",

    # Databases
    "mysql", "postgresql", "mongodb", "redis", "elasticsearch",
    "sqlite", "oracle", "cassandra", "dynamodb",
    "database management",

    # Web / Frontend
    "html", "css", "react", "next.js", "vue.js", "angular",
    "tailwind", "webpack", "vite", "graphql", "rest api",
    "node.js", "django", "flask", "fastapi", "spring boot",
    "websockets", "redux", "grpc", "microservices",
    "mern", "mean", "web development", "full stack",
    "express.js", "express",

    # Cloud / DevOps
    "aws", "azure", "gcp", "docker", "kubernetes", "terraform",
    "ansible", "helm", "argocd", "prometheus", "grafana",
    "ci/cd", "github actions", "jenkins", "linux", "git", "bash",
    "lambda", "s3", "networking",

    # Mobile
    "flutter", "react native", "android", "ios", "swiftui",
    "jetpack compose", "firebase",

    # UI/UX & Design
    "figma", "adobe xd", "wireframing", "prototyping",
    "user research", "design systems", "usability testing",
    "framer",

    # Cybersecurity
    "penetration testing", "ethical hacking", "owasp", "siem",
    "metasploit", "zero trust", "devsecops",
    "cryptography", "wireshark", "kali linux",

    # Soft skills
    "communication", "leadership", "teamwork", "problem solving",
    "agile", "jira", "critical thinking", "analytical skills",
    "time management", "project management",
]

# Aliases and variants mapped to canonical skill names.
ALIAS_TO_CANONICAL = {
    "nodejs": "node.js",
    "node js": "node.js",
    "reactjs": "react",
    "react js": "react",
    "expressjs": "express.js",
    "express js": "express.js",
    "mongodb atlas": "mongodb",
    "postgres": "postgresql",
    "postgre sql": "postgresql",
    "aiml": "machine learning",
    "ai ml": "machine learning",
    "ai/ml": "machine learning",
    "ml": "machine learning",
    "dl": "deep learning",
    "nlp engineering": "nlp",
    "fullstack": "full stack",
    "full-stack": "full stack",
    "frontend": "frontend",
    "backend": "backend",
    "ui ux": "ui/ux",
    "ux ui": "ui/ux",
    "ci cd": "ci/cd",
    "ci-cd": "ci/cd",
    "dev sec ops": "devsecops",
    "l l m": "llm",
    "open ai api": "openai api",
    "hf": "hugging face",
}

# Keep these tokens available for regex boundaries.
_SPECIAL_CANONICAL = {
    "ci/cd": "ci cd",
    "c++": "c plus plus",
    "c#": "c sharp",
    "node.js": "node js",
    "express.js": "express js",
}


def _normalize_text_for_match(text: str) -> str:
    txt = (text or "").lower()

    # Common merged forms before punctuation stripping.
    merged_fix = {
        "nodejs": "node js",
        "reactjs": "react js",
        "expressjs": "express js",
        "fullstack": "full stack",
        "aiml": "ai ml",
        "ai/ml": "ai ml",
        "ci/cd": "ci cd",
        "ci-cd": "ci cd",
        "c++": "c plus plus",
        "c#": "c sharp",
    }
    for old, new in merged_fix.items():
        txt = txt.replace(old, new)

    # Remove punctuation impact for phrase matching.
    txt = re.sub(r"[^a-z0-9]+", " ", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt


def _term_present(normalized_text: str, term: str) -> bool:
    t = _normalize_text_for_match(term)
    if not t:
        return False
    return re.search(rf"(?<!\w){re.escape(t)}(?!\w)", normalized_text) is not None


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract all text from a PDF given its raw bytes."""
    text = ""
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception:
        text = ""
    return text


def extract_skills(text: str) -> list[str]:
    """Return deduplicated canonical skills found in resume text."""
    norm_text = _normalize_text_for_match(text)

    # Preserve deterministic order from SKILLS_LIST.
    found = []
    found_set = set()

    for skill in SKILLS_LIST:
        searchable = _SPECIAL_CANONICAL.get(skill, skill)
        if _term_present(norm_text, searchable) and skill not in found_set:
            found.append(skill)
            found_set.add(skill)

    for alias, canonical in ALIAS_TO_CANONICAL.items():
        if canonical in found_set:
            continue
        alias_search = _SPECIAL_CANONICAL.get(alias, alias)
        if _term_present(norm_text, alias_search):
            if canonical in SKILLS_LIST:
                found.append(canonical)
                found_set.add(canonical)

    # Prefer specific forms when both generic and specific variants match.
    suppress_if_specific_present = {
        "express": "express.js",
    }
    for generic, specific in suppress_if_specific_present.items():
        if specific in found_set and generic in found_set:
            found = [s for s in found if s != generic]
            found_set.discard(generic)

    return found


def extract_skills_section(text: str) -> str | None:
    """Extract explicit skills section text, if present.

    Captures content after common skill headings until the next section heading.
    """
    if not text:
        return None

    pattern = re.compile(
        r"(?:^|\n)\s*(?:technical\s+skills|skills|core\s+skills|key\s+skills)\s*[:\-]?\s*\n?(.*?)"
        r"(?=\n\s*(?:projects?|experience|education|certifications?|internships?|summary|profile|achievements?)\b|\Z)",
        flags=re.IGNORECASE | re.DOTALL,
    )
    m = pattern.search(text)
    if not m:
        return None

    section = m.group(1).strip()
    return section if section else None


def extract_projects(text: str) -> str | None:
    """Return the projects section snippet, or None if not found."""
    match = re.findall(
        r"(project[s]?:?.*?)(?=\n[A-Z]|$)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return match[0][:300].strip() if match else None


def extract_certifications(text: str) -> str | None:
    """Return the certifications section snippet, or None if not found."""
    match = re.findall(
        r"(certification[s]?:?.*?)(?=\n[A-Z]|$)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return match[0][:200].strip() if match else None


def extract_internships(text: str) -> str | None:
    """Return the internship section snippet, or None if not found."""
    match = re.findall(
        r"(internship[s]?:?.*?)(?=\n[A-Z]|$)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return match[0][:200].strip() if match else None


def parse_resume(file_bytes: bytes) -> dict:
    """
    Full pipeline: bytes -> structured dict with parsed fields.

    Returns keys:
        raw_text, skills (list), projects, certifications, internships
    """
    text = extract_text_from_pdf(file_bytes)
    skills_section_text = extract_skills_section(text)

    # Primary skills used for prediction: explicit Skills section when available,
    # otherwise fallback to whole-resume extraction.
    explicit_skills = extract_skills(skills_section_text) if skills_section_text else []
    all_skills = extract_skills(text)

    if explicit_skills:
        primary_skills = explicit_skills
        inferred_skills = [s for s in all_skills if s not in set(explicit_skills)]
    else:
        primary_skills = all_skills
        inferred_skills = []

    return {
        "raw_text": text,
        "skills": primary_skills,
        "skills_explicit": explicit_skills,
        "skills_inferred": inferred_skills,
        "skills_section_found": bool(skills_section_text),
        "projects": extract_projects(text),
        "certifications": extract_certifications(text),
        "internships": extract_internships(text),
    }
