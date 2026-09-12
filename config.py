"""Configuration for the RAG-Based Profile Matching project."""
import os
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "hf").lower()
HF_EMBEDDING_MODEL = os.getenv(
    "HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
OPENAI_EMBEDDING_MODEL = os.getenv(
    "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
)
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "resume_chunks")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "chroma_db")

SEMANTIC_WEIGHT = 0.60
KEYWORD_WEIGHT = 0.25
MUST_HAVE_WEIGHT = 0.15
DEFAULT_TOP_K = 10

# Sections commonly found in resumes. The matcher reports which of these
# contributed evidence to the final score.
SECTION_ALIASES = {
    "summary": ["summary", "profile", "professional summary", "objective"],
    "skills": ["skills", "technical skills", "core competencies", "technologies"],
    "experience": ["experience", "work experience", "professional experience", "employment"],
    "education": ["education", "academic background", "qualifications"],
    "projects": ["projects", "selected projects", "academic projects"],
    "certifications": ["certifications", "certificates"],
}
