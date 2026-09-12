"""
RAG-Based Profile Matching indexing pipeline.

Responsibilities:
1. Load PDF/TXT/DOCX resumes.
2. Detect resume sections and chunk within sections.
3. Extract candidate metadata.
4. Generate embeddings using HuggingFace or OpenAI.
5. Persist chunks and metadata in ChromaDB.

Usage:
    python rag_profile_matching.py --resumes-dir resumes --persist-dir chroma_db
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from config import (
    CHROMA_COLLECTION,
    CHROMA_PERSIST_DIR,
    EMBEDDING_PROVIDER,
    HF_EMBEDDING_MODEL,
    OPENAI_EMBEDDING_MODEL,
    SECTION_ALIASES,
)


@dataclass
class ResumeMetadata:
    name: str
    skills: list[str]
    experience_years: float
    education: str
    resume_path: str


class EmbeddingProvider:
    def __init__(self, provider: str = EMBEDDING_PROVIDER):
        self.provider = provider.lower()
        self.model = None
        self.client = None

        if self.provider == "hf":
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(HF_EMBEDDING_MODEL)
        elif self.provider == "openai":
            from openai import OpenAI
            self.client = OpenAI()
        else:
            raise ValueError("EMBEDDING_PROVIDER must be 'hf' or 'openai'.")

    def encode(self, texts: list[str]) -> list[list[float]]:
        if self.provider == "hf":
            vectors = self.model.encode(
                texts, normalize_embeddings=True, show_progress_bar=False
            )
            return np.asarray(vectors, dtype=np.float32).tolist()

        response = self.client.embeddings.create(
            model=OPENAI_EMBEDDING_MODEL,
            input=texts,
        )
        return [item.embedding for item in response.data]


def load_document(path: Path) -> str:
    """Load TXT, PDF or DOCX. Raises a clear error for unsupported formats."""
    suffix = path.suffix.lower()

    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="ignore")

    if suffix == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    if suffix == ".docx":
        from docx import Document
        doc = Document(str(path))
        return "\n".join(p.text for p in doc.paragraphs)

    raise ValueError(f"Unsupported resume format: {path}")


def iter_resume_files(resumes_dir: str | Path) -> Iterable[Path]:
    root = Path(resumes_dir)
    for pattern in ("*.pdf", "*.txt", "*.md", "*.docx"):
        yield from sorted(root.rglob(pattern))


def _normalize_line(line: str) -> str:
    return re.sub(r"\s+", " ", line.strip())


def detect_sections(text: str) -> dict[str, str]:
    """
    Split a resume into semantic sections. Heading detection intentionally
    supports common formatting variations without depending on exact casing.
    """
    lines = [_normalize_line(x) for x in text.splitlines() if _normalize_line(x)]
    heading_to_section = {}
    for section, aliases in SECTION_ALIASES.items():
        for alias in aliases:
            heading_to_section[re.sub(r"[^a-z0-9]", "", alias.lower())] = section

    sections: dict[str, list[str]] = {"other": []}
    current = "other"

    for line in lines:
        key = re.sub(r"[^a-z0-9]", "", line.lower().rstrip(":"))
        if key in heading_to_section:
            current = heading_to_section[key]
            sections.setdefault(current, [])
            continue
        sections.setdefault(current, []).append(line)

    return {k: "\n".join(v).strip() for k, v in sections.items() if "\n".join(v).strip()}


def chunk_sections(
    sections: dict[str, str],
    chunk_size: int = 900,
    overlap: int = 120,
) -> list[dict]:
    """
    Chunk by section first, then by words. This keeps Education/Experience/etc.
    together and records the source section in metadata.
    """
    chunks = []
    for section, text in sections.items():
        words = text.split()
        if not words:
            continue

        start = 0
        chunk_index = 0
        while start < len(words):
            end = min(len(words), start + chunk_size)
            chunk_text = " ".join(words[start:end])
            chunks.append({
                "section": section,
                "chunk_index": chunk_index,
                "text": chunk_text,
            })
            if end == len(words):
                break
            start = max(end - overlap, start + 1)
            chunk_index += 1
    return chunks


COMMON_SKILLS = [
    "Python", "Java", "JavaScript", "TypeScript", "C++", "C#", "SQL",
    "Spring Boot", "Django", "Flask", "FastAPI", "React", "Angular", "Node.js",
    "AWS", "Azure", "GCP", "Docker", "Kubernetes", "Terraform", "Jenkins",
    "Git", "Linux", "PostgreSQL", "MySQL", "MongoDB", "Redis", "Kafka",
    "Spark", "Hadoop", "Airflow", "Databricks", "Snowflake", "Tableau",
    "Power BI", "Machine Learning", "Deep Learning", "NLP", "LLM",
    "PyTorch", "TensorFlow", "scikit-learn", "Pandas", "NumPy",
    "Computer Vision", "Cybersecurity", "REST APIs", "GraphQL",
    "Microservices", "System Design", "Data Engineering", "ETL",
    "CI/CD", "Selenium", "Playwright", "Figma", "Product Management",
]


def extract_name(text: str, path: Path) -> str:
    lines = [_normalize_line(x) for x in text.splitlines() if _normalize_line(x)]
    if lines:
        first = lines[0]
        # Prefer a human-looking first line; otherwise use the filename.
        if 2 <= len(first.split()) <= 5 and not any(c.isdigit() for c in first):
            return first
    return path.stem.replace("_", " ").replace("-", " ").title()


def extract_skills(text: str) -> list[str]:
    lower = text.lower()
    found = []
    for skill in COMMON_SKILLS:
        if re.search(r"(?<!\w)" + re.escape(skill.lower()) + r"(?!\w)", lower):
            found.append(skill)
    return found


def extract_experience_years(text: str) -> float:
    patterns = [
        r"(\d+(?:\.\d+)?)\s*\+?\s*years?\s+(?:of\s+)?(?:professional\s+)?experience",
        r"experience\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*\+?\s*years?",
    ]
    values = []
    for pattern in patterns:
        values.extend(float(x) for x in re.findall(pattern, text, flags=re.I))
    return max(values) if values else 0.0


def extract_education(text: str) -> str:
    sections = detect_sections(text)
    return sections.get("education", "")[:500]


def extract_metadata(text: str, path: Path) -> ResumeMetadata:
    return ResumeMetadata(
        name=extract_name(text, path),
        skills=extract_skills(text),
        experience_years=extract_experience_years(text),
        education=extract_education(text),
        resume_path=str(path),
    )


class ProfileRAGIndexer:
    def __init__(
        self,
        persist_dir: str = CHROMA_PERSIST_DIR,
        collection_name: str = CHROMA_COLLECTION,
        embedding_provider: str = EMBEDDING_PROVIDER,
    ):
        import chromadb

        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.embedder = EmbeddingProvider(embedding_provider)
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def index_documents(self, documents: list[tuple[str, str]]) -> int:
        """
        Index documents as (path, text). This method is the easiest integration
        point for an existing Milestone 1 filesystem loader.
        """
        ids, texts, metadatas = [], [], []

        for path_str, text in documents:
            path = Path(path_str)
            metadata = extract_metadata(text, path)
            sections = detect_sections(text)
            chunks = chunk_sections(sections)

            for chunk in chunks:
                chunk_id = f"{path.as_posix()}::{chunk['section']}::{chunk['chunk_index']}"
                ids.append(chunk_id)
                texts.append(chunk["text"])
                metadatas.append({
                    "resume_path": metadata.resume_path,
                    "candidate_name": metadata.name,
                    "skills": json.dumps(metadata.skills),
                    "experience_years": metadata.experience_years,
                    "education": metadata.education,
                    "section": chunk["section"],
                    "chunk_index": chunk["chunk_index"],
                })

        if not texts:
            return 0

        embeddings = self.embedder.encode(texts)
        self.collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        return len(texts)

    def index_directory(self, resumes_dir: str) -> int:
        docs = []
        for path in iter_resume_files(resumes_dir):
            try:
                text = load_document(path)
                if text.strip():
                    docs.append((str(path), text))
            except Exception as exc:
                print(f"[WARN] Could not load {path}: {exc}")
        return self.index_documents(docs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resumes-dir", default="resumes")
    parser.add_argument("--persist-dir", default=CHROMA_PERSIST_DIR)
    parser.add_argument("--provider", default=EMBEDDING_PROVIDER, choices=["hf", "openai"])
    args = parser.parse_args()

    indexer = ProfileRAGIndexer(
        persist_dir=args.persist_dir,
        embedding_provider=args.provider,
    )
    count = indexer.index_directory(args.resumes_dir)
    print(json.dumps({
        "status": "success",
        "indexed_chunks": count,
        "persist_dir": args.persist_dir,
        "collection": indexer.collection_name,
    }, indent=2))


if __name__ == "__main__":
    main()
