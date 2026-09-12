"""
Job matching engine for the RAG-Based Profile Matching project.

Features:
- JD embedding
- top-K semantic retrieval
- critical skill keyword matching
- must-have requirement extraction/filtering
- 0-100 scoring
- matched skills, excerpts and reasoning
- JSON output matching the assignment specification

Usage:
    python profile_matcher.py --job-file job_descriptions/jd_01_ml_engineer.txt
"""
from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from config import (
    CHROMA_COLLECTION,
    CHROMA_PERSIST_DIR,
    DEFAULT_TOP_K,
    KEYWORD_WEIGHT,
    MUST_HAVE_WEIGHT,
    SEMANTIC_WEIGHT,
)
from rag_profile_matching import EmbeddingProvider, COMMON_SKILLS


def normalize_skill(s: str) -> str:
    return re.sub(r"[^a-z0-9+#.]+", " ", s.lower()).strip()


def extract_job_skills(job_text: str) -> list[str]:
    lower = job_text.lower()
    return [skill for skill in COMMON_SKILLS
            if re.search(r"(?<!\w)" + re.escape(skill.lower()) + r"(?!\w)", lower)]


def extract_must_haves(job_text: str) -> list[dict]:
    """
    Extract hard requirements from common natural-language patterns.

    Examples:
      "5+ years Python"
      "minimum 3 years of Java"
      "must have Kubernetes"
      "required: AWS"
    """
    reqs = []

    for years, skill in re.findall(
        r"(\d+(?:\.\d+)?)\s*\+?\s*years?\s+(?:of\s+)?(?:experience\s+)?(?:with\s+)?([A-Za-z0-9+#./ -]+?)(?=[,.;:\n]|$)",
        job_text, flags=re.I
    ):
        candidate = skill.strip()
        for known in COMMON_SKILLS:
            if normalize_skill(known) == normalize_skill(candidate) or normalize_skill(known) in normalize_skill(candidate):
                reqs.append({"type": "years", "skill": known, "years": float(years)})
                break

    # "minimum 5 years of Python"
    for years, skill in re.findall(
        r"(?:minimum|at least)\s+(\d+(?:\.\d+)?)\s*years?\s+(?:of\s+)?([A-Za-z0-9+#./ -]+?)(?=[,.;:\n]|$)",
        job_text, flags=re.I
    ):
        for known in COMMON_SKILLS:
            if normalize_skill(known) in normalize_skill(skill):
                reqs.append({"type": "years", "skill": known, "years": float(years)})

    # Explicit required/must-have skill phrases
    for skill in COMMON_SKILLS:
        if re.search(
            r"(?:must\s+have|required|mandatory|essential|must-have)[^.\n]{0,80}"
            + re.escape(skill),
            job_text,
            flags=re.I,
        ):
            reqs.append({"type": "skill", "skill": skill})

    # Deduplicate
    unique = []
    seen = set()
    for req in reqs:
        key = (req["type"], req["skill"], req.get("years"))
        if key not in seen:
            unique.append(req)
            seen.add(key)
    return unique


def cosine_similarity(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0


class ProfileMatcher:
    def __init__(
        self,
        persist_dir: str = CHROMA_PERSIST_DIR,
        collection_name: str = CHROMA_COLLECTION,
        embedding_provider: str = "hf",
    ):
        import chromadb
        self.embedder = EmbeddingProvider(embedding_provider)
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(collection_name)

    def _retrieve(self, job_text: str, n: int = 40) -> list[dict]:
        vector = self.embedder.encode([job_text])[0]
        result = self.collection.query(
            query_embeddings=[vector],
            n_results=min(n, max(self.collection.count(), 1)),
            include=["documents", "metadatas", "distances", "embeddings"],
        )

        rows = []
        for i, doc in enumerate(result["documents"][0]):
            meta = result["metadatas"][0][i]
            # Chroma cosine distance = 1 - cosine similarity for normalized vectors.
            semantic = max(0.0, min(1.0, 1.0 - float(result["distances"][0][i])))
            rows.append({
                "document": doc,
                "metadata": meta,
                "semantic": semantic,
            })
        return rows

    @staticmethod
    def _group_by_candidate(rows: list[dict]) -> dict[str, list[dict]]:
        grouped = defaultdict(list)
        for row in rows:
            grouped[row["metadata"]["resume_path"]].append(row)
        return grouped

    @staticmethod
    def _candidate_skills(meta: dict) -> list[str]:
        try:
            return json.loads(meta.get("skills", "[]"))
        except Exception:
            return []

    def _keyword_score(self, job_skills: list[str], candidate_skills: list[str]) -> float:
        if not job_skills:
            return 0.0
        a = {normalize_skill(x) for x in job_skills}
        b = {normalize_skill(x) for x in candidate_skills}
        return len(a & b) / len(a)

    def _must_have_status(
        self,
        requirements: list[dict],
        meta: dict,
        candidate_skills: list[str],
    ) -> tuple[bool, list[str], list[str]]:
        experience = float(meta.get("experience_years") or 0)
        skill_set = {normalize_skill(x) for x in candidate_skills}
        failures, satisfied = [], []

        for req in requirements:
            skill = req["skill"]
            if req["type"] == "skill":
                if normalize_skill(skill) in skill_set:
                    satisfied.append(skill)
                else:
                    failures.append(f"Missing required skill: {skill}")
            else:
                has_skill = normalize_skill(skill) in skill_set
                has_years = experience >= float(req["years"])
                if has_skill and has_years:
                    satisfied.append(f"{skill} ({req['years']}+ years)")
                else:
                    failures.append(
                        f"Requires {req['years']}+ years {skill}; "
                        f"candidate has {experience:g} years and "
                        f"{'has' if has_skill else 'does not have'} the skill"
                    )
        return not failures, satisfied, failures

    def match(self, job_text: str, top_k: int = DEFAULT_TOP_K) -> dict:
        start = time.perf_counter()
        job_skills = extract_job_skills(job_text)
        requirements = extract_must_haves(job_text)
        rows = self._retrieve(job_text, n=max(40, top_k * 4))
        grouped = self._group_by_candidate(rows)

        scored = []
        for path, candidate_rows in grouped.items():
            meta = candidate_rows[0]["metadata"]
            candidate_skills = self._candidate_skills(meta)

            semantic = max(r["semantic"] for r in candidate_rows)
            keyword = self._keyword_score(job_skills, candidate_skills)
            passes, satisfied, failures = self._must_have_status(
                requirements, meta, candidate_skills
            )
            must_have = 1.0 if not requirements else len(satisfied) / len(requirements)

            raw_score = 100 * (
                SEMANTIC_WEIGHT * semantic
                + KEYWORD_WEIGHT * keyword
                + MUST_HAVE_WEIGHT * must_have
            )

            # Hard filter: the assignment explicitly asks to filter must-have requirements.
            if requirements and not passes:
                continue

            matched_skills = [
                skill for skill in job_skills
                if normalize_skill(skill) in {normalize_skill(x) for x in candidate_skills}
            ]

            # Keep the strongest chunks and preserve their sections.
            strongest = sorted(candidate_rows, key=lambda x: x["semantic"], reverse=True)[:3]
            sections = [r["metadata"].get("section", "other") for r in strongest]
            excerpts = [r["document"][:500] for r in strongest]

            reasoning = (
                f"Strong semantic alignment ({semantic:.2f}) with "
                f"{len(matched_skills)}/{max(len(job_skills), 1)} detected job skills. "
                f"Relevant evidence comes from: {', '.join(dict.fromkeys(sections))}. "
                f"Must-have requirements satisfied: "
                f"{len(satisfied)}/{max(len(requirements), 1)}."
            )

            scored.append({
                "candidate_name": meta.get("candidate_name", "Unknown"),
                "resume_path": path,
                "match_score": round(max(0.0, min(100.0, raw_score)), 2),
                "matched_skills": matched_skills,
                "relevant_excerpts": excerpts,
                "reasoning": reasoning,
                "_semantic": semantic,
                "_keyword": keyword,
                "_must_have": must_have,
                "_failures": failures,
            })

        scored.sort(key=lambda x: x["match_score"], reverse=True)
        elapsed_ms = (time.perf_counter() - start) * 1000

        for item in scored:
            for key in ("_semantic", "_keyword", "_must_have", "_failures"):
                item.pop(key, None)

        return {
            "job_description": job_text,
            "top_matches": scored[:top_k],
            "latency_ms": round(elapsed_ms, 2),
        }


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--job-file")
    group.add_argument("--job-text")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--persist-dir", default=CHROMA_PERSIST_DIR)
    parser.add_argument("--provider", default="hf", choices=["hf", "openai"])
    args = parser.parse_args()

    job_text = (
        Path(args.job_file).read_text(encoding="utf-8")
        if args.job_file
        else args.job_text
    )

    matcher = ProfileMatcher(
        persist_dir=args.persist_dir,
        embedding_provider=args.provider,
    )
    result = matcher.match(job_text, top_k=args.top_k)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
