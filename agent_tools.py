"""Reusable tools for the Agentic Profile Matching LangGraph agent.

The tools are deliberately deterministic.  LLMs are not used to make or invent
candidate facts; resume text and Milestone 2 metadata remain the source of truth.
"""
from __future__ import annotations

import re
import zipfile
from pathlib import Path
from typing import Any

DEFAULT_RESUME_ROOT = Path("resumes")
MAX_RESUME_BYTES = 10 * 1024 * 1024  # 10 MiB source-file safety limit per resume
MAX_EXTRACTED_TEXT_CHARS = 5_000_000  # bound parser output / decompression growth
MAX_PDF_PAGES = 50
MAX_DOCX_UNCOMPRESSED_BYTES = 50 * 1024 * 1024  # limit zip expansion before XML parsing

from profile_matcher import extract_job_skills, extract_must_haves, normalize_skill
from rag_profile_matching import COMMON_SKILLS, extract_experience_years, extract_skills


def _known_skills(text: str) -> list[str]:
    found: list[str] = []
    lowered = (text or "").casefold()
    for skill in sorted(COMMON_SKILLS, key=len, reverse=True):
        if re.search(r"(?<!\w)" + re.escape(skill.casefold()) + r"(?!\w)", lowered):
            found.append(skill)
    result: list[str] = []
    seen: set[str] = set()
    for skill in found:
        key = normalize_skill(skill)
        if key not in seen:
            result.append(skill)
            seen.add(key)
    return result


def estimate_experience_years(text: str) -> float:
    """Estimate experience using explicit statements and dated role ranges."""
    explicit = 0.0
    try:
        explicit = float(extract_experience_years(text) or 0.0)
    except Exception:
        explicit = 0.0

    intervals: list[tuple[int, int]] = []
    for start, end in re.findall(
        r"\b((?:19|20)\d{2})\s*[-–—]\s*((?:19|20)\d{2}|present|current)\b",
        text or "",
        flags=re.IGNORECASE,
    ):
        start_year = int(start)
        if end.casefold() in {"present", "current"}:
            from datetime import datetime

            end_year = datetime.now().year
        else:
            end_year = int(end)
        if end_year >= start_year:
            intervals.append((start_year, end_year))

    if not intervals:
        return explicit

    intervals.sort()
    merged: list[list[int]] = []
    for start, end in intervals:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)

    dated_years = sum(end - start for start, end in merged)
    return round(max(explicit, float(dated_years)), 2)


def extract_requirements(jd: str) -> dict[str, Any]:
    """Extract hard requirements and preferences from a JD or recruiter query.

    In addition to Milestone 2's skill-specific requirements, this tool recognizes
    general experience requirements such as ``3+ years experience``.
    """
    text = (jd or "").strip()
    must_haves = list(extract_must_haves(text))

    # Conversational recruiter queries such as "Find candidates with React and
    # 3+ years experience" should treat the explicitly requested skills as hard
    # criteria, not merely as semantic-search hints.
    query_skill_patterns = (
        r"\b(?:candidates?|applicants?|people|professionals?)\s+(?:with|having)\s+([^.;:!?\n]+)",
        r"\b(?:find|show|give|need|want|looking\s+for)\b[^.;:!?\n]{0,40}\bwith\s+([^.;:!?\n]+)",
    )
    query_skill_texts: list[str] = []
    for pattern in query_skill_patterns:
        query_skill_texts.extend(m.group(1) for m in re.finditer(pattern, text, flags=re.IGNORECASE))
    for clause in query_skill_texts:
        for skill in _known_skills(clause):
            must_haves.append({"type": "skill", "skill": skill})

    detected_skills = extract_job_skills(text)

    # Conversational role queries such as "best candidate for Python developer"
    # explicitly name a skill but do not use a "with"/"must-have" phrase. Treat
    # those detected skills as hard criteria so the shortlist is actually filtered.
    conversational_search = bool(re.search(
        r"\b(?:best|top|find|show|give|need|want|looking)\b.*\b(?:candidate|candidates|developer|engineer|applicant|professional|people)\b",
        text, flags=re.IGNORECASE
    ))
    if conversational_search:
        existing_hard = {
            normalize_skill(req.get("skill", ""))
            for req in must_haves
            if req.get("type") in {"skill", "years"}
        }
        for skill in detected_skills:
            if normalize_skill(skill) not in existing_hard:
                must_haves.append({"type": "skill", "skill": skill})
                existing_hard.add(normalize_skill(skill))

    # General experience requirement when no skill is attached to the years phrase.
    existing_experience = {float(req.get("years", 0)) for req in must_haves if req.get("type") == "experience"}
    general_year_patterns = (
        r"(?:at\s+least|minimum|min\.?|more\s+than)?\s*"
        r"(\d+(?:\.\d+)?)\s*\+?\s*years?\s+(?:of\s+)?(?:professional\s+)?experience\b",
        r"(?:experience|exp\.?)[^\d]{0,10}(\d+(?:\.\d+)?)\s*\+?\s*years?\b",
    )
    for pattern in general_year_patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            years = float(match.group(1))
            # If a known skill occurs immediately before the phrase, Milestone 2's
            # years requirement is more specific and should remain authoritative.
            window = text[max(0, match.start() - 50):match.end()]
            has_specific = any(
                req.get("type") == "years"
                and float(req.get("years", 0)) == years
                and normalize_skill(req.get("skill", "")) in normalize_skill(window)
                for req in must_haves
            )
            if not has_specific and years not in existing_experience:
                must_haves.append({"type": "experience", "years": years})
                existing_experience.add(years)

    preference_pattern = re.compile(
        r"\b(?:nice\s+to\s+have|nice-to-have|preferred|preferably|bonus|"
        r"plus|desirable|good\s+to\s+have)\b",
        re.IGNORECASE,
    )

    hard_skill_names: set[str] = set()
    for requirement in must_haves:
        req_type = requirement.get("type")
        if req_type in {"skill", "years"}:
            hard_skill_names.add(normalize_skill(requirement.get("skill", "")))
        elif req_type == "skill_any":
            hard_skill_names.update(
                normalize_skill(skill) for skill in requirement.get("skills", [])
            )

    nice_skills: list[str] = []
    preference_clauses: list[str] = []
    for clause in re.split(r"(?<=[.!?;:\n])\s+", text):
        clause = clause.strip()
        if not clause or not preference_pattern.search(clause):
            continue
        preference_clauses.append(clause)
        for skill in _known_skills(clause):
            key = normalize_skill(skill)
            if key not in hard_skill_names and key not in {normalize_skill(x) for x in nice_skills}:
                nice_skills.append(skill)

    # Deduplicate all requirements while preserving order.
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for req in must_haves:
        kind = req.get("type")
        if kind == "skill":
            key = (kind, normalize_skill(req.get("skill", "")))
        elif kind == "years":
            key = (kind, normalize_skill(req.get("skill", "")), float(req.get("years", 0)))
        elif kind == "experience":
            key = (kind, float(req.get("years", 0)))
        elif kind == "skill_any":
            key = (kind, tuple(sorted(normalize_skill(x) for x in req.get("skills", []))))
        else:
            continue
        if key not in seen:
            deduped.append(req)
            seen.add(key)

    return {
        "must_have": deduped,
        "nice_to_have": nice_skills,
        "detected_skills": detected_skills,
        "preference_clauses": preference_clauses,
        "summary": {
            "must_have_count": len(deduped),
            "nice_to_have_count": len(nice_skills),
            "detected_skill_count": len(detected_skills),
        },
    }


def _safe_resume_path(raw_path: str, resume_root: str | Path = DEFAULT_RESUME_ROOT) -> Path:
    """Resolve a resume path and ensure it stays inside the resume directory."""
    if not raw_path:
        raise ValueError("Resume path is missing.")

    root = Path(resume_root).expanduser().resolve()
    # Milestone 2 may have been indexed on Windows, where Chroma stores
    # paths such as ``resumes\candidate.pdf``. Normalize separators first so
    # the same trusted-path logic works on Windows and POSIX hosts.
    normalized_raw_path = raw_path.replace("\\", "/")
    candidate_path = Path(normalized_raw_path).expanduser()

    # Milestone 2 stores paths using the project-relative form (for example
    # ``resumes\candidate.pdf``), while this function receives the already
    # trusted ``resumes`` directory as its root. Normalize that representation
    # before applying the containment check so valid indexed resumes are not
    # accidentally resolved as ``resumes/resumes/candidate.pdf``.
    if not candidate_path.is_absolute():
        parts = candidate_path.parts
        root_name = root.name
        if parts and parts[0].casefold() == root_name.casefold():
            candidate_path = root.parent.joinpath(*parts)
        else:
            candidate_path = root / candidate_path

    path = candidate_path.resolve()

    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError("Unsafe resume path: file is outside the configured resume directory.") from exc

    if not path.is_file():
        raise FileNotFoundError(f"Resume file not found: {path}")

    size = path.stat().st_size
    if size > MAX_RESUME_BYTES:
        raise ValueError(f"Resume exceeds the {MAX_RESUME_BYTES // (1024 * 1024)} MiB safety limit.")
    return path


def read_candidate_resume(
    candidate: dict[str, Any], resume_root: str | Path = DEFAULT_RESUME_ROOT
) -> str:
    """Read a resume only from the configured resume directory."""
    raw_path = str(candidate.get("resume_path", "")).strip()
    if not raw_path:
        return ""
    path = _safe_resume_path(raw_path, resume_root)
    suffix = path.suffix.casefold()
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="ignore")[:MAX_EXTRACTED_TEXT_CHARS]
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        if len(reader.pages) > MAX_PDF_PAGES:
            raise ValueError(f"Resume exceeds the {MAX_PDF_PAGES}-page safety limit.")
        chunks: list[str] = []
        total = 0
        for page in reader.pages:
            text = page.extract_text() or ""
            remaining = MAX_EXTRACTED_TEXT_CHARS - total
            if remaining <= 0:
                break
            chunks.append(text[:remaining])
            total += min(len(text), remaining)
        return "\n".join(chunks)
    if suffix == ".docx":
        # DOCX is a ZIP container. Check declared uncompressed size before
        # handing it to python-docx to reduce zip-bomb/resource-exhaustion risk.
        with zipfile.ZipFile(path) as archive:
            declared_size = sum(info.file_size for info in archive.infolist())
            if declared_size > MAX_DOCX_UNCOMPRESSED_BYTES:
                raise ValueError(
                    "DOCX uncompressed content exceeds the 50 MiB safety limit."
                )

        from docx import Document

        chunks: list[str] = []
        total = 0
        for paragraph in Document(str(path)).paragraphs:
            remaining = MAX_EXTRACTED_TEXT_CHARS - total
            if remaining <= 0:
                break
            text = paragraph.text or ""
            chunks.append(text[:remaining])
            total += min(len(text), remaining)
        return "\n".join(chunks)
    raise ValueError(f"Unsupported resume format: {path}")


def candidate_profile(
    candidate: dict[str, Any], resume_root: str | Path = DEFAULT_RESUME_ROOT
) -> dict[str, Any]:
    """Build an authoritative candidate profile from the full resume when available."""
    try:
        resume = read_candidate_resume(candidate, resume_root)
    except (OSError, ValueError, ImportError):
        resume = ""

    if resume:
        return {
            "skills": extract_skills(resume),
            "experience_years": estimate_experience_years(resume),
            "resume_available": True,
            "resume_text": resume,
        }

    raw_skills = candidate.get("skills", candidate.get("matched_skills", []))
    if isinstance(raw_skills, str):
        raw_skills = [item.strip() for item in raw_skills.split(",") if item.strip()]
    return {
        "skills": list(raw_skills or []),
        "experience_years": float(candidate.get("experience_years", 0) or 0),
        "resume_available": False,
        "resume_text": "",
    }


def compare_candidates(candidate_ids: list[str], candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare named or ranked candidates using the current agent scores."""
    requested = candidate_ids or ["1", "2", "3"]
    selected: list[dict[str, Any]] = []
    unresolved: list[str] = []

    for value in requested:
        query = str(value).strip()
        match = next(
            (
                item
                for item in candidates
                if query.casefold() in {
                    str(item.get("candidate_id", "")).casefold(),
                    str(item.get("candidate_name", "")).casefold(),
                }
            ),
            None,
        )
        if match is None:
            try:
                rank = int(query)
            except ValueError:
                rank = 0
            if 1 <= rank <= len(candidates):
                match = candidates[rank - 1]
        if match is None:
            unresolved.append(query)
        elif match not in selected:
            selected.append(match)

    rows: list[dict[str, Any]] = []
    for item in selected[:10]:
        rows.append({
            "candidate_id": item.get("candidate_id", ""),
            "name": item.get("candidate_name", "Unknown"),
            "score": float(item.get("match_score", 0) or 0),
            "base_score": float(item.get("base_score", item.get("match_score", 0)) or 0),
            "nice_to_have_bonus": float(item.get("nice_to_have_bonus", 0) or 0),
            "hard_coverage": float(item.get("hard_coverage", 0) or 0),
            "experience_years": item.get("experience_years"),
            "matched_skills": item.get("matched_skills", []),
            "nice_to_have_matches": item.get("nice_to_have_matches", []),
            "strengths": item.get("strengths", []),
            "gaps": item.get("gaps", []),
            "reasoning": item.get("reasoning", ""),
        })

    winner = max(rows, key=lambda x: x["score"], default=None)
    return {
        "candidates": rows,
        "winner": winner["name"] if winner else None,
        "unresolved": unresolved,
    }


def generate_interview_questions(candidate_id: str, candidates: list[dict[str, Any]]) -> list[str]:
    """Generate evidence-based screening questions for a shortlisted candidate."""
    query = (candidate_id or "").strip().casefold()
    candidate = next(
        (
            item
            for item in candidates
            if query == str(item.get("candidate_id", "")).casefold()
            or query == str(item.get("candidate_name", "")).casefold()
        ),
        None,
    )
    if candidate is None:
        return [f"Candidate '{candidate_id}' could not be identified in the current shortlist."]

    questions: list[str] = []
    for skill in candidate.get("matched_skills", [])[:3]:
        questions.append(
            f"Describe a production project where you used {skill}. What was your contribution, and what measurable outcome did you achieve?"
        )
    for skill in candidate.get("nice_to_have_matches", [])[:2]:
        questions.append(
            f"Your resume indicates {skill}. How deeply have you used it in production, and what trade-offs did you make?"
        )
    for gap in candidate.get("gaps", [])[:2]:
        questions.append(
            f"The current screen flags this gap: {gap}. What evidence from your experience addresses it?"
        )
    questions.append(
        "Walk through one technically difficult project on your resume: architecture, your role, key trade-off, and outcome."
    )
    questions.append(
        "What part of this role would require the most ramp-up for you, and how would you close that gap?"
    )

    unique: list[str] = []
    seen: set[str] = set()
    for question in questions:
        if question not in seen:
            unique.append(question)
            seen.add(question)
    return unique[:6]


__all__ = [
    "candidate_profile",
    "compare_candidates",
    "estimate_experience_years",
    "extract_requirements",
    "generate_interview_questions",
    "read_candidate_resume",
    "MAX_RESUME_BYTES",
]
