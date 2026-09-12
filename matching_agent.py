"""LangGraph conversational agent for resume/profile matching.

The agent keeps the existing Milestone 2 RAG matcher as the retrieval source and
adds deterministic conversation routing, recruiter refinements, explainability,
and the assignment's multi-round screening flow.
"""
from __future__ import annotations

import argparse
import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

try:
    from typing_extensions import TypedDict
except ImportError:  # pragma: no cover
    from typing import TypedDict

try:
    from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Install langchain-core to run the agent: pip install -U langchain-core"
    ) from exc

try:
    from langgraph.graph import END, START, StateGraph
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Install LangGraph to run the agent: pip install -U langgraph"
    ) from exc

from agent_tools import (
    compare_candidates,
    extract_requirements,
    generate_interview_questions,
    read_candidate_resume,
)
from profile_matcher import ProfileMatcher, normalize_skill

TOP_K_INITIAL = 10
RETRIEVAL_POOL = 100  # assignment: top 10 from a 100-resume screening pool
NICE_BONUS_MAX = 10.0
# Bounded fallback bonus used only when the RAG score cannot distinguish candidates.
# It does not replace RAG; it prevents misleading all-tied output when hard filtering
# makes the RAG components identical.
STRONG_THRESHOLD = 80.0
BORDERLINE_THRESHOLD = 60.0
MAX_QUERY_CHARS = 100_000


class AgentState(TypedDict, total=False):
    messages: list[BaseMessage]
    job_description: str
    requirements: dict[str, Any]
    active_refinements: list[str]
    excluded_skills: list[str]
    search_query: str
    candidate_shortlist: list[dict[str, Any]]
    reasoning: dict[str, str]
    report: str
    feedback: str
    feedback_applied: bool
    previous_rankings: list[str]
    current_rankings: list[str]
    previous_scores: dict[str, float]
    current_scores: dict[str, float]
    ranking_changes: list[str]
    screening_round: int
    final_recommendations: list[dict[str, Any]]
    next_action: str


class MatchingAgent:
    """Conversational LangGraph recruiter agent backed by ProfileMatcher."""

    def __init__(
        self,
        persist_dir: str = "chroma_db",
        provider: str = "hf",
        resume_dir: str = "resumes",
    ) -> None:
        self.resume_dir = Path(resume_dir).expanduser().resolve()
        self.matcher = ProfileMatcher(
            persist_dir=persist_dir,
            embedding_provider=provider,
        )
        self.graph = self._build_graph()

    # ------------------------------------------------------------------
    # LangGraph nodes
    # ------------------------------------------------------------------
    def parse_jd(self, state: AgentState) -> dict[str, Any]:
        job = (state.get("job_description") or "").strip()
        if not job:
            job = self._last_user_message(state.get("messages", []))
        search_query = (state.get("search_query") or job).strip()
        return {"job_description": job, "search_query": search_query}

    def extract_requirements_node(self, state: AgentState) -> dict[str, Any]:
        result = extract_requirements(
            state.get("search_query") or state.get("job_description", "")
        )
        result = self._remove_excluded_requirements(
            result,
            state.get("excluded_skills", []),
        )
        return {"requirements": result}

    @staticmethod
    def _remove_excluded_requirements(
        requirements: dict[str, Any], excluded_skills: list[str]
    ) -> dict[str, Any]:
        excluded = {normalize_skill(x) for x in excluded_skills}
        if not excluded:
            return requirements

        kept: list[dict[str, Any]] = []
        for req in requirements.get("must_have", []):
            kind = req.get("type")
            if kind in {"skill", "years"} and normalize_skill(req.get("skill", "")) in excluded:
                continue
            if kind == "skill_any":
                alternatives = [
                    x
                    for x in req.get("skills", [])
                    if normalize_skill(x) not in excluded
                ]
                if not alternatives:
                    continue
                kept.append({**req, "skills": alternatives})
                continue
            kept.append(req)

        nice = [
            x
            for x in requirements.get("nice_to_have", [])
            if normalize_skill(x) not in excluded
        ]
        result = dict(requirements)
        result["must_have"] = kept
        result["nice_to_have"] = nice
        result["summary"] = {
            "must_have_count": len(kept),
            "nice_to_have_count": len(nice),
            "detected_skill_count": len(result.get("detected_skills", [])),
        }
        return result

    def search_resumes(self, state: AgentState) -> dict[str, Any]:
        query = (
            state.get("search_query") or state.get("job_description") or ""
        ).strip()
        if not query:
            return {"candidate_shortlist": []}

        requirements = state.get("requirements", {})
        retrieval_query = self._build_retrieval_query(query, requirements)

        # Search the entire indexed chunk set when available. ProfileMatcher groups
        # chunks by resume_path, so the agent can make the required top-10 decision
        # from the full 100-resume dataset rather than from an accidental subset.
        chunk_count = self.matcher.collection.count()
        retrieval_k = max(RETRIEVAL_POOL, chunk_count) if chunk_count else RETRIEVAL_POOL
        result = self.matcher.match(retrieval_query, top_k=retrieval_k)
        candidates = result.get("top_matches", [])

        metadata_by_path = self._load_authoritative_metadata(candidates)
        excluded = {normalize_skill(x) for x in state.get("excluded_skills", [])}

        filtered: list[dict[str, Any]] = []
        for raw in candidates:
            metadata = metadata_by_path.get(str(raw.get("resume_path", "")), {})
            skills = self._metadata_skills(
                metadata.get("skills", raw.get("skills", raw.get("matched_skills", [])))
            )
            experience = float(
                metadata.get(
                    "experience_years",
                    raw.get("experience_years", 0),
                )
                or 0
            )
            profile = {"skills": skills, "experience_years": experience}

            if excluded and any(normalize_skill(skill) in excluded for skill in skills):
                continue
            if not self._passes_all_requirements(requirements, profile):
                continue

            item = deepcopy(raw)
            item["candidate_id"] = (
                item.get("resume_path")
                or item.get("candidate_id")
                or item.get("candidate_name", "unknown")
            )
            item["skills"] = skills
            item["experience_years"] = experience
            item["resume_available"] = bool(item.get("resume_path"))
            filtered.append(item)

        return {"candidate_shortlist": filtered}

    def _load_authoritative_metadata(
        self, candidates: list[dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        paths = [
            str(item.get("resume_path", ""))
            for item in candidates
            if item.get("resume_path")
        ]
        if not paths:
            return {}
        try:
            data = self.matcher.collection.get(
                where={"resume_path": {"$in": paths}},
                include=["metadatas"],
            )
        except Exception as exc:
            raise RuntimeError(
                "Unable to load authoritative candidate metadata from ChromaDB."
            ) from exc

        output: dict[str, dict[str, Any]] = {}
        for metadata in data.get("metadatas", []) or []:
            path = str(metadata.get("resume_path", ""))
            if path:
                output[path] = metadata
        return output

    @staticmethod
    def _metadata_skills(raw_skills: Any) -> list[str]:
        if isinstance(raw_skills, str):
            try:
                parsed = json.loads(raw_skills)
                raw_skills = parsed
            except (TypeError, json.JSONDecodeError):
                raw_skills = [
                    value.strip()
                    for value in raw_skills.split(",")
                    if value.strip()
                ]
        return [str(value) for value in (raw_skills or []) if str(value).strip()]

    def rank_candidates(self, state: AgentState) -> dict[str, Any]:
        requirements = state.get("requirements", {})
        ranked_input = state.get("candidate_shortlist", [])
        enriched: list[dict[str, Any]] = []

        base_scores = [
            float(item.get("match_score", 0) or 0)
            for item in ranked_input
        ]
        score_tie = len(base_scores) > 1 and max(base_scores) == min(base_scores)

        for raw in ranked_input:
            candidate = deepcopy(raw)
            candidate_skills = candidate.get("skills", [])
            nice_matches = self._nice_matches(requirements, candidate_skills)
            nice_count = len(requirements.get("nice_to_have", []))
            nice_bonus = NICE_BONUS_MAX * (
                len(nice_matches) / nice_count if nice_count else 0.0
            )

            rag_score = float(candidate.get("match_score", 0) or 0)
            hard_coverage, satisfied, gaps = self._requirement_evaluation(
                requirements,
                candidate,
            )
            fallback_score = self._fallback_evidence_score(
                candidate,
                requirements,
            ) if score_tie else rag_score
            effective_base = max(rag_score, fallback_score) if score_tie else rag_score
            final_score = min(100.0, effective_base + nice_bonus)
            strengths = self._derive_strengths(
                requirements,
                candidate,
                satisfied,
                nice_matches,
            )

            satisfied_count = len(satisfied)
            total_requirements = len(requirements.get("must_have", []))
            candidate.update(
                {
                    "base_score": round(rag_score, 2),
                    "nice_to_have_matches": nice_matches,
                    "nice_to_have_bonus": round(nice_bonus, 2),
                    "fallback_evidence_score": round(fallback_score, 2) if score_tie else 0.0,
                    "rag_score_was_tied": score_tie,
                    "hard_coverage": round(hard_coverage * 100, 2),
                    "hard_requirements_satisfied": satisfied,
                    "hard_requirements_satisfied_count": satisfied_count,
                    "hard_requirements_total": total_requirements,
                    "strengths": strengths,
                    "gaps": gaps,
                }
            )
            candidate["match_score"] = round(final_score, 2)
            candidate["screening_status"] = self._screening_status(
                final_score,
                gaps,
            )
            enriched.append(candidate)

        enriched.sort(
            key=lambda item: (
                float(item.get("match_score", 0) or 0),
                float(item.get("base_score", 0) or 0),
                float(item.get("experience_years", 0) or 0),
                str(item.get("candidate_name", "")).casefold(),
                str(item.get("candidate_id", "")).casefold(),
            ),
            reverse=True,
        )
        enriched = enriched[:TOP_K_INITIAL]

        old_rankings = list(state.get("previous_rankings", []))
        old_scores = dict(state.get("previous_scores", {}))
        new_rankings = [self._candidate_key(item) for item in enriched]
        new_scores = {
            self._candidate_key(item): float(item.get("match_score", 0) or 0)
            for item in enriched
        }
        changes = self._ranking_changes(
            old_rankings,
            new_rankings,
            old_scores,
            new_scores,
            enriched,
        )
        return {
            "candidate_shortlist": enriched,
            "current_rankings": new_rankings,
            "current_scores": new_scores,
            "ranking_changes": changes,
            "reasoning": {
                self._candidate_key(item): self._ranking_reason(item)
                for item in enriched
            },
        }

    def generate_report(self, state: AgentState) -> dict[str, Any]:
        candidates = state.get("candidate_shortlist", [])
        lines = [
            "### Screening Round 1 — Candidate Match Report",
            "",
            self._requirements_summary(state.get("requirements", {})),
            "",
        ]

        if not candidates:
            lines.append(
                "**No matching candidates found.** Try relaxing a requirement or using a broader skill."
            )
            return {"report": "\n".join(lines)}

        lines.extend(
            [
                "**Screening pool:** up to 100 resumes → top 10 candidates",
                "**Ranking:** highest final match score first",
                "",
                "#### Top Candidates",
                "",
            ]
        )
        for rank, candidate in enumerate(candidates, start=1):
            name = candidate.get("candidate_name", "Unknown")
            score = float(candidate.get("match_score", 0) or 0)
            experience = float(candidate.get("experience_years", 0) or 0)
            status = str(candidate.get("screening_status", "unknown")).title()
            strengths = self._join(candidate.get("strengths", []))
            gaps = self._join(candidate.get("gaps", []))
            nice = self._join(candidate.get("nice_to_have_matches", []))
            rationale = candidate.get(
                "reasoning",
                "No additional rationale available.",
            )
            satisfied = int(candidate.get("hard_requirements_satisfied_count", 0) or 0)
            total = int(candidate.get("hard_requirements_total", 0) or 0)
            if total:
                requirement_text = f"{satisfied}/{total}"
            else:
                requirement_text = "0/0"

            lines.append(
                f"**{rank}. {name}** — **{score:.2f}/100** · "
                f"{experience:g} years · **{status}**"
            )
            lines.append(
                f"- **Must-haves satisfied:** {requirement_text}"
            )
            if candidate.get("strengths"):
                lines.append(f"- **Strengths:** {strengths}")
            if candidate.get("gaps"):
                lines.append(f"- **Gaps:** {gaps}")
            if candidate.get("nice_to_have_matches"):
                lines.append(f"- **Preferred skills:** {nice}")
            lines.append(f"- **Why:** {rationale}")
            lines.append("")

        if state.get("ranking_changes"):
            lines.extend(
                [
                    "#### What Changed From the Previous Recruiter Turn",
                    "",
                ]
            )
            lines.extend(
                f"- {change}" for change in state["ranking_changes"]
            )
        return {"report": "\n".join(lines).rstrip()}

    def human_feedback_loop(self, state: AgentState) -> dict[str, Any]:
        # Refinement is applied by invoke() before the graph starts. This node keeps
        # the assignment's named feedback-loop stage without causing a duplicate
        # retrieval pass inside the same conversational turn.
        return {"feedback_applied": True, "next_action": "end"}

    # ------------------------------------------------------------------
    # Public conversation API
    # ------------------------------------------------------------------
    def invoke(
        self,
        user_message: str,
        state: AgentState | None = None,
    ) -> AgentState:
        message = (user_message or "").strip()
        if not message:
            return state or {"messages": [], "screening_round": 1}
        if len(message) > MAX_QUERY_CHARS:
            raise ValueError(
                f"Recruiter query exceeds the {MAX_QUERY_CHARS:,}-character safety limit."
            )

        current = deepcopy(state or {"messages": [], "screening_round": 1})
        current.setdefault("messages", []).append(HumanMessage(content=message))
        current.setdefault("current_rankings", [])
        current.setdefault("previous_rankings", [])
        current.setdefault("previous_scores", {})
        current.setdefault("current_scores", {})
        current.setdefault("ranking_changes", [])

        small_talk = self._small_talk_response(message)
        if small_talk is not None:
            current["report"] = small_talk
            current["messages"].append(AIMessage(content=small_talk))
            return current

        # First, recognize criteria refinements. This must happen before the
        # generic "show candidates" intent so phrases such as "Only show candidates
        # who have Docker" update the active requirements instead of merely displaying
        # the old shortlist.
        if current.get("job_description") and self._looks_like_refinement(message):
            old_rankings = list(current.get("current_rankings", []))
            if not old_rankings and current.get("candidate_shortlist"):
                old_rankings = [
                    self._candidate_key(item)
                    for item in current["candidate_shortlist"]
                ]
            current["previous_rankings"] = old_rankings
            current["previous_scores"] = {
                self._candidate_key(item): float(item.get("match_score", 0) or 0)
                for item in current.get("candidate_shortlist", [])
            }
            current["feedback"] = message
            current["feedback_applied"] = False
            current.setdefault("active_refinements", [])
            current.setdefault("excluded_skills", [])
            self._apply_refinement(current, message)
            current["search_query"] = self._effective_job_text(current)
            result = self.graph.invoke(current)
            result["messages"] = result.get("messages", []) + [
                AIMessage(content=result.get("report", "No report generated."))
            ]
            return result

        # Existing-shortlist utilities never rerun RAG. This preserves conversational
        # state for highest/lowest score, comparison, explanation, and interview turns.
        if current.get("candidate_shortlist"):
            response = self._handle_existing_shortlist_turn(message, current)
            if response is not None:
                current["report"] = response
                current["messages"].append(AIMessage(content=response))
                return current

        # Utility commands without a shortlist should not become fake job descriptions.
        if self._is_shortlist_utility(message):
            response = (
                "There is no active candidate shortlist yet. Start with a search, "
                "for example: `Find me candidates with Python and 3+ years experience.`"
            )
            current["report"] = response
            current["messages"].append(AIMessage(content=response))
            return current

        # Otherwise this is a new job/search request. Reset conversation-scoped
        # ranking/refinement state while retaining the message history.
        current["job_description"] = message
        current["search_query"] = message
        current["feedback"] = ""
        current["feedback_applied"] = False
        current["active_refinements"] = []
        current["excluded_skills"] = []
        current["previous_rankings"] = []
        current["current_rankings"] = []
        current["previous_scores"] = {}
        current["ranking_changes"] = []
        current["screening_round"] = 1

        result = self.graph.invoke(current)
        result["messages"] = result.get("messages", []) + [
            AIMessage(content=result.get("report", "No report generated."))
        ]
        return result

    def run_job(self, job_text: str) -> AgentState:
        return self.invoke(job_text)

    def run_multi_round(self, job_text: str) -> AgentState:
        """Run Round 1 top-10 -> Round 2 deep review -> Round 3 recommendation."""
        state = self.invoke(job_text)
        shortlist = state.get("candidate_shortlist", [])[:TOP_K_INITIAL]

        state["screening_round"] = 2
        for candidate in shortlist:
            candidate["deep_analysis"] = self._deep_analysis(candidate)
            candidate["round_2_status"] = "analyzed"

        state["screening_round"] = 3
        state["final_recommendations"] = [
            self._final_recommendation(candidate)
            for candidate in shortlist
        ]
        state["candidate_shortlist"] = shortlist
        state["report"] = self._multi_round_report(state)
        state["messages"] = state.get("messages", []) + [
            AIMessage(content=state["report"])
        ]
        return state

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------
    def _build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("parse_jd", self.parse_jd)
        graph.add_node("extract_requirements", self.extract_requirements_node)
        graph.add_node("search_resumes", self.search_resumes)
        graph.add_node("rank_candidates", self.rank_candidates)
        graph.add_node("generate_report", self.generate_report)
        graph.add_node("human_feedback_loop", self.human_feedback_loop)

        graph.add_edge(START, "parse_jd")
        graph.add_edge("parse_jd", "extract_requirements")
        graph.add_edge("extract_requirements", "search_resumes")
        graph.add_edge("search_resumes", "rank_candidates")
        graph.add_edge("rank_candidates", "generate_report")
        graph.add_edge("generate_report", "human_feedback_loop")
        graph.add_conditional_edges(
            "human_feedback_loop",
            lambda state: state.get("next_action", "end"),
            {"end": END, "refine": "extract_requirements"},
        )
        return graph.compile()

    # ------------------------------------------------------------------
    # Conversational routing
    # ------------------------------------------------------------------
    @staticmethod
    def _small_talk_response(text: str) -> str | None:
        normalized = re.sub(r"[^a-z0-9]+", " ", (text or "").casefold()).strip()
        greetings = {
            "hi",
            "hii",
            "hiii",
            "hello",
            "hey",
            "hiya",
            "good morning",
            "good afternoon",
            "good evening",
        }
        thanks = {"thanks", "thank you", "thx", "ty"}
        exits = {"bye", "goodbye", "see you", "quit", "exit"}

        if normalized in greetings:
            return (
                "Hello! I’m the Agentic Profile Matching assistant.\n\n"
                "I can find and rank candidates, compare matches, explain ranking decisions, "
                "apply recruiter refinements, and generate interview questions.\n\n"
                "Try: `Find me candidates with Python and 3+ years experience.`"
            )
        if normalized in thanks:
            return (
                "You're welcome. Ask me about candidates, ranking, requirements, "
                "or interview questions."
            )
        if normalized in exits:
            return "Goodbye. The current shortlist remains in the conversation state until you reset it."
        if normalized == "help":
            return (
                "Supported actions: search candidates, compare the top 3, explain why one candidate "
                "ranks above another, find the highest/lowest score, rank by score, add/remove requirements, "
                "filter candidates, explain ranking changes, and generate interview questions."
            )
        return None

    def _handle_existing_shortlist_turn(
        self,
        message: str,
        state: AgentState,
    ) -> str | None:
        candidates = state["candidate_shortlist"]

        if self._is_interview_questions(message):
            name = self._extract_candidate_reference(message, candidates)
            questions = generate_interview_questions(name, candidates)
            return "### Interview Questions\n\n" + "\n".join(
                f"{index}. {question}"
                for index, question in enumerate(questions, 1)
            )

        if self._is_compare(message):
            return self._format_comparison(message, candidates)

        if self._is_why(message):
            return self._explain_comparison(message, candidates)

        if self._is_low_score(message):
            worst = min(
                candidates,
                key=lambda item: (
                    float(item.get("match_score", 0) or 0),
                    -float(item.get("experience_years", 0) or 0),
                ),
            )
            return self._format_named_candidate(worst, "Lowest Current Match")

        if self._is_max_score(message):
            best = max(
                candidates,
                key=lambda item: float(item.get("match_score", 0) or 0),
            )
            return self._format_named_candidate(best, "Highest Current Match")

        if self._is_rank_by_score(message):
            ordered = sorted(
                candidates,
                key=lambda item: float(item.get("match_score", 0) or 0),
                reverse=True,
            )
            return self._format_shortlist(ordered, heading="Current Ranking by Score")

        if self._is_ranking_change_question(message):
            changes = state.get("ranking_changes", [])
            if not changes:
                return "### Ranking Changes\n\nNo ranking changes were recorded for the previous recruiter turn."
            return "### Ranking Changes\n\n" + "\n".join(
                f"- {change}" for change in changes
            )

        if self._is_show_shortlist(message):
            return self._format_shortlist(candidates)

        return None

    @staticmethod
    def _is_shortlist_utility(text: str) -> bool:
        return any(
            checker(text)
            for checker in (
                MatchingAgent._is_interview_questions,
                MatchingAgent._is_compare,
                MatchingAgent._is_why,
                MatchingAgent._is_low_score,
                MatchingAgent._is_max_score,
                MatchingAgent._is_rank_by_score,
                MatchingAgent._is_ranking_change_question,
                MatchingAgent._is_show_shortlist,
            )
        )

    # ------------------------------------------------------------------
    # Requirement and score helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _passes_all_requirements(
        requirements: dict[str, Any],
        profile: dict[str, Any],
    ) -> bool:
        skills = {normalize_skill(skill) for skill in profile.get("skills", [])}
        experience = float(profile.get("experience_years", 0) or 0)
        for req in requirements.get("must_have", []):
            kind = req.get("type")
            if kind == "skill" and normalize_skill(req.get("skill", "")) not in skills:
                return False
            if kind == "years":
                if normalize_skill(req.get("skill", "")) not in skills:
                    return False
                if experience < float(req.get("years", 0) or 0):
                    return False
            if kind == "experience" and experience < float(req.get("years", 0) or 0):
                return False
            if kind == "skill_any" and not any(
                normalize_skill(skill) in skills
                for skill in req.get("skills", [])
            ):
                return False
        return True

    @staticmethod
    def _requirement_evaluation(
        requirements: dict[str, Any],
        candidate: dict[str, Any],
    ) -> tuple[float, list[str], list[str]]:
        skills = {normalize_skill(skill) for skill in candidate.get("skills", [])}
        experience = float(candidate.get("experience_years", 0) or 0)
        requirement_list = requirements.get("must_have", [])
        if not requirement_list:
            return 1.0, [], []

        satisfied: list[str] = []
        gaps: list[str] = []
        for req in requirement_list:
            kind = req.get("type")
            if kind == "skill":
                skill = str(req.get("skill", ""))
                if normalize_skill(skill) in skills:
                    satisfied.append(skill)
                else:
                    gaps.append(f"Missing required skill: {skill}")
            elif kind == "years":
                skill = str(req.get("skill", ""))
                years = float(req.get("years", 0) or 0)
                has_skill = normalize_skill(skill) in skills
                has_years = experience >= years
                if has_skill and has_years:
                    satisfied.append(f"{skill} ({years:g}+ years)")
                elif not has_skill:
                    gaps.append(f"Missing required skill: {skill}")
                else:
                    gaps.append(
                        f"Requires {years:g}+ years {skill}; candidate has {experience:g} years"
                    )
            elif kind == "experience":
                years = float(req.get("years", 0) or 0)
                if experience >= years:
                    satisfied.append(f"{years:g}+ years overall experience")
                else:
                    gaps.append(
                        f"Requires {years:g}+ years overall experience; candidate has {experience:g} years"
                    )
            elif kind == "skill_any":
                alternatives = [str(x) for x in req.get("skills", [])]
                matched = [
                    x for x in alternatives
                    if normalize_skill(x) in skills
                ]
                if matched:
                    satisfied.append(
                        "One of: " + ", ".join(alternatives) + f" -> {matched[0]}"
                    )
                else:
                    gaps.append("Requires one of: " + ", ".join(alternatives))

        return len(satisfied) / len(requirement_list), satisfied, gaps

    @staticmethod
    def _nice_matches(requirements: dict[str, Any], skills: list[str]) -> list[str]:
        candidate = {normalize_skill(x) for x in skills}
        return [
            skill
            for skill in requirements.get("nice_to_have", [])
            if normalize_skill(skill) in candidate
        ]

    @staticmethod
    def _derive_strengths(
        requirements: dict[str, Any],
        candidate: dict[str, Any],
        satisfied: list[str],
        nice_matches: list[str],
    ) -> list[str]:
        strengths: list[str] = []
        for value in candidate.get("matched_skills", []) + satisfied + nice_matches:
            if value not in strengths:
                strengths.append(value)
        return strengths[:8]

    @staticmethod
    def _fallback_evidence_score(
        candidate: dict[str, Any],
        requirements: dict[str, Any],
    ) -> float:
        """Produce a bounded fallback score only when RAG scores are identical.

        This is deliberately independent of semantic retrieval. It rewards complete
        hard-requirement coverage, relevant skill coverage, and experience evidence
        so a broken/non-discriminative embedding score cannot make every candidate
        look identical.
        """
        hard_coverage, _, _ = MatchingAgent._requirement_evaluation(
            requirements, candidate
        )
        candidate_skills = {normalize_skill(x) for x in candidate.get("skills", [])}
        required_skill_keys: set[str] = set()
        matched_skill_keys: set[str] = set()
        for req in requirements.get("must_have", []):
            if req.get("type") in {"skill", "years"}:
                key = normalize_skill(req.get("skill", ""))
                if key:
                    required_skill_keys.add(key)
            elif req.get("type") == "skill_any":
                alternatives = [normalize_skill(x) for x in req.get("skills", [])]
                required_skill_keys.update(x for x in alternatives if x)
        for key in required_skill_keys:
            if key in candidate_skills:
                matched_skill_keys.add(key)
        if required_skill_keys:
            skill_coverage = len(matched_skill_keys) / len(required_skill_keys)
        else:
            skill_coverage = 0.0

        experience = float(candidate.get("experience_years", 0) or 0)
        target_years = max(
            [
                float(req.get("years", 0) or 0)
                for req in requirements.get("must_have", [])
                if req.get("type") in {"years", "experience"}
            ],
            default=0.0,
        )
        reference_years = max(target_years, 10.0)
        experience_fit = min(1.0, experience / reference_years)

        score = (
            60.0 * hard_coverage
            + 30.0 * experience_fit
            + 10.0 * skill_coverage
        )
        return round(min(100.0, max(0.0, score)), 2)

    @classmethod
    def _ranking_reason(cls, candidate: dict[str, Any]) -> str:
        fallback_note = ""
        if candidate.get("rag_score_was_tied"):
            fallback_note = (
                f" Because the RAG scores were tied, the evidence fallback score "
                f"was used ({float(candidate.get('fallback_evidence_score', 0) or 0):.2f})."
            )
        return (
            f"Final score {float(candidate.get('match_score', 0) or 0):.2f} = "
            f"RAG score {float(candidate.get('base_score', 0) or 0):.2f} + "
            f"preferred-skill bonus {float(candidate.get('nice_to_have_bonus', 0) or 0):.2f}."
            f"{fallback_note} "
            f"Hard-requirement coverage: {float(candidate.get('hard_coverage', 0) or 0):.0f}%. "
            f"Satisfied hard requirements: "
            f"{int(candidate.get('hard_requirements_satisfied_count', 0) or 0)}/"
            f"{int(candidate.get('hard_requirements_total', 0) or 0)}."
        )

    @staticmethod
    def _screening_status(score: float, gaps: list[str]) -> str:
        if gaps:
            return "borderline" if score >= BORDERLINE_THRESHOLD else "weak"
        if score >= STRONG_THRESHOLD:
            return "strong"
        if score >= BORDERLINE_THRESHOLD:
            return "borderline"
        return "weak"

    @staticmethod
    def _build_retrieval_query(
        job_text: str,
        requirements: dict[str, Any],
    ) -> str:
        query = job_text
        for skill in requirements.get("nice_to_have", []):
            query = re.sub(
                r"(?<!\w)" + re.escape(skill) + r"(?!\w)",
                " ",
                query,
                flags=re.I,
            )
        query = re.sub(
            r"\b(?:nice\s+to\s+have|nice-to-have|preferred|preferably|bonus|desirable|good\s+to\s+have)\b[^.;:!\n]*",
            " ",
            query,
            flags=re.I,
        )
        return re.sub(r"\s+", " ", query).strip() or job_text

    # ------------------------------------------------------------------
    # Refinement state
    # ------------------------------------------------------------------
    def _apply_refinement(self, state: AgentState, message: str) -> None:
        lower = message.casefold()
        skills = self._known_skills(message)

        if re.search(r"\b(?:remove|exclude|drop|without)\b", lower):
            existing = {
                normalize_skill(x): x
                for x in state.get("excluded_skills", [])
            }
            for skill in skills:
                existing[normalize_skill(skill)] = skill
            state["excluded_skills"] = list(existing.values())
            return

        # "prefer/preferred" creates a nice-to-have refinement.  It is retained as
        # recruiter context and extracted by the normal requirement parser.
        if re.search(r"\b(?:prefer|preferred|preferably)\b", lower):
            normalized = message
            state.setdefault("active_refinements", []).append(normalized)
            return

        normalized = message
        if re.search(r"\bonly\b", lower):
            if skills:
                normalized = "must have " + " and ".join(dict.fromkeys(skills))
        elif re.search(r"\b(?:require|required|mandatory|essential)\b", lower):
            normalized = re.sub(
                r"\brequire\b",
                "must have",
                normalized,
                flags=re.I,
            )
            normalized = re.sub(
                r"\brequired\b",
                "must have",
                normalized,
                flags=re.I,
            )
        elif re.search(r"\badd\b", lower):
            normalized = re.sub(
                r"\badd\b",
                "must have",
                normalized,
                flags=re.I,
            )
        state.setdefault("active_refinements", []).append(normalized)

    @staticmethod
    def _effective_job_text(state: AgentState) -> str:
        base = state.get("job_description", "").strip()
        refinements = state.get("active_refinements", [])
        return base + ("\n" + "\n".join(refinements) if refinements else "")

    @staticmethod
    def _known_skills(text: str) -> list[str]:
        from rag_profile_matching import COMMON_SKILLS

        lowered = text.casefold()
        found: list[str] = []
        seen: set[str] = set()
        for skill in sorted(COMMON_SKILLS, key=len, reverse=True):
            key = normalize_skill(skill)
            if key in seen:
                continue
            if re.search(
                r"(?<!\w)" + re.escape(skill.casefold()) + r"(?!\w)",
                lowered,
            ):
                found.append(skill)
                seen.add(key)
        return found

    # ------------------------------------------------------------------
    # Round 2/3
    # ------------------------------------------------------------------
    def _deep_analysis(self, candidate: dict[str, Any]) -> dict[str, Any]:
        try:
            full_resume = read_candidate_resume(
                candidate,
                self.resume_dir,
            )
            error = ""
        except (OSError, ValueError, ImportError) as exc:
            full_resume = ""
            error = str(exc)

        evidence = [
            value
            for value in candidate.get("relevant_excerpts", [])
            if value
        ]
        if full_resume:
            evidence.append(full_resume[:2500])
        return {
            "resume_read": bool(full_resume),
            "resume_character_count": len(full_resume),
            "evidence_excerpts": evidence[:5],
            "strengths": candidate.get("strengths", []),
            "gaps": candidate.get("gaps", []),
            "interview_questions": generate_interview_questions(
                candidate.get("candidate_name", ""),
                [candidate],
            ),
            "read_error": error,
        }

    @staticmethod
    def _final_recommendation(candidate: dict[str, Any]) -> dict[str, Any]:
        score = float(candidate.get("match_score", 0) or 0)
        gaps = candidate.get("gaps", [])
        deep = candidate.get("deep_analysis", {})
        if score >= STRONG_THRESHOLD and not gaps and deep.get("resume_read"):
            recommendation = "HIRE"
            reason = (
                "Strong score, all current hard requirements are satisfied, "
                "and full-resume evidence was available for review."
            )
        elif score >= BORDERLINE_THRESHOLD and not gaps:
            recommendation = "BORDERLINE"
            reason = (
                "Promising match, but the current evidence does not justify "
                "a strong final recommendation."
            )
        else:
            recommendation = "NO-HIRE"
            reason = (
                "The current screening evidence is below the required bar "
                "or contains unresolved gaps."
            )
        return {
            "candidate_name": candidate.get("candidate_name", "Unknown"),
            "score": score,
            "recommendation": recommendation,
            "reason": reason,
        }

    @staticmethod
    def _requirements_summary(requirements: dict[str, Any]) -> str:
        hard: list[str] = []
        for req in requirements.get("must_have", []):
            kind = req.get("type")
            if kind == "skill":
                hard.append(str(req.get("skill")))
            elif kind == "years":
                hard.append(
                    f"{req.get('skill')} ({req.get('years')}+ years)"
                )
            elif kind == "experience":
                hard.append(f"{req.get('years')}+ years overall")
            elif kind == "skill_any":
                hard.append(
                    "one of " + "/".join(map(str, req.get("skills", [])))
                )
        nice = [str(value) for value in requirements.get("nice_to_have", [])]
        hard_text = ", ".join(hard) if hard else "None specified"
        nice_text = ", ".join(nice) if nice else "None specified"
        return (
            "**Requirements**\n\n"
            f"- **Must-have:** {hard_text}\n"
            f"- **Nice-to-have:** {nice_text}"
        )

    @staticmethod
    def _candidate_key(candidate: dict[str, Any]) -> str:
        return str(
            candidate.get("candidate_id")
            or candidate.get("resume_path")
            or candidate.get("candidate_name")
            or "unknown"
        )

    @staticmethod
    def _ranking_changes(
        old: list[str],
        new: list[str],
        old_scores: dict[str, float] | None = None,
        new_scores: dict[str, float] | None = None,
        candidates: list[dict[str, Any]] | None = None,
    ) -> list[str]:
        if not old:
            return []
        old_scores = old_scores or {}
        new_scores = new_scores or {}
        display_names = {
            MatchingAgent._candidate_key(item): str(
                item.get("candidate_name", "Unknown")
            )
            for item in (candidates or [])
        }
        old_pos = {key: idx for idx, key in enumerate(old, 1)}
        new_pos = {key: idx for idx, key in enumerate(new, 1)}
        changes: list[str] = []

        for key, position in new_pos.items():
            name = display_names.get(key, key)
            previous = old_pos.get(key)
            if previous is None:
                changes.append(f"{name} entered the shortlist at #{position}.")
            elif previous != position:
                direction = "up" if previous > position else "down"
                changes.append(
                    f"{name} moved {direction} from #{previous} to #{position}."
                )
            if key in old_scores and key in new_scores:
                delta = float(new_scores[key]) - float(old_scores[key])
                if abs(delta) >= 0.01:
                    sign = "+" if delta > 0 else ""
                    changes.append(
                        f"{name} score changed from {float(old_scores[key]):.2f} "
                        f"to {float(new_scores[key]):.2f} ({sign}{delta:.2f})."
                    )

        for key in old_pos:
            if key not in new_pos:
                changes.append(
                    f"{display_names.get(key, key)} left the current shortlist."
                )
        return changes

    # ------------------------------------------------------------------
    # Command parsing and formatting
    # ------------------------------------------------------------------
    @staticmethod
    def _is_max_score(text: str) -> bool:
        return bool(
            re.search(
                r"\b(?:highest|maximum|max)\s+score\b"
                r"|\b(?:candidate|match)\b.{0,20}\b(?:highest|maximum|max)\b.{0,15}\bscore\b"
                r"|\bwho\b.{0,30}\b(?:highest|maximum|max)\b.{0,15}\bscore\b",
                text,
                re.I,
            )
        ) or bool(
            re.search(r"\b(?:best|top)\s+(?:candidate|match)\b", text, re.I)
        )

    @staticmethod
    def _is_low_score(text: str) -> bool:
        return bool(
            re.search(
                r"\b(?:lowest|minimum|min)\s+score\b"
                r"|\b(?:candidate|match)\b.{0,20}\b(?:lowest|minimum|min)\b.{0,15}\bscore\b"
                r"|\bwho\b.{0,30}\b(?:lowest|minimum|min)\b.{0,15}\bscore\b",
                text,
                re.I,
            )
        )

    @staticmethod
    def _is_rank_by_score(text: str) -> bool:
        return bool(
            re.search(
                r"\b(?:rank|sort|order)\b.*\b(?:candidate|candidates|matches?)\b.*\bscore\b"
                r"|\b(?:rank|sort|order)\b.*\b(?:by|according to)\b.*\bscore\b",
                text,
                re.I,
            )
        )

    @staticmethod
    def _is_interview_questions(text: str) -> bool:
        return bool(
            re.search(
                r"\b(?:generate|give|create|prepare|ask)\b.{0,40}\binterview\s+questions?\b"
                r"|\binterview\s+questions?\b",
                text,
                re.I,
            )
        )

    @staticmethod
    def _is_compare(text: str) -> bool:
        return bool(
            re.search(
                r"\bcompare\b|side[- ]by[- ]side|head[- ]to[- ]head",
                text,
                re.I,
            )
        )

    @staticmethod
    def _is_why(text: str) -> bool:
        return bool(
            re.search(
                r"\bwhy\b.*\brank|\bwhy\b.*\bhigher|\bwhy\b.*\blower|"
                r"\bwhy\b.*\bscore|\breason\b.*\brank",
                text,
                re.I,
            )
        )

    @staticmethod
    def _is_show_shortlist(text: str) -> bool:
        return bool(
            re.search(
                r"\b(?:show|list|display|give)\b.*\b(?:top\s*10|shortlist|candidates|matches)\b"
                r"|\btop\s*10\b",
                text,
                re.I,
            )
        )

    @staticmethod
    def _is_ranking_change_question(text: str) -> bool:
        return bool(
            re.search(
                r"\b(?:what|show|explain)\b.{0,30}\b(?:changed|changes)\b"
                r"|\branking\s+changes?\b"
                r"|\bwhy\b.{0,40}\branking\b.{0,20}\bchange",
                text,
                re.I,
            )
        )

    @staticmethod
    def _looks_like_refinement(text: str) -> bool:
        lower = text.casefold()
        skill_words = re.search(
            r"\b(?:with|having|contains?|include(?:s)?|who\s+have)\b",
            lower,
        )
        action = re.search(
            r"\b(?:add|remove|exclude|drop|without|prefer|preferred|require|required|mandatory|essential)\b",
            lower,
        )
        threshold = re.search(
            r"\b(?:at\s+least|minimum|min\.?|more\s+than|less\s+than)\s+\d+(?:\.\d+)?\s*years?\b",
            lower,
        )
        only_filter = re.search(r"\bonly\b", lower) and bool(skill_words)
        return bool(action or threshold or only_filter)

    @staticmethod
    def _extract_candidate_reference(
        text: str,
        candidates: list[dict[str, Any]],
    ) -> str:
        lowered = text.casefold()
        if re.search(r"\btop\s+(?:candidate|match)\b", lowered):
            return str(candidates[0].get("candidate_id", ""))
        if re.search(r"\bsecond\s+(?:candidate|match)\b", lowered) and len(candidates) > 1:
            return str(candidates[1].get("candidate_id", ""))
        ranks = re.findall(r"#\s*(\d+)|\b(?:candidate|match)\s*(\d+)\b", lowered)
        for first, second in ranks:
            number = int(first or second)
            if 1 <= number <= len(candidates):
                return str(candidates[number - 1].get("candidate_id", ""))
        for candidate in candidates:
            name = str(candidate.get("candidate_name", "")).strip()
            if name and name.casefold() in lowered:
                return str(candidate.get("candidate_id", name))
        return str(candidates[0].get("candidate_id", "")) if candidates else ""

    @staticmethod
    def _selected_candidates(
        text: str,
        candidates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        lowered = text.casefold()
        selected: list[dict[str, Any]] = []

        # Explicit rank references (#1/#2, candidate 1/candidate 2).
        numbers: list[int] = []
        for match in re.finditer(r"#\s*(\d+)|\b(?:candidate|match)\s*(\d+)\b", lowered):
            numbers.append(int(match.group(1) or match.group(2)))
        for number in numbers:
            if 1 <= number <= len(candidates):
                item = candidates[number - 1]
                if item not in selected:
                    selected.append(item)

        if re.search(r"\btop\s*(?:3|three)\b", lowered):
            selected.extend(item for item in candidates[:3] if item not in selected)

        # Explicit names.
        for candidate in candidates:
            name = str(candidate.get("candidate_name", "")).strip()
            if name and name.casefold() in lowered and candidate not in selected:
                selected.append(candidate)

        return selected[:3] or candidates[:3]

    def _format_comparison(
        self,
        text: str,
        candidates: list[dict[str, Any]],
    ) -> str:
        selected = self._selected_candidates(text, candidates)
        if len(selected) < 3 and re.search(r"top\s*(?:3|three)", text, re.I):
            selected = candidates[:3]
        comparison = compare_candidates(
            [self._candidate_key(item) for item in selected],
            candidates,
        )
        lines = [
            "### Candidate Comparison",
            "",
            "| Rank | Candidate | Score | Experience | Strengths | Gaps |",
            "|---:|---|---:|---:|---|---|",
        ]
        for rank, item in enumerate(comparison.get("candidates", [])[:3], 1):
            strengths = self._join(item.get("strengths", []))
            gaps = self._join(item.get("gaps", []))
            experience = item.get("experience_years", 0)
            lines.append(
                f"| {rank} | {item.get('name', 'Unknown')} | "
                f"{float(item.get('score', 0) or 0):.2f} | {experience} | "
                f"{strengths} | {gaps} |"
            )
        if comparison.get("winner"):
            lines.extend(["", f"**Highest current score:** {comparison['winner']}"])
        if comparison.get("unresolved"):
            lines.extend(
                [
                    "",
                    "I could not identify: "
                    + ", ".join(comparison["unresolved"]),
                ]
            )
        return "\n".join(lines)

    @staticmethod
    def _explain_comparison(
        text: str,
        candidates: list[dict[str, Any]],
    ) -> str:
        selected = MatchingAgent._selected_candidates(text, candidates)
        if len(selected) < 2:
            return (
                "I need two identifiable candidates to explain the ranking difference. "
                "Use names or ranks, for example: `Why did candidate #1 rank higher than #2?`"
            )

        first, second = selected[:2]
        first_score = float(first.get("match_score", 0) or 0)
        second_score = float(second.get("match_score", 0) or 0)
        delta = abs(first_score - second_score)
        if first_score >= second_score:
            higher, lower = first, second
        else:
            higher, lower = second, first

        return "\n".join(
            [
                "### Ranking Explanation",
                "",
                f"**{higher.get('candidate_name')}** ranks above "
                f"**{lower.get('candidate_name')}** by **{delta:.2f} points** "
                f"({float(higher.get('match_score', 0) or 0):.2f} vs "
                f"{float(lower.get('match_score', 0) or 0):.2f}).",
                "",
                f"- **{higher.get('candidate_name')}** — "
                f"{float(higher.get('hard_coverage', 0) or 0):.0f}% hard-requirement coverage; "
                f"{higher.get('experience_years', 0)} years experience.",
                f"- **{lower.get('candidate_name')}** — "
                f"{float(lower.get('hard_coverage', 0) or 0):.0f}% hard-requirement coverage; "
                f"{lower.get('experience_years', 0)} years experience.",
                f"- **Score components ({higher.get('candidate_name')}):** "
                f"RAG {float(higher.get('base_score', 0) or 0):.2f}"
                f"{(' → evidence fallback ' + format(float(higher.get('fallback_evidence_score', 0) or 0), '.2f')) if higher.get('rag_score_was_tied') else ''} + "
                f"preferred {float(higher.get('nice_to_have_bonus', 0) or 0):.2f}.",
                f"- **Score components ({lower.get('candidate_name')}):** "
                f"RAG {float(lower.get('base_score', 0) or 0):.2f}"
                f"{(' → evidence fallback ' + format(float(lower.get('fallback_evidence_score', 0) or 0), '.2f')) if lower.get('rag_score_was_tied') else ''} + "
                f"preferred {float(lower.get('nice_to_have_bonus', 0) or 0):.2f}.",
                f"- **Higher candidate strengths:** "
                f"{MatchingAgent._join(higher.get('strengths', []))}.",
                f"- **Lower candidate gaps:** "
                f"{MatchingAgent._join(lower.get('gaps', []))}.",
            ]
        )

    @staticmethod
    def _format_named_candidate(
        candidate: dict[str, Any],
        heading: str,
    ) -> str:
        return "\n".join(
            [
                f"### {heading}",
                "",
                f"**{candidate.get('candidate_name', 'Unknown')}** — "
                f"**{float(candidate.get('match_score', 0) or 0):.2f}/100** · "
                f"{float(candidate.get('experience_years', 0) or 0):g} years",
                f"- **Strengths:** {MatchingAgent._join(candidate.get('strengths', []))}",
                f"- **Gaps:** {MatchingAgent._join(candidate.get('gaps', []))}",
                f"- **Why:** {candidate.get('reasoning', 'No additional rationale available.')}",
            ]
        )

    @staticmethod
    def _format_shortlist(
        candidates: list[dict[str, Any]],
        heading: str = "Current Top Candidates",
    ) -> str:
        lines = [f"### {heading}", ""]
        for rank, candidate in enumerate(candidates[:TOP_K_INITIAL], 1):
            lines.append(
                f"**{rank}. {candidate.get('candidate_name', 'Unknown')}** — "
                f"**{float(candidate.get('match_score', 0) or 0):.2f}/100** · "
                f"{float(candidate.get('experience_years', 0) or 0):g} years · "
                f"{str(candidate.get('screening_status', 'unknown')).title()}"
            )
        return "\n".join(lines)

    @staticmethod
    def _join(values: Any, fallback: str = "none") -> str:
        if not values:
            return fallback
        return ", ".join(map(str, values))

    @staticmethod
    def _multi_round_report(state: AgentState) -> str:
        lines = [
            "### Multi-Round Candidate Review",
            "",
            "**Round 1:** Initial screening of the 100-resume pool",
            "**Round 2:** Detailed review of the top 10 candidates",
            "**Round 3:** Final hiring recommendation",
            "",
            "#### Final Recommendations",
            "",
        ]
        for index, result in enumerate(
            state.get("final_recommendations", []),
            1,
        ):
            lines.extend(
                [
                    f"**{index}. {result['candidate_name']}** — "
                    f"**{result['score']:.2f}/100** · **{result['recommendation']}**",
                    f"- {result['reason']}",
                    "",
                ]
            )
        return "\n".join(lines)

    @staticmethod
    def _last_user_message(messages: list[BaseMessage]) -> str:
        for message in reversed(messages):
            if isinstance(message, HumanMessage):
                return str(message.content)
        return ""


def cli() -> None:
    parser = argparse.ArgumentParser(
        description="Agentic Profile Matching — LangGraph recruiter agent"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--job-file", help="Job-description text file")
    group.add_argument("--job-text", help="Job description or recruiter query")
    parser.add_argument("--persist-dir", default="chroma_db")
    parser.add_argument("--provider", choices=["hf", "openai"], default="hf")
    parser.add_argument("--multi-round", action="store_true")
    parser.add_argument("--resume-dir", default="resumes")
    args = parser.parse_args()

    try:
        job_text = (
            Path(args.job_file).read_text(encoding="utf-8")
            if args.job_file
            else args.job_text
        )
    except OSError as exc:
        parser.error(f"Unable to read job file: {exc}")

    try:
        agent = MatchingAgent(
            persist_dir=args.persist_dir,
            provider=args.provider,
            resume_dir=args.resume_dir,
        )
        state = (
            agent.run_multi_round(job_text)
            if args.multi_round
            else agent.run_job(job_text)
        )
    except Exception:
        parser.exit(
            1,
            "Agent initialization/execution failed. Check your environment, "
            "ChromaDB index, and Python traceback.\n",
        )

    print(state.get("report", "No report generated."))
    if args.multi_round:
        return

    print(
        "\nInteractive commands: compare top 3 | why did #1 rank higher than #2 "
        "| add AWS as a must-have | exclude Java | lowest score | "
        "interview questions for the top candidate | exit"
    )
    while True:
        try:
            prompt = input("\nRecruiter> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if prompt.casefold() in {"exit", "quit"}:
            return
        try:
            state = agent.invoke(prompt, state)
            print("\n" + state.get("report", "No response."))
        except Exception:
            print("\nAgent error. Check the traceback and application logs for details.")


if __name__ == "__main__":
    cli()
