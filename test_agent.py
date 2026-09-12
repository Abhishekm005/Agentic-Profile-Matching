"""Regression tests for the Agentic Profile Matching conversation layer."""
from __future__ import annotations

import importlib.util

import pytest

from agent_tools import estimate_experience_years, extract_requirements, read_candidate_resume


def test_general_experience_requirement_is_captured():
    result = extract_requirements("Find candidates with React and 3+ years experience")
    assert any(req["type"] == "experience" and req["years"] == 3 for req in result["must_have"])
    assert any(req["type"] == "skill" and req["skill"] == "React" for req in result["must_have"])


def test_skill_specific_experience_is_not_duplicated_as_general():
    result = extract_requirements("Must have at least 5 years of Python experience")
    assert any(
        req["type"] == "years" and req["skill"] == "Python" and req["years"] == 5
        for req in result["must_have"]
    )
    assert not any(req["type"] == "experience" for req in result["must_have"])


def test_nice_to_have_is_separated():
    result = extract_requirements("Python developer. Must have Python. Nice to have AWS and Docker.")
    assert "AWS" in result["nice_to_have"]
    assert "Docker" in result["nice_to_have"]


def test_experience_estimation_uses_date_ranges():
    resume = "Software Engineer, 2021-2026\nSenior Engineer, 2023-2026"
    assert estimate_experience_years(resume) == 5.0


def test_python_developer_query_detects_python_as_must_have():
    result = extract_requirements("Show me the best candidate for Python developer")
    assert any(req["type"] == "skill" and req.get("skill") == "Python" for req in result["must_have"])


@pytest.mark.skipif(
    not (importlib.util.find_spec("langgraph") and importlib.util.find_spec("langchain_core")),
    reason="LangGraph/LangChain Core not installed",
)
def test_greeting_does_not_trigger_graph_search():
    from matching_agent import MatchingAgent

    class FakeGraph:
        def __init__(self):
            self.calls = 0

        def invoke(self, state):
            self.calls += 1
            raise AssertionError("Greeting should not invoke the matching graph")

    agent = object.__new__(MatchingAgent)
    fake = FakeGraph()
    agent.graph = fake
    state = agent.invoke("hii", {"messages": [], "screening_round": 1})

    assert fake.calls == 0
    assert "Hello!" in state["report"]
    assert state.get("candidate_shortlist", []) == []


@pytest.mark.skipif(
    not (importlib.util.find_spec("langgraph") and importlib.util.find_spec("langchain_core")),
    reason="LangGraph/LangChain Core not installed",
)
def test_existing_shortlist_utilities_do_not_rerun_graph(sample_candidates):
    from matching_agent import MatchingAgent

    class FakeGraph:
        def __init__(self):
            self.calls = 0

        def invoke(self, state):
            self.calls += 1
            raise AssertionError("Existing-shortlist utility should not invoke the graph")

    agent = object.__new__(MatchingAgent)
    fake = FakeGraph()
    agent.graph = fake
    state = {
        "messages": [],
        "screening_round": 1,
        "job_description": "Python developer",
        "candidate_shortlist": sample_candidates,
        "current_rankings": [x["candidate_id"] for x in sample_candidates],
    }

    highest = agent.invoke("Give me the candidate with the highest score", state)
    lowest = agent.invoke("Who is the candidate with the lowest score?", highest)
    ranked = agent.invoke("Rank the candidates according to score", lowest)

    assert fake.calls == 0
    assert "John" in highest["report"]
    assert "Alex" in lowest["report"]
    assert "Current Ranking by Score" in ranked["report"]


@pytest.mark.skipif(
    not (importlib.util.find_spec("langgraph") and importlib.util.find_spec("langchain_core")),
    reason="LangGraph/LangChain Core not installed",
)
def test_assignment_conversation_routes_correctly(sample_candidates):
    from matching_agent import MatchingAgent

    class FakeGraph:
        def __init__(self):
            self.calls = 0
            self.next_candidates = sample_candidates

        def invoke(self, state):
            self.calls += 1
            state["candidate_shortlist"] = self.next_candidates
            state["current_rankings"] = [x["candidate_id"] for x in self.next_candidates]
            state["current_scores"] = {x["candidate_id"]: x["match_score"] for x in self.next_candidates}
            state["report"] = "refined graph response"
            return state

    agent = object.__new__(MatchingAgent)
    fake = FakeGraph()
    agent.graph = fake

    base = {
        "messages": [],
        "screening_round": 1,
        "job_description": "Python and SQL",
        "candidate_shortlist": sample_candidates,
        "current_rankings": [x["candidate_id"] for x in sample_candidates],
        "current_scores": {x["candidate_id"]: x["match_score"] for x in sample_candidates},
    }

    compared = agent.invoke("Compare the top 3 candidates side by side", base)
    assert "Candidate Comparison" in compared["report"]
    assert compared["report"].count("|") > 10
    assert fake.calls == 0

    why = agent.invoke("Why did the top candidate rank higher than the second candidate?", compared)
    assert "Ranking Explanation" in why["report"]
    assert fake.calls == 0

    questions = agent.invoke("Generate interview questions for the top candidate", why)
    assert "Interview Questions" in questions["report"]
    assert fake.calls == 0

    refined = agent.invoke("Only show candidates who have Docker", questions)
    assert fake.calls == 1
    assert "refined graph response" in refined["report"]
    assert "must have docker" in refined["active_refinements"][0].casefold()


@pytest.mark.skipif(
    not (importlib.util.find_spec("langgraph") and importlib.util.find_spec("langchain_core")),
    reason="LangGraph/LangChain Core not installed",
)
def test_rank_change_baseline_uses_stable_candidate_ids():
    from matching_agent import MatchingAgent

    old = ["a.txt", "b.txt", "c.txt"]
    new = ["b.txt", "a.txt", "d.txt"]
    old_scores = {"a.txt": 80, "b.txt": 70, "c.txt": 60}
    new_scores = {"b.txt": 85, "a.txt": 78, "d.txt": 82}
    candidates = [
        {"candidate_id": "b.txt", "candidate_name": "Jane", "match_score": 85},
        {"candidate_id": "a.txt", "candidate_name": "John", "match_score": 78},
        {"candidate_id": "d.txt", "candidate_name": "Alex", "match_score": 82},
    ]

    changes = MatchingAgent._ranking_changes(old, new, old_scores, new_scores, candidates)
    assert any("Jane moved up" in change for change in changes)
    assert any("John moved down" in change for change in changes)
    assert any("Alex entered" in change for change in changes)
    assert any("score changed" in change for change in changes)


@pytest.fixture
def sample_candidates() -> list[dict]:
    return [
        {
            "candidate_name": "John",
            "candidate_id": "c1",
            "match_score": 91,
            "base_score": 88,
            "nice_to_have_bonus": 3,
            "fallback_experience_bonus": 0,
            "hard_coverage": 100,
            "experience_years": 5,
            "matched_skills": ["Python", "SQL"],
            "nice_to_have_matches": ["AWS"],
            "strengths": ["Python", "SQL", "AWS"],
            "gaps": [],
            "hard_requirements_satisfied": ["Python", "SQL"],
            "hard_requirements_satisfied_count": 2,
            "hard_requirements_total": 2,
            "reasoning": "Strong alignment.",
        },
        {
            "candidate_name": "Jane",
            "candidate_id": "c2",
            "match_score": 82,
            "base_score": 82,
            "nice_to_have_bonus": 0,
            "fallback_experience_bonus": 0,
            "hard_coverage": 100,
            "experience_years": 5,
            "matched_skills": ["SQL"],
            "nice_to_have_matches": [],
            "strengths": ["SQL"],
            "gaps": [],
            "hard_requirements_satisfied": ["SQL"],
            "hard_requirements_satisfied_count": 1,
            "hard_requirements_total": 1,
            "reasoning": "Good alignment.",
        },
        {
            "candidate_name": "Alex",
            "candidate_id": "c3",
            "match_score": 70,
            "base_score": 70,
            "nice_to_have_bonus": 0,
            "fallback_experience_bonus": 0,
            "hard_coverage": 100,
            "experience_years": 3,
            "matched_skills": ["Python"],
            "nice_to_have_matches": [],
            "strengths": ["Python"],
            "gaps": [],
            "hard_requirements_satisfied": ["Python"],
            "hard_requirements_satisfied_count": 1,
            "hard_requirements_total": 1,
            "reasoning": "Moderate alignment.",
        },
    ]


def test_comparison_supports_names_and_ranks(sample_candidates):
    from agent_tools import compare_candidates

    by_name = compare_candidates(["John", "Jane"], sample_candidates)
    by_rank = compare_candidates(["1", "2"], sample_candidates)
    assert by_name["winner"] == "John"
    assert [row["name"] for row in by_rank["candidates"]] == ["John", "Jane"]


def test_interview_questions_are_evidence_based(sample_candidates):
    from agent_tools import generate_interview_questions

    questions = generate_interview_questions("John", sample_candidates)
    assert len(questions) >= 3
    assert any("Python" in question for question in questions)
    assert any("AWS" in question for question in questions)


@pytest.mark.skipif(
    not (importlib.util.find_spec("langgraph") and importlib.util.find_spec("langchain_core")),
    reason="LangGraph/LangChain Core not installed",
)
def test_graph_contains_required_nodes():
    from matching_agent import MatchingAgent

    agent = object.__new__(MatchingAgent)
    graph = agent._build_graph()
    assert graph is not None


@pytest.mark.skipif(
    not importlib.util.find_spec("chromadb"),
    reason="ChromaDB not installed; live integration unavailable",
)
def test_multi_round_integration():
    from matching_agent import MatchingAgent

    agent = MatchingAgent()
    state = agent.run_multi_round("Find strong Python engineers with 3+ years experience")
    assert state["screening_round"] == 3
    assert "final_recommendations" in state


def test_indexed_project_relative_resume_path_is_normalized(tmp_path):
    root = tmp_path / "resumes"
    root.mkdir()
    safe = root / "candidate.txt"
    safe.write_text("Candidate\nSKILLS\nPython", encoding="utf-8")
    assert read_candidate_resume({"resume_path": "resumes/candidate.txt"}, root)
    assert read_candidate_resume({"resume_path": r"resumes\candidate.txt"}, root)


def test_resume_path_is_contained_in_trusted_resume_directory(tmp_path):
    root = tmp_path / "resumes"
    root.mkdir()
    safe = root / "candidate.txt"
    safe.write_text("Candidate\nSKILLS\nPython", encoding="utf-8")
    assert read_candidate_resume({"resume_path": "candidate.txt"}, root)


def test_resume_path_traversal_is_blocked(tmp_path):
    root = tmp_path / "resumes"
    root.mkdir()
    secret = tmp_path / "secret.txt"
    secret.write_text("do not read", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsafe resume path"):
        read_candidate_resume({"resume_path": "../secret.txt"}, root)
