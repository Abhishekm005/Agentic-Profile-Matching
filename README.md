# Agentic Profile Matching

A LangGraph-based conversational recruiter agent built on the existing Milestone 2 RAG/ProfileMatcher implementation.

## Assignment coverage

### Part A — Agent Architecture

Core graph:

`START → Parse JD → Extract Requirements → Search Resumes → Rank Candidates → Generate Report → Human Feedback Loop → END`

State includes conversation history, active job requirements, refinements, exclusions, candidate shortlist, ranking history, and multi-round recommendations.

Tools:

- `extract_requirements(jd)`
- `compare_candidates(candidate_ids, candidates)`
- `generate_interview_questions(candidate_id, candidates)`

### Part B — Interactive recruiter conversation

Supported flows include:

```text
Find me candidates with React and 3+ years experience
Compare the top 3 candidates side by side
Why did the top candidate rank higher than the second candidate?
Give me the candidate with the highest score
Rank the candidates according to score
Add AWS as a must-have
Only show candidates who have Docker
Exclude Java
What changed in the ranking after adding Docker?
Generate interview questions for the top candidate
```

Follow-up utility commands operate on the current shortlist. Criteria refinements preserve the existing job context and trigger a new screening pass.

### Part C — Advanced screening

- Round 1: screen the 100-resume dataset and return the top 10.
- Round 2: deep evidence review of the top 10.
- Round 3: HIRE / BORDERLINE / NO-HIRE recommendation.

## Run

Build or refresh the ChromaDB index:

```powershell
python rag_profile_matching.py --resumes-dir resumes --persist-dir chroma_db
```

Start Streamlit:

```powershell
python -m streamlit run app.py
```

Run the CLI:

```powershell
python matching_agent.py --job-file job_descriptions\jd_01_ml_engineer.txt
```

Run a direct natural-language search:

```powershell
python matching_agent.py --job-text "Find me candidates with React and 3+ years experience"
```

Run the assignment's multi-round flow:

```powershell
python matching_agent.py --job-file job_descriptions\jd_01_ml_engineer.txt --multi-round
```

## Important environment note

The repository does not package a prebuilt Chroma database. Build the local index once on the machine used for the demo. The dataset contains 100 resume files and 5 job descriptions.

## Regression tests

Run:

```powershell
pytest -q
```

The live integration test is skipped automatically when ChromaDB is not installed or a live index is unavailable.
