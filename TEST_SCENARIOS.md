# Conversation Test Scenarios

These scenarios are the acceptance tests for the recruiter chat flow.

## 1. Initial search

**Input**

`Find me candidates with Python and 3+ years experience`

**Expected**

- Python is a hard requirement.
- 3+ years overall experience is a hard requirement.
- The agent screens the indexed 100-resume dataset and returns up to the top 10 matches.
- Each candidate shows score, experience, requirement coverage, strengths, gaps, and rationale.
- A candidate who has Python and at least 3 years must not be reported as `0/1` satisfied.

## 2. Highest score

**Input**

`Give me the candidate with the highest score`

**Expected**

Use the existing shortlist. Do not start a new RAG search.

## 3. Lowest score

**Input**

`Give me the candidate with the lowest score`

**Expected**

Use the existing shortlist. Do not replace the active job requirements with the sentence "lowest score".

## 4. Rank by score

**Input**

`Rank the candidates according to score`

**Expected**

Show the current shortlist ordered from highest to lowest score without running another search.

## 5. Top-3 comparison

**Input**

`Compare the top 3 candidates side by side`

**Expected**

Return exactly three candidates when three exist, in a comparison table with score, experience, strengths, and gaps.

## 6. Ranking explanation

**Input**

`Why did the top candidate rank higher than the second candidate?`

**Expected**

Compare the current #1 and #2 candidates and explain the actual score difference using score components, hard-requirement coverage, experience, strengths, and gaps.

## 7. Add requirement

**Input**

`Add Docker as a must-have`

**Expected**

Preserve the existing job criteria, add Docker as a hard requirement, rerun retrieval/filtering/ranking, and record ranking changes from the prior shortlist.

## 8. Filter wording

**Input**

`Only show candidates who have Docker`

**Expected**

Treat this as a recruiter refinement, not as a display command. Preserve the original requirements and add Docker as a hard criterion.

## 9. Exclude a skill

**Input**

`Exclude Java`

**Expected**

Add Java to the exclusion set and remove candidates whose authoritative resume metadata contains Java.

## 10. Ranking changes

**Input**

`What changed in the ranking after adding Docker?`

**Expected**

Report candidates who entered/left/moved and any score changes compared with the immediately preceding shortlist.

## 11. Interview questions

**Input**

`Generate interview questions for the top candidate`

**Expected**

Generate evidence-based questions for the current #1 candidate, without starting another candidate search.

## 12. Multi-round screening

Run:

```powershell
python matching_agent.py --job-file job_descriptions\jd_01_ml_engineer.txt --multi-round
```

**Expected**

- Round 1: initial screening from the 100-resume pool.
- Round 2: deep evidence review for the top 10.
- Round 3: HIRE / BORDERLINE / NO-HIRE recommendation.

## 13. Conversation sanity

**Input**

`hii`

**Expected**

Return a greeting only. No RAG search and no empty candidate report.
