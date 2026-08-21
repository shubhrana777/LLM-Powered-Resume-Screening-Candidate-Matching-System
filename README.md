# LLM-Powered Resume Screening & Candidate Matching System

A system that ranks candidates against a job description. It is built in phases;
**Phases 1 to 3 are complete**.

| Phase | Scope | Status |
| --- | --- | --- |
| 1 | PDF resume ingestion and text extraction | Done |
| 2 | Semantic matching with embeddings and FAISS | Done |
| 3 | Skill extraction and candidate analysis | Done |
| 4+ | LLM/RAG, API, dashboard, Docker | Not started |

Explicitly **not** implemented yet: LLM calls, LangChain, RAG, FastAPI,
Streamlit, Docker, databases, authentication, and OCR for scanned PDFs.

## Phase 1 scope — resume parsing

- Accept a path to a PDF resume.
- Open the file safely and extract text from every page.
- Normalize whitespace and drop empty lines.
- Fail with clear messages for missing files, non-PDF files, corrupted PDFs,
  and PDFs with no selectable text.

## Phase 2 scope — semantic matching

- Embed resumes and job descriptions with Sentence Transformers.
- Index resume vectors in FAISS.
- Rank candidates against a job description by cosine similarity.
- Return `candidate_id`, `candidate_name`, `similarity_score`, and `rank`.

## Phase 3 scope — skill and candidate analysis

- Extract skills from resumes and job descriptions against a configurable taxonomy.
- Report matched and missing skills.
- Extract stated years of experience, conservatively.
- Extract common degree formats.
- Build a structured `CandidateProfile` per candidate.
- Compare candidate experience against an explicitly stated requirement.

All Phase 3 extraction is deterministic string matching. No model is involved.

## Project structure

```
resume-screening-ai/
│
├── app/
│   ├── __init__.py
│   ├── resume_parser.py     # Phase 1: PDF validation, extraction, normalization
│   ├── embeddings.py        # Phase 2: Sentence Transformers wrapper
│   ├── vector_store.py      # Phase 2: FAISS index + metadata
│   ├── matching.py          # Phase 2: ranking engine
│   ├── skill_taxonomy.py    # Phase 3: configurable skill vocabulary
│   ├── skill_extractor.py   # Phase 3: skill extraction and comparison
│   ├── experience_extractor.py  # Phase 3: stated years of experience
│   ├── education_extractor.py   # Phase 3: degree extraction
│   ├── candidate_analyzer.py    # Phase 3: builds CandidateProfile records
│   ├── models.py            # Typed records shared across phases
│   └── main.py              # CLI: `extract`, `match`, `analyze` subcommands
│
├── data/
│   ├── resumes/             # Put your PDFs here (git-ignored)
│   └── job_descriptions/    # Sample job descriptions (committed, fictional)
│
├── scripts/
│   └── generate_sample_data.py   # Writes synthetic sample resumes
│
├── tests/
│   ├── conftest.py          # Fixtures + offline FakeEmbedder
│   ├── test_resume_parser.py
│   ├── test_main.py
│   ├── test_embeddings.py
│   ├── test_vector_store.py
│   ├── test_matching.py
│   ├── test_skill_taxonomy.py
│   ├── test_skill_extractor.py
│   ├── test_experience_extractor.py
│   ├── test_education_extractor.py
│   ├── test_candidate_analyzer.py
│   ├── test_integration.py
│   └── test_integration_phase3.py
│
├── .gitignore
├── .env.example
├── pytest.ini
├── README.md
└── requirements.txt
```

## Installation

Requires Python 3.10 or newer.

### 1. Create a virtual environment

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

> **Windows path-length note.** Phase 2 pulls in PyTorch, which ships files
> nested deeply enough to exceed the legacy 260-character `MAX_PATH` limit. If
> `pip install` fails with `[WinError 206] The filename or extension is too
> long`, either enable long paths
> (`HKLM\SYSTEM\CurrentControlSet\Control\FileSystem\LongPathsEnabled = 1`,
> needs admin) or create the virtual environment somewhere short, such as
> `python -m venv C:\venvs\rsai`.

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

This installs PyTorch and roughly 1.5 GB of packages. The first `match` run also
downloads the embedding model (~90 MB) into the Hugging Face cache
(`~/.cache/huggingface`); later runs are offline.

### 3. Optional environment file

No secrets are needed yet, but the template is there for later phases:

```bash
cp .env.example .env      # Windows: copy .env.example .env
```

## Sample data

Generate a set of fictional resumes to try things out:

```bash
python scripts/generate_sample_data.py
```

That writes nine invented candidates into `data/resumes/` — six technical
profiles plus three finance ones (a strong, a partial and a poor match) used to
demonstrate Phase 3. The folder is
git-ignored (except `.gitkeep`), so neither the samples nor any real candidate
resume ever reaches version control. Sample job descriptions live in
`data/job_descriptions/` and are committed, since they contain no personal data.

## Phase 1 — extracting text from one resume

```bash
python -m app.main extract data/resumes/priya_sharma.pdf
```

The Phase 1 form without a subcommand still works and means the same thing:

```bash
python -m app.main data/resumes/priya_sharma.pdf
```

Write the result to a file instead of the terminal:

```bash
python -m app.main extract data/resumes/priya_sharma.pdf -o output/priya.txt
```

### Expected output

```
$ python -m app.main extract data/resumes/priya_sharma.pdf
Priya Sharma
Senior Python Backend Engineer
priya.sharma@example.com | Bengaluru, India
SUMMARY
Backend engineer with 8 years building scalable Python microservices
and REST APIs on AWS. Strong focus on distributed systems and
database performance.
...
```

Blank lines are stripped: PDFs encode visual spacing as coordinates, not as
empty text lines, so the normalizer has nothing to preserve there. Section
headings still land on their own lines.

## Phase 2 — ranking candidates against a job description

```bash
python -m app.main match --job-description data/job_descriptions/backend_engineer.txt
```

Useful options:

```bash
# Inline job description instead of a file
python -m app.main match --job-text "Senior Python backend engineer, FastAPI and PostgreSQL"

# A different resume folder, top 3 only
python -m app.main match -r data/resumes -j data/job_descriptions/ml_engineer.txt -k 3
```

### Example output

```
$ python -m app.main match --job-description data/job_descriptions/backend_engineer.txt
Embedding 6 resume(s) with sentence-transformers/all-MiniLM-L6-v2 ...
Rank  Candidate        Similarity
----  ---------------  ----------
   1  Priya Sharma         0.7851
   2  David Kim            0.6705
   3  Marcus Chen          0.4753
   4  Elena Rodriguez      0.3991
   5  Aisha Okafor         0.3540
   6  Tom Baker            0.1528
```

Priya (Python backend) tops a backend role and Tom (pastry chef) sits last.
Swap in `ml_engineer.txt` and Marcus Chen moves to rank 1 — the ranking follows
the job description, which is the whole point of matching on meaning.

## How Phase 2 works

### What "semantic matching" means

Keyword matching asks *does this resume contain the word "Python"?* It misses a
resume that says "Django" but never spells out "backend", and it is fooled by a
resume that lists a keyword once in passing.

Semantic matching compares *meaning* instead. Each document becomes a point in a
high-dimensional space, positioned so that texts about similar things land near
each other. "Built REST APIs in Django" ends up close to "Python backend
development" even with no shared words.

### How embeddings work, briefly

An embedding model reads text and outputs a fixed-length vector of numbers — 384
of them for the model used here. The model was trained so that sentence pairs
meaning the same thing produce vectors pointing in similar directions.
"Direction" is the operative word: similarity is measured as the **cosine of the
angle** between two vectors, which ignores length and captures only orientation.

### Why Sentence Transformers

Raw transformer models such as BERT produce one vector per token, and naively
averaging them compares poorly. Sentence Transformers models are fine-tuned
specifically so that a single vector represents a whole passage and cosine
similarity between two such vectors is meaningful.

The default is `sentence-transformers/all-MiniLM-L6-v2`: 384 dimensions, ~90 MB,
fast enough on CPU for local development, and a well-established general-purpose
baseline. Override it with `--model`.

### Why FAISS

FAISS is a similarity-search library for dense vectors. This project uses
`IndexFlatIP`, an exact inner-product index.

Because the embeddings are **L2-normalized when generated**, the inner product
between two vectors *is* their cosine similarity. That means the number FAISS
returns is used directly as the score — **no rescaling or transformation is
applied**. Exact search is the right default at this scale: for hundreds or
thousands of resumes it is fast, needs no training step, and unlike the
approximate indexes (IVF, HNSW) it never misses a relevant candidate. The index
is in-memory only; persistence is deliberately left for a later phase.

### How ranking works

1. Every resume PDF in the folder is parsed to text (Phase 1).
2. All resume texts are embedded in **one batched call**.
3. The vectors go into a FAISS index, each carrying its `Candidate` as metadata.
4. The job description is embedded with the same model.
5. FAISS returns the nearest resume vectors, sorted by descending similarity.
6. Each hit becomes a `MatchResult` with a 1-based `rank`.

### What the similarity score means — and does not

`similarity_score` is the cosine similarity between the job-description
embedding and the resume embedding.

- **Range.** Mathematically `[-1.0, 1.0]`. In practice this model maps English
  prose into a narrow cone, so on the sample data a strong match scores about
  `0.70–0.80`, a loosely related profile `0.35–0.50`, and an unrelated one
  around `0.15`.
- **It is a semantic similarity score.** It measures how close two pieces of
  text are in the embedding space.
- **It is not a probability of being hired**, not a percentage of requirements
  met, and not a measure of candidate quality. It is deliberately reported as a
  raw score rather than dressed up as a percentage, because scaling it to
  "78% match" would imply a statistical meaning it does not have.
- **Only relative ordering within one run is meaningful.** Absolute values are
  not comparable across different job descriptions.

## Phase 3 — analysing candidates

```bash
python -m app.main analyze --job-description data/job_descriptions/financial_analyst.txt
```

Same options as `match`, plus `--show-extra-skills` to list skills the job did
not ask for:

```bash
python -m app.main analyze -j data/job_descriptions/financial_analyst.txt -k 3
python -m app.main analyze -t "Financial analyst, 3+ years, Python and SQL" --show-extra-skills
```

### Example output

```
JOB REQUIREMENTS
  Required Skills: Python, SQL, Excel, Power BI, Tableau, Data Analysis, ...
  Minimum Experience: 3 years

CANDIDATES

2. Sarah Wilson
  Semantic Match Score: 0.5940
  Matched Skills:
    - Python
    - SQL
    - Excel
    - Power BI
    - Tableau
    ...
  Missing Skills:
    - Investment Analysis
  Skill Coverage: 12/13 required skills (92%)
  Experience:
    Candidate: 4 years
    Required:  3 years
    Requirement Met: Yes
  Education:
    MBA - Finance
    Bachelor of Commerce - Accounting

3. James Patel
  Semantic Match Score: 0.4416
  Skill Coverage: 4/13 required skills (31%)
  Experience:
    Candidate: 2 years
    Required:  3 years
    Requirement Met: No

8. Nina Volkov
  Semantic Match Score: 0.1491
  Matched Skills: none
  Skill Coverage: 0/13 required skills (0%)
  Experience:
    Candidate: not stated on resume
    Required:  3 years
    Requirement Met: Unknown
      (unknown is reported as unknown, not as a pass or a fail)
```

Note that Sarah is ranked **2**, not 1, by semantic similarity, yet covers 92% of
the required skills and clears the experience bar. This is exactly why Phase 3
exists: a single similarity number cannot tell you *which* requirements a
candidate meets, and the candidate the embedding likes most is not always the
one who best fits the stated requirements. The two views are complementary, and
both are shown.

## How Phase 3 works

### Skill extraction

Extraction is exact matching against a fixed vocabulary — no model, no
statistics. The same resume always yields the same skills, and every hit can be
traced to the substring that produced it (`SkillExtractor.find_mentions` returns
the matched text and its offsets).

Naive substring matching would be badly wrong here. `"SQL" in text` reports SQL
for *PostgreSQL*, *MySQL* and *sqlalchemy*; `"R"` reports R for every word
containing the letter. Each alias is therefore compiled into a regex with custom
boundaries:

| Guard | Effect |
| --- | --- |
| `(?<![A-Za-z0-9+#&])` before | *MySQL* does not yield SQL; *R&D* does not yield R |
| `(?![A-Za-z0-9+#&])` after | *C++* does not yield C; *sqlalchemy* does not yield SQL |
| `(?!\.\w)` after | *Node.js* does not yield Node, while *"Python."* still yields Python |

`\b` alone cannot do this: it is defined in terms of `\w`, so it misbehaves
around `+`, `#` and `.` — precisely the characters real skill names contain.

Two further rules:

- Matching is case-insensitive, **except single-character skills** (`C`, `R`),
  which must be capitalised. Otherwise a bullet like `c) managed the team` would
  register the C language.
- Spaces, hyphens and underscores inside an alias match any run of the same, so
  `Power BI` also matches `power-bi`, `powerbi`, and a `Power` / `BI` line break
  — which PDF extraction produces often.

### Configurable skill taxonomy

Every recognised skill lives in `app/skill_taxonomy.py`, never inline in the
extraction code. A skill has a canonical name, a category, and any number of
aliases. The canonical name is what gets reported, whichever alias matched, so a
resume saying `k8s` satisfies a job asking for `Kubernetes`.

Categories shipped: Programming, Data, AI/ML, Cloud/DevOps, Backend, Finance.

Three ways to extend it, in increasing order of separation from the code:

```python
# 1. Add an entry to DEFAULT_SKILL_DEFINITIONS in the module.

# 2. Extend at runtime; the original is left unchanged.
from app.skill_taxonomy import DEFAULT_TAXONOMY, SkillDefinition
taxonomy = DEFAULT_TAXONOMY.extended([SkillDefinition("Rust", "Programming", ("rustlang",))])

# 3. Load from JSON, keeping the vocabulary out of the codebase entirely.
from app.skill_taxonomy import SkillTaxonomy
taxonomy = SkillTaxonomy.from_json_file("my_skills.json")
# JSON shape: {"Category": {"Skill Name": ["alias", ...]}}
```

Pass a taxonomy to `SkillExtractor(taxonomy)` and the rest of the pipeline
follows it.

### Matched and missing skills

```
required_skills ─┐
                 ├─► matched_skills     (required AND candidate)
candidate_skills ┘   missing_skills     (required NOT candidate)
                     additional_skills  (candidate NOT required)
```

Comparison is by canonical name and case-insensitive. Ordering is deterministic:
`matched` and `missing` follow the order of the required list, `additional`
follows the candidate order. `SkillComparison.coverage` is a plain count ratio,
`len(matched) / required_count` — not a probability, not weighted, and `None`
when the job named no recognised skills.

### Experience extraction

Deliberately conservative: a number is returned only when the text states it in
words, near "experience".

| Input | Result |
| --- | --- |
| `4 years of experience` | `4.0` |
| `3+ years experience` | `3.0` |
| `2.5 years of professional experience` | `2.5` |
| `3-5 years of experience` | `3.0` (lower bound) |
| `Experience: 5 years` | `5.0` |
| `Graduated in 2015` | `None` |
| `Acme Corp (2020-2024)` | `None` |
| `Senior Engineer` | `None` |
| `8 years building Python services` | `None` (no explicit "experience") |

Nothing is inferred from graduation years, employment dates, or seniority words,
and employment periods are never summed. When a resume states several figures the
**largest** is taken as the overall total; when a job description states several,
the **smallest** is taken as the bar to clear.

### Education extraction

Line-by-line matching of known degree spellings — `B.S.`, `BSc`, `B.Tech`,
`B.E.`, `B.Com`, `M.S.`, `MSc`, `M.Tech`, `MBA`, `PhD`, `Ph.D.`, `Diploma`, and
the spelled-out forms — plus the field of study when the same line names one.

Bare `BS` and `MS` are deliberately **not** recognised, because "MS Excel" and
"MS Office" appear on far more resumes than "MS Physics".

### Candidate profiles

`CandidateProfile` is a frozen dataclass carrying `candidate_id`,
`candidate_name`, `skills`, `years_experience`, `education`, `matched_skills`,
`missing_skills`, `additional_skills`, `semantic_match_score`, `rank`,
`required_experience`, `meets_experience_requirement` and `source_path`.

### Experience comparison

| Required | Candidate | `meets_experience_requirement` |
| --- | --- | --- |
| 3.0 | 4.0 | `True` |
| 3.0 | 2.0 | `False` |
| 3.0 | `None` | `None` |
| `None` | 4.0 | `None` |

Unknown is never resolved into a pass or a fail. A resume that simply does not
state its years is not a resume that fails the requirement.

### Using Phase 3 as a library

```python
from pathlib import Path

from app.candidate_analyzer import analyze_candidates_for_job
from app.matching import load_candidates_from_directory

loaded = load_candidates_from_directory("data/resumes")
job_text = Path("data/job_descriptions/financial_analyst.txt").read_text(encoding="utf-8")

requirements, profiles = analyze_candidates_for_job(loaded.candidates, job_text)

for profile in profiles:
    print(profile.rank, profile.display_name)
    print("  matched:", profile.matched_skills)
    print("  missing:", profile.missing_skills)
    print("  experience:", profile.years_experience,
          "meets:", profile.meets_experience_requirement)
```

Or use the pieces on their own:

```python
from app.education_extractor import extract_education
from app.experience_extractor import extract_years_of_experience
from app.skill_extractor import compare_skills, extract_skills

extract_skills("Python, SQL and Power BI for forecasting")
# ('Python', 'SQL', 'Power BI', 'Forecasting')

compare_skills(["Python", "SQL", "Tableau"], ["Python", "SQL"]).missing
# ('Tableau',)

extract_years_of_experience("4 years of experience")   # 4.0
extract_education("MBA in Finance")                    # (EducationEntry('MBA', 'Finance', ...),)
```

## Using the engine as a library

```python
from app.matching import load_candidates_from_directory, rank_candidates

loaded = load_candidates_from_directory("data/resumes")
for failure in loaded.failures:
    print(f"skipped {failure.path.name}: {failure.reason}")

results = rank_candidates(loaded.candidates, "Senior Python backend engineer")
for result in results:
    print(result.rank, result.display_name, round(result.similarity_score, 4))
```

To score one set of resumes against several job descriptions, index once:

```python
from app.matching import CandidateMatcher

matcher = CandidateMatcher()
matcher.index_candidates(loaded.candidates)          # embeds resumes once
backend = matcher.match("Senior Python backend engineer")
ml_role = matcher.match("NLP engineer, PyTorch and embeddings")
```

## Running the tests

```bash
pytest                    # everything
pytest -m "not model"     # offline only, no model download
pytest -m model           # only the tests that use real model weights
```

Most tests run against a deterministic bag-of-words `FakeEmbedder` defined in
`tests/conftest.py`, so the suite is fast, offline, and reproducible. Depending
on real transformer weights everywhere would make it slow and network-bound, and
floating-point output can shift between model revisions. The fake still produces
genuine cosine similarities — texts sharing vocabulary score higher — so the
ranking assertions test real logic.

Tests that must exercise the actual model are marked `model` and **skip
automatically** when the weights cannot be loaded, so the suite still passes on a
machine with no network access.

Test PDFs are generated at runtime with PyMuPDF, so no binary fixtures are
committed. The Phase 3 suites need neither a model nor a network: skill,
experience and education extraction are pure string matching, so they are
tested directly.

## Limitations

### Resume parsing

- Scanned/image-only PDFs yield no text; OCR is out of scope.
- Multi-column layouts extract in PyMuPDF's reading order, which can interleave
  columns.
- Only `.pdf` is supported; DOCX and plain text are not handled.

### Embedding model

- **Input truncation is the most important limitation here.**
  `all-MiniLM-L6-v2` has `max_seq_length = 256` word pieces — roughly 200 words.
  Anything beyond that is silently discarded: two resumes sharing their first
  256 tokens but ending completely differently embed to the *same* vector
  (measured cosine 1.000). A typical one-page resume is 400–800 tokens, so
  **the latter half of a real resume may not influence its score at all** —
  often the skills and education sections. Chunking long resumes, or moving to
  a longer-context embedding model, is the obvious next improvement.
- 384 dimensions is small. The model is a fast general-purpose baseline, not a
  recruitment-domain model, and it has no notion of seniority, recency, or how
  long a skill was used.
- It compares whole documents. A resume that mentions a required skill once
  scores much like one built around that skill, and the score cannot explain
  *which* requirements were met — that is Phase 3 (skill analysis).
- Longer documents tend to score slightly lower simply by covering more topics,
  so resume length is a mild confound.
- English only in practice.

### Skill extraction (Phase 3)

Extraction is exact matching against a fixed list. It does not understand
resumes; it recognises strings. The consequences are worth stating plainly:

- **Only taxonomy skills are ever found.** A skill not in
  `app/skill_taxonomy.py` does not exist as far as the system is concerned. A
  candidate whose strongest skill is missing from the vocabulary is silently
  under-reported, which is the most likely way this stage misjudges someone.
- **No synonym or morphology handling beyond declared aliases.** "Forecasting"
  is recognised because it is declared; "forecasted revenue monthly" matches via
  the `forecast` alias, but an undeclared phrasing simply will not match.
- **No context, negation, or proficiency.** "Familiar with Python", "learning
  Python", "Python (beginner)" and "8 years of Python" all yield the same bare
  `Python`. So does "we are replacing our Python stack". The extractor cannot
  tell a skill someone *has* from a skill merely *mentioned*.
- **No section awareness.** A skill in a job description's "Nice to have"
  section is reported as required, exactly like one under "Requirements". In the
  bundled sample this is visible: `Investment Analysis` appears in the required
  list although the posting lists it as nice-to-have.
- **Job titles can register as skills.** "Financial analyst" yields the skill
  `Financial Analysis` via an alias. Usually right, occasionally not.
- **Bare `C` and `R` remain the riskiest entries.** The boundary and
  capitalisation rules block the common false positives (`c)`, `R&D`, `rapport`,
  `C.S.`), but an isolated capital `C` or `R` in some other context would still
  register. Drop them from the taxonomy if that trade-off does not suit you.
- **Skill coverage is a flat count.** Every skill counts equally; a missing core
  requirement weighs the same as a missing nice-to-have.

### Experience extraction (Phase 3)

- Only explicit statements near the word "experience" are read. A resume listing
  a decade of dated roles but never writing "N years of experience" yields
  `None` — correct behaviour by design, but it means **`None` is common on real
  resumes**, and unknown must never be read as zero.
- Where several figures appear, the largest is taken for a candidate and the
  smallest for a job posting. A resume saying "3 years of experience with
  Kubernetes" and nothing else yields `3.0` as its overall total, which
  overstates a specific figure as a general one.
- No domain awareness: it cannot distinguish total career experience from
  experience relevant to the role.
- Values above 60 years, and fragments of longer numbers, are rejected.

### Education extraction (Phase 3)

- Degree and field must appear on the same line; nothing is stitched across line
  breaks, so a PDF that wraps awkwardly loses the field.
- Bare `BS`/`MS` are not recognised, so "BS Computer Science" is missed. This is
  a deliberate trade against "MS Excel" false positives.
- The field is taken as text following the degree, cut at a comma, bracket or
  year. Unusual layouts produce an odd field or none at all.
- No institution, no dates, no accreditation, no honours, no verification. The
  extractor keys on the word, not the context: "organised a bachelor party"
  registers a bachelor's degree.
- `extract_highest_degree` ranks by degree level only. It says nothing about
  quality or relevance.

### What Phase 3 does not do

There is no judgement of candidate quality here, and no hiring decision. The
output is a structured, checkable summary of what a resume *says*, next to a
semantic similarity score — intended to help a human review faster, not to
screen anyone out automatically. Every field is either extracted verbatim or
reported as unknown.

### Matching engine

- The FAISS index is in-memory and rebuilt on every run; nothing is persisted.
- Every resume is re-embedded on each CLI invocation. There is no embedding
  cache yet.
- Ranking is a single similarity number. There is no filtering on hard
  requirements (years of experience, location, visa status), and the system
  cannot justify a ranking.
- Semantic similarity reflects patterns in the model's training data and can
  carry those biases. Treat the output as a way to order resumes for human
  review, never as an automated screening decision.
