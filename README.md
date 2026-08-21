# LLM-Powered Resume Screening & Candidate Matching System

A system that ranks candidates against a job description. It is built in phases;
**Phases 1 and 2 are complete**.

| Phase | Scope | Status |
| --- | --- | --- |
| 1 | PDF resume ingestion and text extraction | Done |
| 2 | Semantic matching with embeddings and FAISS | Done |
| 3+ | Skill analysis, LLM/RAG, API, dashboard, Docker | Not started |

Explicitly **not** implemented yet: LLM calls, LangChain, RAG, skill extraction,
FastAPI, Streamlit, Docker, databases, authentication, and OCR for scanned PDFs.

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
│   ├── models.py            # Phase 2: Candidate / MatchResult records
│   └── main.py              # CLI: `extract` and `match` subcommands
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
│   └── test_integration.py
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

That writes six invented candidates into `data/resumes/`. The folder is
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
committed.

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
