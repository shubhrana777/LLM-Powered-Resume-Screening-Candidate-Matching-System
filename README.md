# LLM-Powered Resume Screening & Candidate Matching System

A system that will eventually rank candidates against a job description using
embeddings and an LLM. Everything starts with reliable text: this repository is
being built in phases, and **Phase 1 covers PDF resume ingestion only**.

## Phase 1 scope

In scope:

- Accept a path to a PDF resume.
- Open the file safely and extract text from every page.
- Normalize whitespace and drop empty lines.
- Return clean, readable text.
- Fail with clear messages for missing files, non-PDF files, corrupted PDFs,
  and PDFs with no selectable text.
- A small command-line entry point and unit tests.

Explicitly **not** in Phase 1: embeddings, FAISS, Sentence Transformers,
LangChain, LLM calls, FastAPI, Streamlit, Docker, databases, candidate matching,
and OCR for scanned resumes.

## Project structure

```
resume-screening-ai/
│
├── app/
│   ├── __init__.py
│   ├── resume_parser.py     # PDF validation, extraction, text normalization
│   └── main.py              # Command-line entry point
│
├── data/
│   └── resumes/             # Put your PDFs here (git-ignored)
│
├── tests/
│   ├── conftest.py          # PDF fixtures generated at runtime
│   ├── test_resume_parser.py
│   └── test_main.py
│
├── .gitignore
├── .env.example
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

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Optional environment file

Phase 1 needs no secrets, but the template is there for later phases:

```bash
cp .env.example .env      # Windows: copy .env.example .env
```

## Adding a resume

Copy any text-based PDF resume into `data/resumes/`:

```
data/resumes/sample_resume.pdf
```

That folder is git-ignored (except for `.gitkeep`), so real candidate resumes
never end up in version control.

## Running the parser

```bash
python -m app.main data/resumes/sample_resume.pdf
```

Write the result to a file instead of the terminal:

```bash
python -m app.main data/resumes/sample_resume.pdf --output output/sample_resume.txt
```

### Expected output

```
$ python -m app.main data/resumes/sample_resume.pdf
Jane Doe
Senior Python Engineer
jane.doe@example.com | +1 555 0100 | San Francisco, CA
SUMMARY
Backend engineer with 6 years of experience building data-intensive
Python services and machine learning pipelines.
EXPERIENCE
Acme Corp - Backend Engineer (2020-2024)
Built data pipelines and REST APIs serving 2M requests/day.
EDUCATION
B.S. Computer Science, State University (2018)
SKILLS
Python, SQL, Docker, AWS, PostgreSQL, Git
```

Note that blank lines are stripped: PDFs encode visual spacing as coordinates,
not as empty text lines, so the normalizer has nothing to preserve there.
Section headings still land on their own lines, which is what downstream phases
will chunk on.

### Error handling

User errors print a single-line message to stderr and exit with status `1` --
never a traceback:

```
$ python -m app.main data/resumes/missing.pdf
Error: No such file: /.../data/resumes/missing.pdf

$ python -m app.main notes.txt
Error: Expected a .pdf file but got '.txt': /.../notes.txt

$ python -m app.main data/resumes/scanned.pdf
Error: No extractable text found in /.../scanned.pdf. The file may be a
scanned image; OCR is out of scope for Phase 1.
```

## Using the parser as a library

```python
from app.resume_parser import extract_text_from_pdf, ResumeParserError

try:
    text = extract_text_from_pdf("data/resumes/sample_resume.pdf")
except ResumeParserError as exc:
    print(f"Could not parse resume: {exc}")
```

## Running the tests

```bash
pytest -v
```

Test PDFs are generated at runtime with PyMuPDF, so no sample files are
committed and no extra PDF-writing dependency is required.

## Limitations

- Scanned/image-only PDFs yield no text. OCR is out of scope for Phase 1.
- Multi-column layouts are extracted in PyMuPDF's reading order, which may
  interleave columns. Layout-aware parsing is a later concern.
- Only `.pdf` is supported; DOCX and plain text are not handled yet.
