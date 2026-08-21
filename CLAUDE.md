# CLAUDE.md

## Project Overview

This repository contains an end-to-end AI-powered resume screening and candidate matching platform.

The system is designed to help recruiters:
- Upload multiple resumes.
- Extract and process resume content.
- Compare resumes against job descriptions.
- Rank candidates using semantic similarity.
- Identify matched and missing skills.
- Generate AI-powered candidate summaries and recommendations.
- Search and retrieve relevant resume information using RAG.
- Use a web-based recruiter dashboard.

This is a portfolio project intended to demonstrate practical skills in:
- Python
- NLP
- Machine Learning
- Embeddings
- Vector Search
- LLMs
- RAG
- FastAPI
- Streamlit
- Docker
- Software Engineering
- Git/GitHub

---

## Main Technology Stack

Use the following technologies unless there is a clear technical reason to change them:

### Core
- Python 3.11+
- PyMuPDF for PDF text extraction

### NLP / Machine Learning
- Hugging Face
- Sentence Transformers
- FAISS
- Scikit-learn where appropriate

### Generative AI
- LangChain where it provides real value
- An LLM provider selected during implementation
- Structured LLM outputs where appropriate

### Backend
- FastAPI
- Pydantic

### Frontend
- Streamlit

### Deployment
- Docker
- Docker Compose when useful

### Development
- pytest
- Git
- GitHub

Do not add libraries simply because they are popular. Prefer the simplest tool that solves the problem.

---

# Project Architecture

The intended high-level architecture is:

Recruiter
    ↓
Streamlit Dashboard
    ↓
FastAPI Backend
    ↓
Resume Processing
    ↓
Sentence Transformer Embeddings
    ↓
FAISS Vector Search
    ↓
Candidate Ranking
    ↓
LLM / RAG Analysis
    ↓
Candidate Summary & Recommendation
    ↓
Streamlit Dashboard


The architecture may evolve during development, but changes should be intentional and documented.

---

# Development Phases

Build the project incrementally.

## Phase 1 — Resume Processing

Goal:
Create a reliable PDF resume ingestion and text extraction system.

Includes:
- PDF upload/input
- PDF validation
- Text extraction
- Text cleaning
- Basic error handling
- Unit tests

Do NOT implement:
- Embeddings
- FAISS
- LLMs
- LangChain
- FastAPI
- Streamlit
- Docker

---

## Phase 2 — Semantic Matching

Goal:
Build the core candidate matching engine.

Includes:
- Sentence Transformers
- Resume embeddings
- Job-description embeddings
- FAISS
- Similarity search
- Candidate ranking
- Match scores

---

## Phase 3 — Candidate & Skill Analysis

Goal:
Create structured candidate information.

Includes:
- Skill extraction
- Matched skills
- Missing skills
- Experience analysis
- Candidate profile representation

---

## Phase 4 — LLM + RAG

Goal:
Add intelligent candidate analysis.

Includes:
- LLM integration
- Prompt templates
- Retrieval
- RAG
- Candidate summaries
- Skill-gap analysis
- Hiring recommendations
- Structured outputs

The LLM must base its analysis on retrieved resume/job-description information.

Avoid unsupported claims or hallucinated candidate information.

---

## Phase 5 — FastAPI Backend

Goal:
Expose the application functionality through a clean REST API.

Potential endpoints:

POST /upload-resume
POST /match-candidates
POST /analyze-candidate
GET /candidates
GET /health

Use Pydantic models for request/response validation.

---

## Phase 6 — Streamlit Dashboard

Goal:
Create a recruiter-friendly interface.

The dashboard should support:
- Job description input
- Resume uploads
- Candidate ranking
- Match scores
- Candidate details
- Matched skills
- Missing skills
- AI summaries
- Recommendations

Keep the UI clean and professional.

---

## Phase 7 — Docker & Production Polish

Goal:
Make the application easy to run and demonstrate.

Includes:
- Dockerfile
- Docker Compose if useful
- Environment variables
- Logging
- Configuration
- Error handling
- Health checks
- Basic security considerations

---

## Phase 8 — Testing & Portfolio

Goal:
Prepare the project for GitHub and interviews.

Includes:
- Unit tests
- Integration tests where useful
- README
- Architecture diagram
- Screenshots
- API documentation
- Setup instructions
- Example workflow
- Known limitations
- Future improvements

---

# Important Development Rules

## 1. Work One Phase at a Time

Do not implement future phases unless explicitly requested.

If we are working on Phase 1, do not introduce:
- FAISS
- embeddings
- LLMs
- LangChain
- FastAPI
- Streamlit
- Docker

Keep the current phase working before moving forward.

---

## 2. Keep the Architecture Modular

Separate responsibilities into appropriate modules.

For example:

app/
├── resume_parser.py
├── embeddings.py
├── vector_store.py
├── llm.py
├── matching.py
├── api/
└── ...

Do not put the entire application into one large Python file.

---

## 3. Prefer Simple Solutions

This is a portfolio project, but unnecessary complexity should be avoided.

Before adding a dependency, ask:

"Do we actually need this?"

Prefer:
- Simple functions
- Clear interfaces
- Standard Python libraries when sufficient
- Well-maintained dependencies
- Readable code

---

## 4. Never Hardcode Secrets

Never put API keys, passwords, tokens, or credentials in source code.

Use environment variables.

Example:

.env

LLM_API_KEY=your_key_here

Never commit `.env` to GitHub.

Maintain `.env.example` with placeholder values.

---

# Data & Privacy

Resume files may contain personally identifiable information.

Therefore:

- Do not commit real resumes to GitHub.
- Do not commit personal candidate information.
- Do not include API keys or secrets.
- Use synthetic/sample resumes for testing and demonstrations.
- Add appropriate resume/data directories to `.gitignore` when necessary.
- Avoid exposing sensitive information in logs.

If sample resumes are needed, create fictional candidate data.

---

# Code Quality

Use:

- Python type hints
- Clear naming
- Docstrings for public functions/classes
- Small functions
- Meaningful error messages
- pathlib for filesystem operations
- Proper exception handling

Avoid:

- Giant functions
- Duplicate code
- Unnecessary abstractions
- Global mutable state
- Hardcoded absolute paths
- Silent exception handling

Example:

GOOD:

```python
def extract_resume_text(pdf_path: Path) -> str:
    """Extract and clean text from a PDF resume.""" 