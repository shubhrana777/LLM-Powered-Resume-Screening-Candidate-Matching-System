"""Shared pytest fixtures.

Test PDFs are generated at runtime with PyMuPDF, which the project already
depends on. That keeps the dependency list minimal (no reportlab/fpdf) and
guarantees the fixtures stay in sync with the library doing the parsing.

Embedding strategy for tests
---------------------------
Most tests use :class:`FakeEmbedder`, a deterministic bag-of-words embedder, so
that the suite is fast, offline, and reproducible. Real transformer weights are
a ~90 MB download and several seconds of load time; depending on them for every
assertion would make the suite slow and network-dependent, and floating-point
output can shift between model revisions.

The fake still produces genuinely meaningful cosine similarities -- texts that
share vocabulary score higher -- so ranking-order assertions are real tests of
the matching logic, not tautologies.

Tests that must exercise the actual Sentence Transformers model are marked
``@pytest.mark.model`` and request the :func:`real_embedder` fixture, which
skips automatically when the model cannot be loaded (offline CI, no cache).
Run them with ``pytest -m model``; skip them with ``pytest -m "not model"``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import fitz
import numpy as np
import pytest

from app.embeddings import DEFAULT_MODEL_NAME, VECTOR_DTYPE, _validate_texts
from app.models import Candidate

SAMPLE_LINES = [
    "Jane Doe",
    "Senior Python Engineer",
    "jane.doe@example.com | +1 555 0100",
    "",
    "EXPERIENCE",
    "Acme Corp - Backend Engineer (2020-2024)",
    "Built data pipelines and REST APIs.",
    "",
    "SKILLS",
    "Python, SQL, Docker, AWS",
]


def _write_pdf(path: Path, pages: list[list[str]]) -> Path:
    """Write a PDF at ``path`` where each inner list is one page of lines."""
    document = fitz.open()
    try:
        for lines in pages:
            page = document.new_page()
            y = 72
            for line in lines:
                if line:
                    page.insert_text((72, y), line, fontsize=11)
                y += 16
        document.save(path)
    finally:
        document.close()
    return path


@pytest.fixture
def valid_pdf(tmp_path: Path) -> Path:
    """A single-page PDF containing selectable resume text."""
    return _write_pdf(tmp_path / "valid_resume.pdf", [SAMPLE_LINES])


@pytest.fixture
def multipage_pdf(tmp_path: Path) -> Path:
    """A two-page PDF with distinct text on each page."""
    return _write_pdf(
        tmp_path / "multipage_resume.pdf",
        [["Page one content"], ["Page two content"]],
    )


@pytest.fixture
def empty_pdf(tmp_path: Path) -> Path:
    """A structurally valid PDF with a blank page (no extractable text)."""
    return _write_pdf(tmp_path / "empty_resume.pdf", [[]])


@pytest.fixture
def text_file(tmp_path: Path) -> Path:
    """A plain text file used to check file-type validation."""
    path = tmp_path / "resume.txt"
    path.write_text("Jane Doe\nSenior Python Engineer\n", encoding="utf-8")
    return path


@pytest.fixture
def corrupted_pdf(tmp_path: Path) -> Path:
    """A file with a .pdf suffix whose bytes are not a valid PDF."""
    path = tmp_path / "corrupted_resume.pdf"
    path.write_bytes(b"this is definitely not a pdf")
    return path


@pytest.fixture
def missing_pdf(tmp_path: Path) -> Path:
    """A path inside a real directory that does not exist on disk."""
    return tmp_path / "does_not_exist.pdf"


# --------------------------------------------------------------------------
# Phase 2 fixtures
# --------------------------------------------------------------------------

VOCABULARY = (
    "python",
    "backend",
    "api",
    "sql",
    "database",
    "docker",
    "kubernetes",
    "machine",
    "learning",
    "pytorch",
    "nlp",
    "embeddings",
    "react",
    "frontend",
    "javascript",
    "pastry",
    "baking",
    "chocolate",
    "dessert",
    "kitchen",
)


class FakeEmbedder:
    """A deterministic, offline stand-in for the real embedder.

    Builds an L2-normalized bag-of-words vector over a fixed vocabulary, so
    cosine similarity between two texts rises with their shared vocabulary.
    That is enough for the matching engine's ranking behaviour to be tested
    meaningfully without downloading transformer weights.

    A constant bias component keeps the vector norm non-zero even for text that
    contains none of the vocabulary, avoiding a divide-by-zero.

    Input validation is delegated to the same helper the real embedder uses, so
    the fake and the real implementation reject exactly the same inputs.

    Args:
        vocabulary: Tokens that make up the vector space.
    """

    BIAS = 0.1

    def __init__(self, vocabulary: Sequence[str] = VOCABULARY) -> None:
        self._vocabulary = tuple(vocabulary)

    @property
    def dimension(self) -> int:
        """Vocabulary size plus one for the bias component."""
        return len(self._vocabulary) + 1

    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single string as a 1-D unit vector."""
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Embed several strings as a 2-D array of unit vectors."""
        cleaned = _validate_texts(texts)

        vectors = np.zeros((len(cleaned), self.dimension), dtype=VECTOR_DTYPE)
        for row, text in enumerate(cleaned):
            words = text.lower().replace(",", " ").replace(".", " ").split()
            for column, token in enumerate(self._vocabulary):
                vectors[row, column] = words.count(token)
            vectors[row, -1] = self.BIAS
            vectors[row] /= np.linalg.norm(vectors[row])

        return np.ascontiguousarray(vectors, dtype=VECTOR_DTYPE)


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    """A deterministic offline embedder."""
    return FakeEmbedder()


@pytest.fixture(scope="session")
def real_embedder():
    """The actual Sentence Transformers embedder, or skip if unavailable.

    Skips rather than fails when the model cannot be loaded, so the suite still
    passes on a machine with no network access and no cached weights.
    """
    from app.embeddings import SentenceTransformerEmbedder

    embedder = SentenceTransformerEmbedder(DEFAULT_MODEL_NAME)
    try:
        embedder.embed_text("warm up the model")
    except Exception as exc:  # noqa: BLE001 - any failure means "cannot test"
        pytest.skip(f"Sentence Transformers model unavailable: {exc}")
    return embedder


BACKEND_RESUME = "Python backend engineer building api services with sql database and docker"
ML_RESUME = "Machine learning engineer using pytorch for nlp and embeddings in python"
FRONTEND_RESUME = "Frontend developer building react interfaces in javascript"
CHEF_RESUME = "Pastry chef specialising in baking chocolate dessert and kitchen management"

BACKEND_JOB = "Hiring a python backend engineer for api and database work with docker"


@pytest.fixture
def sample_candidates() -> list[Candidate]:
    """Four candidates spanning clearly different domains."""
    return [
        Candidate("c-backend", BACKEND_RESUME, "Backend Person"),
        Candidate("c-ml", ML_RESUME, "ML Person"),
        Candidate("c-frontend", FRONTEND_RESUME, "Frontend Person"),
        Candidate("c-chef", CHEF_RESUME, "Chef Person"),
    ]


@pytest.fixture
def resume_dir(tmp_path: Path) -> Path:
    """A directory containing three resume PDFs and one non-PDF file."""
    folder = tmp_path / "resumes"
    folder.mkdir()

    _write_pdf(folder / "alice_backend.pdf", [[BACKEND_RESUME]])
    _write_pdf(folder / "bob_ml_resume.pdf", [[ML_RESUME]])
    _write_pdf(folder / "carol_chef.pdf", [[CHEF_RESUME]])
    (folder / "notes.txt").write_text("not a resume", encoding="utf-8")

    return folder


# --------------------------------------------------------------------------
# Phase 3 fixtures
# --------------------------------------------------------------------------

STRONG_ANALYST_RESUME = """Sarah Wilson
Senior Financial Analyst

SUMMARY
Financial analyst with 4 years of experience in budgeting and forecasting.

EXPERIENCE
Senior Financial Analyst, Northgate Retail
Built rolling forecasts and financial modeling for a large business unit.
Automated reporting using Python and SQL.
Developed Power BI dashboards for the executive team.

SKILLS
Excel, financial modeling, forecasting, SQL, Python, Power BI, budgeting

EDUCATION
MBA in Finance, Manchester Business School
"""

PARTIAL_ANALYST_RESUME = """James Patel
Junior Finance Associate

SUMMARY
Finance associate with 2 years of experience supporting month-end reporting.

EXPERIENCE
Finance Associate, Ridgeway Logistics
Prepared reconciliations and assisted with budgeting.
Maintained workbooks in Excel and ran SQL queries.

SKILLS
Excel, SQL, budgeting, accounting

EDUCATION
B.Com Accounting and Finance, Aston University
"""

POOR_MATCH_RESUME = """Nina Volkov
Graphic Designer

EXPERIENCE
Senior Graphic Designer, Studio Vlna
Led brand identity projects and packaging design.

SKILLS
Illustrator, Photoshop, InDesign, typography, branding

EDUCATION
Diploma in Graphic Design, Prague College of Art
"""

ANALYST_JOB_DESCRIPTION = """Financial Analyst

We are looking for a financial analyst with 3+ years of experience to support
budgeting and forecasting.

Requirements
- Strong Excel skills, including financial modeling.
- Working knowledge of SQL.
- Experience with Python for data analysis.
- Proficiency with Power BI and Tableau.
"""


@pytest.fixture
def strong_candidate() -> Candidate:
    """A candidate matching the analyst job well, with stated experience."""
    return Candidate("c-strong", STRONG_ANALYST_RESUME, "Sarah Wilson")


@pytest.fixture
def partial_candidate() -> Candidate:
    """A candidate matching some requirements, below the experience bar."""
    return Candidate("c-partial", PARTIAL_ANALYST_RESUME, "James Patel")


@pytest.fixture
def poor_candidate() -> Candidate:
    """An unrelated candidate who never states years of experience."""
    return Candidate("c-poor", POOR_MATCH_RESUME, "Nina Volkov")


@pytest.fixture
def analyst_candidates(
    strong_candidate: Candidate,
    partial_candidate: Candidate,
    poor_candidate: Candidate,
) -> list[Candidate]:
    """Strong, partial and poor candidates for the analyst job."""
    return [strong_candidate, partial_candidate, poor_candidate]


@pytest.fixture
def analyst_job() -> str:
    """A job description naming skills and a minimum experience."""
    return ANALYST_JOB_DESCRIPTION


@pytest.fixture
def analyst_resume_dir(tmp_path: Path) -> Path:
    """A directory of three analyst-candidate resume PDFs."""
    folder = tmp_path / "analyst_resumes"
    folder.mkdir()

    for stem, text in (
        ("sarah_wilson", STRONG_ANALYST_RESUME),
        ("james_patel", PARTIAL_ANALYST_RESUME),
        ("nina_volkov", POOR_MATCH_RESUME),
    ):
        _write_pdf(folder / f"{stem}.pdf", [text.splitlines()])

    return folder
