"""Shared pytest fixtures.

Test PDFs are generated at runtime with PyMuPDF, which the project already
depends on. That keeps the dependency list minimal (no reportlab/fpdf) and
guarantees the fixtures stay in sync with the library doing the parsing.
"""

from __future__ import annotations

from pathlib import Path

import fitz
import pytest

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
