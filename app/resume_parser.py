"""Resume parsing utilities.

Phase 1 scope: read a PDF resume from disk and return clean, normalized text.
No embeddings, no LLM calls, no matching logic -- just reliable text extraction.
"""

from __future__ import annotations

import re
from pathlib import Path

import fitz  # PyMuPDF

__all__ = [
    "ResumeParserError",
    "ResumeFileNotFoundError",
    "UnsupportedFileTypeError",
    "CorruptedPDFError",
    "NoExtractableTextError",
    "validate_resume_path",
    "normalize_text",
    "extract_text_from_pdf",
]

PDF_SUFFIX = ".pdf"

# Collapses runs of spaces/tabs, but never newlines.
_HORIZONTAL_WHITESPACE = re.compile(r"[^\S\n]+")
# Three or more newlines become a single blank line separator.
_EXCESS_BLANK_LINES = re.compile(r"\n{3,}")


class ResumeParserError(Exception):
    """Base class for every error raised by this module."""


class ResumeFileNotFoundError(ResumeParserError):
    """The given path does not point at an existing file."""


class UnsupportedFileTypeError(ResumeParserError):
    """The given file is not a PDF."""


class CorruptedPDFError(ResumeParserError):
    """The file has a .pdf suffix but PyMuPDF cannot open it."""


class NoExtractableTextError(ResumeParserError):
    """The PDF opened fine but contains no selectable text (e.g. a scan)."""


def validate_resume_path(file_path: str | Path) -> Path:
    """Validate that ``file_path`` is an existing PDF file.

    Args:
        file_path: Path to a resume, relative or absolute.

    Returns:
        The resolved :class:`~pathlib.Path` of the file.

    Raises:
        ResumeFileNotFoundError: If the path does not exist or is a directory.
        UnsupportedFileTypeError: If the file does not have a ``.pdf`` suffix.
    """
    path = Path(file_path).expanduser().resolve()

    if not path.exists():
        raise ResumeFileNotFoundError(f"No such file: {path}")

    if not path.is_file():
        raise ResumeFileNotFoundError(f"Not a file (is it a directory?): {path}")

    if path.suffix.lower() != PDF_SUFFIX:
        raise UnsupportedFileTypeError(
            f"Expected a {PDF_SUFFIX} file but got '{path.suffix or 'no extension'}': {path}"
        )

    return path


def normalize_text(text: str) -> str:
    """Normalize whitespace in extracted resume text.

    Trims each line, drops empty lines that carry no meaning, collapses runs of
    spaces and tabs, and limits consecutive blank lines to one.

    Args:
        text: Raw text as returned by the PDF reader.

    Returns:
        The cleaned text, without leading or trailing whitespace.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _HORIZONTAL_WHITESPACE.sub(" ", text)

    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)

    text = _EXCESS_BLANK_LINES.sub("\n\n", text)
    return text.strip()


def _read_pdf_pages(path: Path) -> list[str]:
    """Return the raw text of every page in the PDF at ``path``."""
    try:
        with fitz.open(path) as document:
            return [page.get_text("text") for page in document]
    except ResumeParserError:
        raise
    except Exception as exc:  # PyMuPDF raises several unrelated exception types
        raise CorruptedPDFError(f"Could not open PDF: {path} ({exc})") from exc


def extract_text_from_pdf(file_path: str | Path) -> str:
    """Extract clean text from a PDF resume.

    Args:
        file_path: Path to a ``.pdf`` resume file.

    Returns:
        Normalized text from all pages, joined in page order.

    Raises:
        ResumeFileNotFoundError: If the file is missing.
        UnsupportedFileTypeError: If the file is not a PDF.
        CorruptedPDFError: If the PDF cannot be opened.
        NoExtractableTextError: If the PDF has no selectable text.
    """
    path = validate_resume_path(file_path)
    pages = _read_pdf_pages(path)

    cleaned = normalize_text("\n\n".join(pages))
    if not cleaned:
        raise NoExtractableTextError(
            f"No extractable text found in {path}. "
            "The file may be a scanned image; OCR is out of scope for Phase 1."
        )

    return cleaned
