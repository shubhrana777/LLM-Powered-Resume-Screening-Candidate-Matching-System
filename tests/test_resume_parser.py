"""Unit tests for app.resume_parser."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.resume_parser import (
    CorruptedPDFError,
    NoExtractableTextError,
    ResumeFileNotFoundError,
    ResumeParserError,
    UnsupportedFileTypeError,
    extract_text_from_pdf,
    normalize_text,
    validate_resume_path,
)


class TestValidateResumePath:
    def test_returns_resolved_path_for_valid_pdf(self, valid_pdf: Path) -> None:
        assert validate_resume_path(valid_pdf) == valid_pdf.resolve()

    def test_accepts_uppercase_suffix(self, valid_pdf: Path) -> None:
        renamed = valid_pdf.with_name("RESUME.PDF")
        valid_pdf.rename(renamed)
        assert validate_resume_path(renamed) == renamed.resolve()

    def test_missing_file_raises(self, missing_pdf: Path) -> None:
        with pytest.raises(ResumeFileNotFoundError):
            validate_resume_path(missing_pdf)

    def test_directory_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ResumeFileNotFoundError):
            validate_resume_path(tmp_path)

    def test_non_pdf_raises(self, text_file: Path) -> None:
        with pytest.raises(UnsupportedFileTypeError):
            validate_resume_path(text_file)


class TestNormalizeText:
    def test_collapses_horizontal_whitespace(self) -> None:
        assert normalize_text("Jane    Doe\tEngineer") == "Jane Doe Engineer"

    def test_strips_each_line(self) -> None:
        assert normalize_text("  Jane Doe  \n   Engineer  ") == "Jane Doe\nEngineer"

    def test_collapses_excess_blank_lines(self) -> None:
        assert normalize_text("A\n\n\n\n\nB") == "A\n\nB"

    def test_normalizes_windows_line_endings(self) -> None:
        assert normalize_text("A\r\nB") == "A\nB"

    def test_whitespace_only_input_becomes_empty(self) -> None:
        assert normalize_text("   \n\n \t \n ") == ""


class TestExtractTextFromPDF:
    def test_extracts_expected_content(self, valid_pdf: Path) -> None:
        text = extract_text_from_pdf(valid_pdf)
        assert "Jane Doe" in text
        assert "jane.doe@example.com" in text
        assert "Python, SQL, Docker, AWS" in text

    def test_output_has_no_leading_or_trailing_whitespace(self, valid_pdf: Path) -> None:
        text = extract_text_from_pdf(valid_pdf)
        assert text == text.strip()

    def test_output_has_no_runs_of_blank_lines(self, valid_pdf: Path) -> None:
        assert "\n\n\n" not in extract_text_from_pdf(valid_pdf)

    def test_accepts_string_path(self, valid_pdf: Path) -> None:
        assert "Jane Doe" in extract_text_from_pdf(str(valid_pdf))

    def test_combines_all_pages(self, multipage_pdf: Path) -> None:
        text = extract_text_from_pdf(multipage_pdf)
        assert "Page one content" in text
        assert "Page two content" in text
        assert text.index("Page one content") < text.index("Page two content")

    def test_missing_file_raises(self, missing_pdf: Path) -> None:
        with pytest.raises(ResumeFileNotFoundError):
            extract_text_from_pdf(missing_pdf)

    def test_invalid_file_type_raises(self, text_file: Path) -> None:
        with pytest.raises(UnsupportedFileTypeError):
            extract_text_from_pdf(text_file)

    def test_corrupted_pdf_raises(self, corrupted_pdf: Path) -> None:
        with pytest.raises(CorruptedPDFError):
            extract_text_from_pdf(corrupted_pdf)

    def test_pdf_without_text_raises(self, empty_pdf: Path) -> None:
        with pytest.raises(NoExtractableTextError):
            extract_text_from_pdf(empty_pdf)

    @pytest.mark.parametrize(
        "fixture_name",
        ["missing_pdf", "text_file", "corrupted_pdf", "empty_pdf"],
    )
    def test_all_failures_share_a_base_class(
        self, fixture_name: str, request: pytest.FixtureRequest
    ) -> None:
        path = request.getfixturevalue(fixture_name)
        with pytest.raises(ResumeParserError):
            extract_text_from_pdf(path)
