"""Unit tests for the app.main command-line interface."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.main import main


def test_prints_extracted_text(valid_pdf: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main([str(valid_pdf)]) == 0
    assert "Jane Doe" in capsys.readouterr().out


def test_writes_to_output_file(valid_pdf: Path, tmp_path: Path) -> None:
    destination = tmp_path / "out" / "resume.txt"
    assert main([str(valid_pdf), "--output", str(destination)]) == 0
    assert "Jane Doe" in destination.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "fixture_name",
    ["missing_pdf", "text_file", "corrupted_pdf", "empty_pdf"],
)
def test_user_errors_exit_cleanly(
    fixture_name: str,
    request: pytest.FixtureRequest,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """User errors must return exit code 1 with a message, not a traceback."""
    path = request.getfixturevalue(fixture_name)
    assert main([str(path)]) == 1

    captured = capsys.readouterr()
    assert captured.err.startswith("Error: ")
    assert "Traceback" not in captured.err
    assert captured.out == ""


def test_missing_argument_exits_with_usage_error(valid_pdf: Path) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main([])
    assert exc_info.value.code == 2
