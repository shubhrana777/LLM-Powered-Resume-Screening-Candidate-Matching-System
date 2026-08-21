"""Tests for POST /upload-resume.

The endpoint accepts a file from an untrusted client, so most of these tests are
about what it refuses: a file that is not a PDF, a file that only claims to be
one, an empty file, an oversized file, and a name pretending to be a path. The
rest check that nothing is left behind on disk afterwards.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from app.api.config import Settings
from app.api.schemas import PREVIEW_CHARS, UploadResumeResponse
from tests.conftest import build_test_client


def _upload(client, name: str, content: bytes, content_type: str = "application/pdf"):
    return client.post("/upload-resume", files={"file": (name, content, content_type)})


# --------------------------------------------------------------------------
# Success
# --------------------------------------------------------------------------


def test_a_valid_pdf_is_accepted(api_client, valid_pdf_bytes: bytes):
    response = _upload(api_client, "sarah_wilson.pdf", valid_pdf_bytes)
    assert response.status_code == 200, response.text


def test_response_matches_its_schema(api_client, valid_pdf_bytes: bytes):
    UploadResumeResponse.model_validate(_upload(api_client, "cv.pdf", valid_pdf_bytes).json())


def test_response_reports_success(api_client, valid_pdf_bytes: bytes):
    assert _upload(api_client, "cv.pdf", valid_pdf_bytes).json()["status"] == "success"


def test_response_echoes_the_filename(api_client, valid_pdf_bytes: bytes):
    assert _upload(api_client, "cv.pdf", valid_pdf_bytes).json()["filename"] == "cv.pdf"


def test_response_reports_the_extracted_length(api_client, valid_pdf_bytes: bytes):
    body = _upload(api_client, "cv.pdf", valid_pdf_bytes).json()
    assert body["text_length"] > 0
    assert body["word_count"] > 0


def test_response_previews_the_extracted_text(api_client, valid_pdf_bytes: bytes):
    body = _upload(api_client, "cv.pdf", valid_pdf_bytes).json()
    assert "Jane Doe" in body["preview"]


def test_the_preview_is_bounded(api_client, valid_pdf_bytes: bytes):
    body = _upload(api_client, "cv.pdf", valid_pdf_bytes).json()
    assert len(body["preview"]) <= PREVIEW_CHARS


def test_uppercase_extensions_are_accepted(api_client, valid_pdf_bytes: bytes):
    assert _upload(api_client, "CV.PDF", valid_pdf_bytes).status_code == 200


def test_a_multipage_pdf_is_accepted(api_client, multipage_pdf: Path):
    response = _upload(api_client, "long.pdf", multipage_pdf.read_bytes())
    assert response.status_code == 200
    assert "Page two content" in response.json()["preview"]


# --------------------------------------------------------------------------
# Rejected uploads
# --------------------------------------------------------------------------


def test_a_missing_file_is_rejected(api_client):
    response = api_client.post("/upload-resume")
    assert response.status_code == 422
    assert response.json()["code"] == "validation_error"


def test_a_text_file_is_rejected(api_client):
    response = _upload(api_client, "resume.txt", b"Jane Doe, engineer", "text/plain")
    assert response.status_code == 400
    assert response.json()["code"] == "unsupported_file_type"


def test_a_file_with_no_extension_is_rejected(api_client, valid_pdf_bytes: bytes):
    response = _upload(api_client, "resume", valid_pdf_bytes)
    assert response.status_code == 400
    assert response.json()["code"] == "unsupported_file_type"


def test_a_non_pdf_wearing_a_pdf_extension_is_rejected(api_client):
    """The extension is a claim; the leading bytes are the evidence."""
    response = _upload(api_client, "resume.pdf", b"PK\x03\x04 this is a zip file")
    assert response.status_code == 400
    assert response.json()["code"] == "unsupported_file_type"


def test_an_empty_file_is_rejected(api_client):
    response = _upload(api_client, "resume.pdf", b"")
    assert response.status_code == 400
    assert response.json()["code"] == "empty_file"


def test_a_corrupt_pdf_is_rejected(api_client, corrupt_pdf_bytes: bytes):
    response = _upload(api_client, "resume.pdf", corrupt_pdf_bytes)
    assert response.status_code == 400
    assert response.json()["code"] == "invalid_pdf"


def test_a_pdf_with_no_selectable_text_is_rejected(api_client, empty_page_pdf_bytes: bytes):
    response = _upload(api_client, "scan.pdf", empty_page_pdf_bytes)
    assert response.status_code == 400
    assert response.json()["code"] == "no_extractable_text"


def test_the_scanned_pdf_message_explains_the_problem(api_client, empty_page_pdf_bytes: bytes):
    detail = _upload(api_client, "scan.pdf", empty_page_pdf_bytes).json()["detail"].lower()
    assert "ocr" in detail


def test_an_oversized_file_is_rejected(api_settings, api_service, valid_pdf_bytes: bytes):
    tiny_limit = Settings(resume_dir=api_settings.resume_dir, max_upload_bytes=16)

    with build_test_client(tiny_limit, api_service) as client:
        response = _upload(client, "resume.pdf", valid_pdf_bytes)

    assert response.status_code == 413
    assert response.json()["code"] == "payload_too_large"


# --------------------------------------------------------------------------
# Safety
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sent, expected",
    [
        ("../../../etc/passwd.pdf", "passwd.pdf"),
        ("..\\..\\windows\\system32\\evil.pdf", "evil.pdf"),
        ("/absolute/path/cv.pdf", "cv.pdf"),
    ],
)
def test_a_path_in_the_filename_is_reduced_to_a_name(
    api_client, valid_pdf_bytes: bytes, sent: str, expected: str
):
    response = _upload(api_client, sent, valid_pdf_bytes)
    assert response.status_code == 200
    assert response.json()["filename"] == expected


def test_no_error_response_exposes_a_server_path(api_client, corrupt_pdf_bytes: bytes):
    body = _upload(api_client, "resume.pdf", corrupt_pdf_bytes).text
    assert tempfile.gettempdir() not in body
    assert ":\\" not in body


def test_a_successful_upload_leaves_no_temporary_file(
    api_client, valid_pdf_bytes: bytes, tmp_path: Path, monkeypatch
):
    workspace = tmp_path / "tempdir"
    workspace.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(workspace))

    assert _upload(api_client, "cv.pdf", valid_pdf_bytes).status_code == 200
    assert list(workspace.iterdir()) == []


def test_a_rejected_upload_leaves_no_temporary_file(
    api_client, corrupt_pdf_bytes: bytes, tmp_path: Path, monkeypatch
):
    workspace = tmp_path / "tempdir"
    workspace.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(workspace))

    assert _upload(api_client, "cv.pdf", corrupt_pdf_bytes).status_code == 400
    assert list(workspace.iterdir()) == []


def test_an_oversized_upload_leaves_no_temporary_file(
    api_settings, api_service, valid_pdf_bytes: bytes, tmp_path: Path, monkeypatch
):
    workspace = tmp_path / "tempdir"
    workspace.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(workspace))

    tiny_limit = Settings(resume_dir=api_settings.resume_dir, max_upload_bytes=16)
    with build_test_client(tiny_limit, api_service) as client:
        assert _upload(client, "cv.pdf", valid_pdf_bytes).status_code == 413

    assert list(workspace.iterdir()) == []


def test_an_upload_does_not_join_the_candidate_pool(api_client, valid_pdf_bytes: bytes):
    """Uploads are parsed and discarded. The pool is the server's own directory."""
    before = api_client.get("/candidates").json()

    assert _upload(api_client, "dana_doe.pdf", valid_pdf_bytes).status_code == 200

    assert api_client.get("/candidates").json() == before


def test_an_upload_does_not_write_into_the_resume_directory(
    api_client, valid_pdf_bytes: bytes, analyst_resume_dir: Path
):
    before = sorted(path.name for path in analyst_resume_dir.iterdir())

    assert _upload(api_client, "dana_doe.pdf", valid_pdf_bytes).status_code == 200

    assert sorted(path.name for path in analyst_resume_dir.iterdir()) == before
