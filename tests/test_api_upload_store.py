"""Tests for `POST /upload-resume` with `store=true`.

Storing is what turns an upload into a rankable candidate, which is what the
dashboard needs. It is opt-in: the default is still parse-and-discard, and
`tests/test_api_upload.py` proves that default has not changed.

The risk storing introduces is writing a client-controlled file to disk, so most
of these tests are about the destination name: it is generated from a slug of
the submitted name, never taken from it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.api.config import Settings
from app.api.errors import NotFoundError
from app.api.schemas import UploadResumeResponse
from app.api.service import FALLBACK_SLUG, MAX_SLUG_LENGTH, ScreeningService, slugify_candidate_id
from tests.conftest import ANALYST_JOB_DESCRIPTION, build_test_client


def _upload(client, name: str, content: bytes, store: bool = True):
    return client.post(
        "/upload-resume",
        files={"file": (name, content, "application/pdf")},
        data={"store": str(store).lower()},
    )


# --------------------------------------------------------------------------
# Storing
# --------------------------------------------------------------------------


def test_a_stored_upload_reports_that_it_was_stored(api_client, valid_pdf_bytes: bytes):
    body = _upload(api_client, "dana_doe.pdf", valid_pdf_bytes).json()
    assert body["stored"] is True
    assert body["candidate_id"] == "dana_doe"


def test_the_response_still_matches_its_schema(api_client, valid_pdf_bytes: bytes):
    UploadResumeResponse.model_validate(_upload(api_client, "dana_doe.pdf", valid_pdf_bytes).json())


def test_a_stored_resume_joins_the_candidate_pool(api_client, valid_pdf_bytes: bytes):
    before = api_client.get("/candidates").json()["count"]

    assert _upload(api_client, "dana_doe.pdf", valid_pdf_bytes).status_code == 200

    listing = api_client.get("/candidates").json()
    assert listing["count"] == before + 1
    assert "dana_doe" in {entry["candidate_id"] for entry in listing["candidates"]}


def test_a_stored_resume_is_written_into_the_resume_directory(
    api_client, valid_pdf_bytes: bytes, analyst_resume_dir: Path
):
    _upload(api_client, "dana_doe.pdf", valid_pdf_bytes)
    assert (analyst_resume_dir / "dana_doe.pdf").is_file()


def test_a_stored_resume_can_be_ranked(api_client, valid_pdf_bytes: bytes):
    """The whole point: an uploaded resume becomes rankable."""
    _upload(api_client, "dana_doe.pdf", valid_pdf_bytes)

    body = api_client.post(
        "/match-candidates", json={"job_description": ANALYST_JOB_DESCRIPTION, "top_k": 10}
    ).json()

    assert "dana_doe" in {entry["candidate_id"] for entry in body["results"]}


def test_a_stored_resume_can_be_analysed(api_client, valid_pdf_bytes: bytes):
    _upload(api_client, "dana_doe.pdf", valid_pdf_bytes)

    response = api_client.post(
        "/analyze-candidate",
        json={"candidate": "dana_doe", "job_description": ANALYST_JOB_DESCRIPTION},
    )

    assert response.status_code == 200
    assert response.json()["candidate_id"] == "dana_doe"


def test_several_resumes_can_be_stored_in_turn(api_client, valid_pdf_bytes: bytes):
    for name in ("one.pdf", "two.pdf", "three.pdf"):
        assert _upload(api_client, name, valid_pdf_bytes).status_code == 200

    ids = {entry["candidate_id"] for entry in api_client.get("/candidates").json()["candidates"]}
    assert {"one", "two", "three"} <= ids


def test_re_uploading_the_same_name_replaces_rather_than_duplicates(
    api_client, valid_pdf_bytes: bytes, multipage_pdf: Path
):
    _upload(api_client, "dana_doe.pdf", valid_pdf_bytes)
    first = api_client.get("/candidates").json()["count"]

    _upload(api_client, "dana_doe.pdf", multipage_pdf.read_bytes())
    listing = api_client.get("/candidates").json()

    assert listing["count"] == first
    stored = [e for e in listing["candidates"] if e["candidate_id"] == "dana_doe"]
    assert len(stored) == 1


# --------------------------------------------------------------------------
# The default is unchanged
# --------------------------------------------------------------------------


def test_storing_is_opt_in(api_client, valid_pdf_bytes: bytes):
    body = _upload(api_client, "dana_doe.pdf", valid_pdf_bytes, store=False).json()
    assert body["stored"] is False
    assert body["candidate_id"] is None


def test_omitting_the_flag_does_not_store(api_client, valid_pdf_bytes: bytes, analyst_resume_dir):
    response = api_client.post(
        "/upload-resume", files={"file": ("dana_doe.pdf", valid_pdf_bytes, "application/pdf")}
    )
    assert response.json()["stored"] is False
    assert not (analyst_resume_dir / "dana_doe.pdf").exists()


# --------------------------------------------------------------------------
# Rejected uploads are never stored
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, content",
    [
        ("resume.txt", b"%PDF-1.4 but a txt name"),
        ("resume.pdf", b"PK\x03\x04 not a pdf"),
        ("resume.pdf", b""),
    ],
)
def test_a_rejected_upload_writes_nothing(
    api_client, analyst_resume_dir: Path, name: str, content: bytes
):
    before = sorted(path.name for path in analyst_resume_dir.iterdir())

    assert _upload(api_client, name, content).status_code == 400

    assert sorted(path.name for path in analyst_resume_dir.iterdir()) == before


def test_a_corrupt_pdf_is_not_stored(
    api_client, analyst_resume_dir: Path, corrupt_pdf_bytes: bytes
):
    assert _upload(api_client, "broken.pdf", corrupt_pdf_bytes).status_code == 400
    assert not (analyst_resume_dir / "broken.pdf").exists()


def test_a_pdf_without_text_is_not_stored(
    api_client, analyst_resume_dir: Path, empty_page_pdf_bytes: bytes
):
    assert _upload(api_client, "scan.pdf", empty_page_pdf_bytes).status_code == 400
    assert not (analyst_resume_dir / "scan.pdf").exists()


# --------------------------------------------------------------------------
# The destination name is generated, never supplied
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "submitted, expected",
    [
        ("sarah_wilson.pdf", "sarah_wilson"),
        ("Sarah Wilson.pdf", "sarah_wilson"),
        ("Sarah Wilson (CV).pdf", "sarah_wilson_cv"),
        ("résumé-2024.pdf", "r_sum_2024"),
        ("../../../etc/passwd.pdf", "passwd"),
        ("..\\..\\windows\\system32\\evil.pdf", "evil"),
        ("/absolute/path/cv.pdf", "cv"),
        ("CON.pdf", "con"),
        ("...pdf", FALLBACK_SLUG),
        ("....pdf", FALLBACK_SLUG),
        ("   .pdf", FALLBACK_SLUG),
    ],
)
def test_a_submitted_name_is_reduced_to_a_slug(submitted: str, expected: str):
    assert slugify_candidate_id(submitted) == expected


def test_a_slug_contains_only_safe_characters():
    slug = slugify_candidate_id("../<>:|?*weird\\name!!.pdf")
    assert all(char.isalnum() or char == "_" for char in slug)


def test_a_slug_is_length_capped():
    assert len(slugify_candidate_id("x" * 500 + ".pdf")) <= MAX_SLUG_LENGTH


@pytest.mark.parametrize("submitted", ["../../../etc/passwd.pdf", "..\\..\\evil.pdf"])
def test_a_traversing_name_cannot_escape_the_resume_directory(
    api_client, analyst_resume_dir: Path, valid_pdf_bytes: bytes, submitted: str
):
    assert _upload(api_client, submitted, valid_pdf_bytes).status_code == 200

    written = {path.name for path in analyst_resume_dir.iterdir()}
    assert written <= {
        "sarah_wilson.pdf",
        "james_patel.pdf",
        "nina_volkov.pdf",
        "passwd.pdf",
        "evil.pdf",
    }
    # Nothing landed beside the directory rather than inside it.
    assert not (analyst_resume_dir.parent / "passwd.pdf").exists()
    assert not (analyst_resume_dir.parent / "evil.pdf").exists()


def test_the_response_never_reveals_where_the_file_went(api_client, valid_pdf_bytes: bytes):
    body = _upload(api_client, "dana_doe.pdf", valid_pdf_bytes).text
    assert ":\\" not in body
    assert "/tmp" not in body


# --------------------------------------------------------------------------
# Service-level behaviour
# --------------------------------------------------------------------------


def test_storing_into_a_missing_directory_is_reported(tmp_path: Path, valid_pdf: Path, analyst_embedder):
    service = ScreeningService(Settings(resume_dir=tmp_path / "gone"), embedder=analyst_embedder)

    with pytest.raises(NotFoundError) as info:
        service.store_resume(valid_pdf, "dana_doe.pdf")

    assert info.value.code == "resume_directory_not_found"


def test_storing_invalidates_the_cached_indexes(api_settings, analyst_embedder, valid_pdf: Path):
    from app.llm import FakeLLMProvider

    service = ScreeningService(api_settings, embedder=analyst_embedder, llm=FakeLLMProvider())
    first_matcher = service.matcher()

    service.store_resume(valid_pdf, "dana_doe.pdf")

    assert service.matcher() is not first_matcher
    assert service.matcher().candidate_count == 4


def test_the_stored_candidate_is_returned(api_settings, analyst_embedder, valid_pdf: Path):
    service = ScreeningService(api_settings, embedder=analyst_embedder)

    candidate = service.store_resume(valid_pdf, "Dana Doe.pdf")

    assert candidate.candidate_id == "dana_doe"
    assert candidate.resume_text.strip()
