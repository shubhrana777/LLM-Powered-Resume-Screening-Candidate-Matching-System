"""Tests for GET /candidates."""

from __future__ import annotations

from pathlib import Path

from app.api.config import Settings
from app.api.schemas import CandidateListResponse
from app.api.service import ScreeningService
from tests.conftest import build_test_client


def _listing(client) -> dict:
    response = client.get("/candidates")
    assert response.status_code == 200, response.text
    return response.json()


def test_listing_returns_200(api_client):
    assert api_client.get("/candidates").status_code == 200


def test_listing_matches_its_schema(api_client):
    CandidateListResponse.model_validate(_listing(api_client))


def test_listing_returns_every_readable_resume(api_client):
    body = _listing(api_client)
    assert {entry["candidate_id"] for entry in body["candidates"]} == {
        "sarah_wilson",
        "james_patel",
        "nina_volkov",
    }


def test_count_matches_the_list_length(api_client):
    body = _listing(api_client)
    assert body["count"] == len(body["candidates"]) == 3


def test_listing_reports_display_names(api_client):
    names = {entry["candidate_id"]: entry["name"] for entry in _listing(api_client)["candidates"]}
    assert names["sarah_wilson"] == "Sarah Wilson"


def test_listing_reports_filenames(api_client):
    files = {entry["candidate_id"]: entry["filename"] for entry in _listing(api_client)["candidates"]}
    assert files["nina_volkov"] == "nina_volkov.pdf"


def test_listing_reports_text_length(api_client):
    for entry in _listing(api_client)["candidates"]:
        assert entry["text_length"] > 0


def test_listing_does_not_return_resume_text(api_client):
    """A listing is metadata. Resume contents are candidate information."""
    assert "Financial Analyst with 4 years" not in api_client.get("/candidates").text


def test_listing_reports_no_failures_for_a_clean_directory(api_client):
    assert _listing(api_client)["unreadable"] == []


def test_unreadable_resume_is_reported_not_hidden(
    analyst_resume_dir: Path, analyst_embedder, api_service
):
    (analyst_resume_dir / "broken.pdf").write_bytes(b"not a pdf at all")
    settings = Settings(resume_dir=analyst_resume_dir)
    service = ScreeningService(settings, embedder=analyst_embedder)

    with build_test_client(settings, service) as client:
        body = _listing(client)

    assert body["count"] == 3
    assert [entry["filename"] for entry in body["unreadable"]] == ["broken.pdf"]


def test_failure_reason_does_not_leak_the_server_path(
    analyst_resume_dir: Path, analyst_embedder
):
    (analyst_resume_dir / "broken.pdf").write_bytes(b"not a pdf at all")
    settings = Settings(resume_dir=analyst_resume_dir)
    service = ScreeningService(settings, embedder=analyst_embedder)

    with build_test_client(settings, service) as client:
        reason = _listing(client)["unreadable"][0]["reason"]

    assert str(analyst_resume_dir) not in reason
    assert reason  # but it still says something useful


def test_empty_directory_is_an_empty_list_not_an_error(tmp_path: Path, analyst_embedder):
    empty = tmp_path / "no_resumes"
    empty.mkdir()
    settings = Settings(resume_dir=empty)
    service = ScreeningService(settings, embedder=analyst_embedder)

    with build_test_client(settings, service) as client:
        response = client.get("/candidates")

    assert response.status_code == 200
    assert response.json() == {"candidates": [], "count": 0, "unreadable": []}


def test_missing_directory_returns_404(tmp_path: Path, analyst_embedder):
    settings = Settings(resume_dir=tmp_path / "nowhere")
    service = ScreeningService(settings, embedder=analyst_embedder)

    with build_test_client(settings, service) as client:
        response = client.get("/candidates")

    assert response.status_code == 404
    assert response.json()["code"] == "resume_directory_not_found"


def test_missing_directory_error_does_not_leak_the_path(tmp_path: Path, analyst_embedder):
    missing = tmp_path / "nowhere"
    settings = Settings(resume_dir=missing)
    service = ScreeningService(settings, embedder=analyst_embedder)

    with build_test_client(settings, service) as client:
        body = client.get("/candidates").text

    assert str(missing) not in body


def test_new_resume_appears_without_a_restart(
    api_client, analyst_resume_dir: Path, valid_pdf: Path
):
    """The pool is keyed by directory contents, so a new file is picked up."""
    assert _listing(api_client)["count"] == 3

    (analyst_resume_dir / "dana_doe.pdf").write_bytes(valid_pdf.read_bytes())

    body = _listing(api_client)
    assert body["count"] == 4
    assert "dana_doe" in {entry["candidate_id"] for entry in body["candidates"]}
