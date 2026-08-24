"""Tests for candidate deletion.

Deleting is the one operation here that destroys data, so most of these tests
are about its boundaries: it may only ever touch a file that is already a
pooled candidate, and it must leave everything else in the directory alone.

The other half is cache invalidation, which is the subtle risk. A FAISS index
outlives the files it was built from. If deleting a resume did not drop the
matcher and the pipeline, a deleted candidate would keep appearing in rankings
and keep being analysable -- reading, to a recruiter, as if the deletion had
silently failed. Several tests below exist only to prove that cannot happen.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.api.config import Settings
from app.api.errors import BadRequestError, NotFoundError
from app.api.schemas import DeleteCandidatesResponse
from app.api.service import ScreeningService
from tests.conftest import ANALYST_JOB_DESCRIPTION, build_test_client

ALL_IDS = {"sarah_wilson", "james_patel", "nina_volkov"}


def _ids(client) -> set[str]:
    listing = client.get("/candidates").json()
    return {entry["candidate_id"] for entry in listing["candidates"]}


def _rank(client, top_k: int = 10) -> set[str]:
    body = client.post(
        "/match-candidates", json={"job_description": ANALYST_JOB_DESCRIPTION, "top_k": top_k}
    ).json()
    return {row["candidate_id"] for row in body["results"]}


# --------------------------------------------------------------------------
# Delete one
# --------------------------------------------------------------------------


def test_deleting_one_candidate_returns_200(api_client):
    assert api_client.delete("/candidates/james_patel").status_code == 200


def test_the_response_matches_its_schema(api_client):
    DeleteCandidatesResponse.model_validate(api_client.delete("/candidates/james_patel").json())


def test_the_response_names_what_was_deleted(api_client):
    body = api_client.delete("/candidates/james_patel").json()
    assert body["deleted"] == ["james_patel"]
    assert body["failed"] == []


def test_the_response_reports_what_remains(api_client):
    assert api_client.delete("/candidates/james_patel").json()["remaining"] == 2


def test_a_deleted_candidate_leaves_the_listing(api_client):
    api_client.delete("/candidates/james_patel")
    assert _ids(api_client) == ALL_IDS - {"james_patel"}


def test_a_deleted_candidates_file_is_removed(api_client, analyst_resume_dir: Path):
    api_client.delete("/candidates/james_patel")
    assert not (analyst_resume_dir / "james_patel.pdf").exists()


def test_the_other_resumes_are_untouched(api_client, analyst_resume_dir: Path):
    api_client.delete("/candidates/james_patel")
    assert (analyst_resume_dir / "sarah_wilson.pdf").exists()
    assert (analyst_resume_dir / "nina_volkov.pdf").exists()


@pytest.mark.parametrize(
    "reference", ["james_patel", "James Patel", "james_patel.pdf", "JAMES_PATEL"]
)
def test_a_candidate_can_be_deleted_by_any_of_its_names(api_client, reference: str):
    assert api_client.delete(f"/candidates/{reference}").status_code == 200
    assert "james_patel" not in _ids(api_client)


def test_deleting_an_unknown_candidate_is_a_404(api_client):
    response = api_client.delete("/candidates/nobody_at_all")
    assert response.status_code == 404
    assert response.json()["code"] == "candidate_not_found"


def test_deleting_the_same_candidate_twice_is_a_404(api_client):
    assert api_client.delete("/candidates/james_patel").status_code == 200
    assert api_client.delete("/candidates/james_patel").status_code == 404


# --------------------------------------------------------------------------
# Delete several
# --------------------------------------------------------------------------


def test_several_candidates_can_be_deleted_at_once(api_client):
    body = api_client.post(
        "/candidates/delete", json={"candidates": ["james_patel", "nina_volkov"]}
    ).json()

    assert set(body["deleted"]) == {"james_patel", "nina_volkov"}
    assert body["remaining"] == 1
    assert _ids(api_client) == {"sarah_wilson"}


def test_a_batch_delete_reports_each_failure_without_abandoning_the_rest(api_client):
    """One bad name must not cost the caller the other deletions."""
    body = api_client.post(
        "/candidates/delete", json={"candidates": ["james_patel", "ghost", "nina_volkov"]}
    ).json()

    assert set(body["deleted"]) == {"james_patel", "nina_volkov"}
    assert len(body["failed"]) == 1
    assert "ghost" in body["failed"][0]
    assert body["remaining"] == 1


def test_an_empty_batch_is_rejected(api_client):
    assert api_client.post("/candidates/delete", json={"candidates": []}).status_code == 422


def test_a_missing_body_is_rejected(api_client):
    assert api_client.post("/candidates/delete").status_code == 422


@pytest.mark.parametrize(
    "reference", ["../../../etc/passwd", "..\\..\\evil.pdf", "data/resumes/sarah_wilson.pdf"]
)
def test_a_path_shaped_reference_is_rejected(api_client, reference: str):
    response = api_client.post("/candidates/delete", json={"candidates": [reference]})
    assert response.status_code == 422


def test_a_blank_reference_is_rejected(api_client):
    assert api_client.post("/candidates/delete", json={"candidates": ["   "]}).status_code == 422


# --------------------------------------------------------------------------
# Clear all
# --------------------------------------------------------------------------


def test_clearing_removes_every_candidate(api_client):
    body = api_client.delete("/candidates").json()

    assert set(body["deleted"]) == ALL_IDS
    assert body["remaining"] == 0


def test_clearing_empties_the_listing(api_client):
    api_client.delete("/candidates")
    listing = api_client.get("/candidates").json()

    assert listing["candidates"] == []
    assert listing["count"] == 0


def test_clearing_removes_the_files(api_client, analyst_resume_dir: Path):
    api_client.delete("/candidates")
    assert list(analyst_resume_dir.glob("*.pdf")) == []


def test_clearing_an_empty_pool_is_not_an_error(api_client):
    api_client.delete("/candidates")
    response = api_client.delete("/candidates")

    assert response.status_code == 200
    assert response.json()["deleted"] == []


def test_clearing_leaves_non_candidate_files_alone(api_client, analyst_resume_dir: Path):
    """Only pooled candidates are deleted, never whatever else lives there."""
    keep = analyst_resume_dir / "notes.txt"
    keep.write_text("not a resume", encoding="utf-8")

    api_client.delete("/candidates")

    assert keep.exists()


def test_clearing_a_missing_directory_is_a_404(tmp_path: Path, analyst_embedder):
    settings = Settings(resume_dir=tmp_path / "gone")
    service = ScreeningService(settings, embedder=analyst_embedder)

    with build_test_client(settings, service) as client:
        assert client.delete("/candidates").status_code == 404


# --------------------------------------------------------------------------
# The pool actually refreshes -- deleted candidates cannot come back
# --------------------------------------------------------------------------


def test_a_deleted_candidate_disappears_from_ranking(api_client):
    """The whole point: a cached FAISS index must not outlive its files."""
    assert "james_patel" in _rank(api_client)

    api_client.delete("/candidates/james_patel")

    assert "james_patel" not in _rank(api_client)


def test_a_deleted_candidate_disappears_even_after_being_ranked(api_client):
    """Ranking first builds the index; the delete has to invalidate it."""
    _rank(api_client)
    api_client.delete("/candidates/james_patel")

    ranked = _rank(api_client)
    assert ranked == ALL_IDS - {"james_patel"}


def test_a_deleted_candidate_cannot_be_analysed(api_client):
    response = api_client.post(
        "/analyze-candidate",
        json={"candidate": "james_patel", "job_description": ANALYST_JOB_DESCRIPTION},
    )
    assert response.status_code == 200

    api_client.delete("/candidates/james_patel")

    after = api_client.post(
        "/analyze-candidate",
        json={"candidate": "james_patel", "job_description": ANALYST_JOB_DESCRIPTION},
    )
    assert after.status_code == 404
    assert after.json()["code"] == "candidate_not_found"


def test_a_deleted_candidate_is_not_retrievable_as_evidence(api_client):
    """A stale retrieval index could leak a deleted person's resume text."""
    api_client.post(
        "/analyze-candidate",
        json={"candidate": "james_patel", "job_description": ANALYST_JOB_DESCRIPTION},
    )
    api_client.delete("/candidates/james_patel")

    body = api_client.post(
        "/analyze-candidate",
        json={"candidate": "sarah_wilson", "job_description": ANALYST_JOB_DESCRIPTION},
    ).json()

    assert {item["candidate_id"] for item in body["evidence"]} == {"sarah_wilson"}


def test_clearing_leaves_nothing_to_rank(api_client):
    api_client.delete("/candidates")

    response = api_client.post(
        "/match-candidates", json={"job_description": ANALYST_JOB_DESCRIPTION}
    )
    assert response.status_code == 404
    assert response.json()["code"] == "no_resumes"


def test_a_resume_can_be_re_added_after_being_deleted(api_client, valid_pdf_bytes: bytes):
    api_client.delete("/candidates/james_patel")

    api_client.post(
        "/upload-resume",
        files={"file": ("james_patel.pdf", valid_pdf_bytes, "application/pdf")},
        data={"store": "true"},
    )

    assert "james_patel" in _ids(api_client)
    assert "james_patel" in _rank(api_client)


# --------------------------------------------------------------------------
# Service-level behaviour
# --------------------------------------------------------------------------


@pytest.fixture
def service(api_settings, analyst_embedder) -> ScreeningService:
    from app.llm import FakeLLMProvider

    return ScreeningService(api_settings, embedder=analyst_embedder, llm=FakeLLMProvider())


def test_deleting_drops_the_cached_matcher(service):
    first = service.matcher()
    service.delete_candidate("james_patel")

    second = service.matcher()
    assert second is not first
    assert second.candidate_count == 2


def test_deleting_drops_the_cached_pipeline(service):
    first = service.pipeline()
    service.delete_candidate("james_patel")

    assert service.pipeline() is not first


def test_deleting_drops_the_cached_pool(service):
    first = service.load_pool()
    service.delete_candidate("james_patel")

    assert service.load_pool() is not first


def test_deleting_an_unknown_candidate_raises_not_found(service):
    with pytest.raises(NotFoundError):
        service.delete_candidate("nobody")


def test_a_candidate_outside_the_resume_directory_is_refused(service, tmp_path: Path):
    """Defence in depth: a pooled candidate should never point elsewhere."""
    from app.models import Candidate

    outside = tmp_path / "elsewhere.pdf"
    outside.write_bytes(b"%PDF-1.4")

    stray = Candidate("stray", "some text", "Stray", source_path=outside)

    with pytest.raises(BadRequestError) as info:
        service._resume_path_for(stray)

    assert info.value.code == "not_deletable"
    assert outside.exists()


def test_a_candidate_with_no_source_file_is_refused(service):
    from app.models import Candidate

    with pytest.raises(BadRequestError):
        service._resume_path_for(Candidate("memory_only", "text", "No File"))


def test_clear_returns_the_ids_it_removed(service):
    assert set(service.clear_candidates()) == ALL_IDS


def test_clear_on_an_empty_pool_returns_nothing(service):
    service.clear_candidates()
    assert service.clear_candidates() == []
