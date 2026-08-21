"""Tests for the API service layer, independent of HTTP.

The service adds caching and candidate resolution on top of the Phase 1-4
components. These tests check that the caching is correct -- it rebuilds when
the directory changes and not otherwise -- and that resolving a candidate can
only ever reach the pool.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.api.config import (
    DEFAULT_CORS_ORIGINS,
    DEFAULT_MAX_UPLOAD_BYTES,
    DEFAULT_RESUME_DIR,
    Settings,
    load_settings,
)
from app.api.errors import NotFoundError
from app.api.service import ScreeningService, scrub_path
from app.embeddings import DEFAULT_MODEL_NAME
from tests.conftest import ANALYST_JOB_DESCRIPTION


@pytest.fixture
def service(api_settings, analyst_embedder) -> ScreeningService:
    from app.llm import FakeLLMProvider

    return ScreeningService(api_settings, embedder=analyst_embedder, llm=FakeLLMProvider())


# --------------------------------------------------------------------------
# Pool loading and caching
# --------------------------------------------------------------------------


def test_the_pool_holds_every_readable_resume(service):
    assert len(service.load_pool().candidates) == 3


def test_the_pool_is_reused_when_nothing_changed(service):
    assert service.load_pool() is service.load_pool()


def test_the_pool_is_rebuilt_when_a_resume_is_added(service, analyst_resume_dir, valid_pdf):
    first = service.load_pool()
    (analyst_resume_dir / "new_person.pdf").write_bytes(valid_pdf.read_bytes())

    second = service.load_pool()

    assert second is not first
    assert len(second.candidates) == 4


def test_the_pool_is_rebuilt_when_a_resume_is_removed(service, analyst_resume_dir):
    service.load_pool()
    (analyst_resume_dir / "nina_volkov.pdf").unlink()

    assert len(service.load_pool().candidates) == 2


def test_the_matcher_is_rebuilt_when_the_pool_changes(service, analyst_resume_dir, valid_pdf):
    first = service.matcher()
    (analyst_resume_dir / "new_person.pdf").write_bytes(valid_pdf.read_bytes())

    second = service.matcher()

    assert second is not first
    assert second.candidate_count == 4


def test_the_matcher_is_reused_when_nothing_changed(service):
    assert service.matcher() is service.matcher()


def test_the_pipeline_is_reused_when_nothing_changed(service):
    assert service.pipeline() is service.pipeline()


def test_the_pipeline_is_rebuilt_when_the_pool_changes(service, analyst_resume_dir):
    first = service.pipeline()
    (analyst_resume_dir / "nina_volkov.pdf").unlink()

    assert service.pipeline() is not first


def test_an_unreadable_file_is_recorded_not_raised(service, analyst_resume_dir):
    (analyst_resume_dir / "broken.pdf").write_bytes(b"garbage")

    pool = service.load_pool()

    assert len(pool.candidates) == 3
    assert len(pool.failures) == 1


# --------------------------------------------------------------------------
# Missing and empty directories
# --------------------------------------------------------------------------


def test_a_missing_directory_raises_not_found(tmp_path: Path, analyst_embedder):
    service = ScreeningService(Settings(resume_dir=tmp_path / "gone"), embedder=analyst_embedder)

    with pytest.raises(NotFoundError) as info:
        service.load_pool()

    assert info.value.status_code == 404
    assert info.value.code == "resume_directory_not_found"


def test_an_empty_directory_loads_but_cannot_be_matched(tmp_path: Path, analyst_embedder):
    empty = tmp_path / "empty"
    empty.mkdir()
    service = ScreeningService(Settings(resume_dir=empty), embedder=analyst_embedder)

    assert service.load_pool().candidates == ()

    with pytest.raises(NotFoundError) as info:
        service.require_pool()

    assert info.value.code == "no_resumes"


# --------------------------------------------------------------------------
# Candidate resolution
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "reference",
    ["sarah_wilson", "  sarah_wilson  ", "SARAH_WILSON", "Sarah Wilson", "sarah_wilson.pdf"],
)
def test_a_candidate_resolves_by_id_name_or_filename(service, reference: str):
    assert service.resolve_candidate(reference).candidate_id == "sarah_wilson"


def test_an_unknown_reference_raises_not_found(service):
    with pytest.raises(NotFoundError) as info:
        service.resolve_candidate("someone else")

    assert info.value.code == "candidate_not_found"


@pytest.mark.parametrize(
    "reference",
    ["../nina_volkov", "data/resumes/sarah_wilson.pdf", "C:\\resumes\\sarah_wilson.pdf"],
)
def test_a_path_shaped_reference_resolves_to_nothing(service, reference: str):
    """Resolution is a lookup in the pool, never a filesystem operation."""
    with pytest.raises(NotFoundError):
        service.resolve_candidate(reference)


def test_the_unknown_candidate_message_names_the_available_ones(service):
    with pytest.raises(NotFoundError) as info:
        service.resolve_candidate("nobody")

    assert "sarah_wilson" in info.value.detail


# --------------------------------------------------------------------------
# Delegation
# --------------------------------------------------------------------------


def test_match_delegates_to_the_phase_2_engine(service):
    results, considered = service.match(ANALYST_JOB_DESCRIPTION, top_k=2)

    assert considered == 3
    assert [result.rank for result in results] == [1, 2]
    assert results[0].candidate_id == "sarah_wilson"


def test_analyze_delegates_to_the_phase_4_pipeline(service):
    analysis = service.analyze("Sarah Wilson", ANALYST_JOB_DESCRIPTION)

    assert analysis.candidate_id == "sarah_wilson"
    assert analysis.evidence
    assert all(item.candidate_id == "sarah_wilson" for item in analysis.evidence)


def test_analyze_refuses_a_candidate_outside_the_pool(service):
    with pytest.raises(NotFoundError):
        service.analyze("someone_else", ANALYST_JOB_DESCRIPTION)


# --------------------------------------------------------------------------
# Path scrubbing
# --------------------------------------------------------------------------


def test_scrub_path_replaces_a_full_path_with_the_filename():
    path = Path("/srv/app/data/resumes/jane.pdf")
    message = f"Could not open PDF: {path} (bad xref)"

    assert scrub_path(message, path) == "Could not open PDF: jane.pdf (bad xref)"


def test_scrub_path_replaces_a_bare_directory():
    path = Path("/srv/app/data/resumes/jane.pdf")

    assert "/srv/app" not in scrub_path(f"somewhere under {path.parent}", path)


def test_scrub_path_leaves_an_unrelated_message_alone():
    assert scrub_path("no paths here", Path("/srv/x.pdf")) == "no paths here"


# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------


def test_settings_default_to_the_project_layout(monkeypatch):
    for name in (
        "RESUME_DIR",
        "EMBEDDING_MODEL",
        "LLM_PROVIDER",
        "API_CORS_ORIGINS",
        "API_MAX_UPLOAD_BYTES",
        "API_LOG_LEVEL",
    ):
        monkeypatch.delenv(name, raising=False)

    settings = load_settings()

    assert settings.resume_dir == DEFAULT_RESUME_DIR
    assert settings.embedding_model == DEFAULT_MODEL_NAME
    assert settings.llm_provider == "fake"
    assert settings.max_upload_bytes == DEFAULT_MAX_UPLOAD_BYTES
    assert settings.cors_origins == DEFAULT_CORS_ORIGINS


def test_settings_read_the_environment(monkeypatch):
    monkeypatch.setenv("RESUME_DIR", "somewhere/else")
    monkeypatch.setenv("API_MAX_UPLOAD_BYTES", "2048")
    monkeypatch.setenv("API_CORS_ORIGINS", "http://a.test, http://b.test")

    settings = load_settings()

    assert settings.resume_dir == Path("somewhere/else")
    assert settings.max_upload_bytes == 2048
    assert settings.cors_origins == ("http://a.test", "http://b.test")


@pytest.mark.parametrize("value", ["not a number", "0", "-5", ""])
def test_an_unusable_upload_limit_falls_back_to_the_default(monkeypatch, value: str):
    """A typo in an optional variable must not stop the server from starting."""
    monkeypatch.setenv("API_MAX_UPLOAD_BYTES", value)

    assert load_settings().max_upload_bytes == DEFAULT_MAX_UPLOAD_BYTES


def test_a_wildcard_origin_is_recognised_as_permissive(monkeypatch):
    monkeypatch.setenv("API_CORS_ORIGINS", "*")

    assert load_settings().allows_any_origin is True


def test_the_default_origins_are_not_permissive():
    assert Settings().allows_any_origin is False


def test_settings_never_carry_a_credential(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "sk-ant-should-never-appear")

    assert "sk-ant-should-never-appear" not in repr(load_settings())
