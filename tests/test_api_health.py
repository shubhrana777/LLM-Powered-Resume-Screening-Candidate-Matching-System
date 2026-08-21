"""Tests for GET /health and the generated documentation."""

from __future__ import annotations

from pathlib import Path

import pytest

from app import __version__
from app.api.config import Settings
from app.api.schemas import HealthResponse
from tests.conftest import build_test_client


def test_health_returns_200(api_client):
    assert api_client.get("/health").status_code == 200


def test_health_reports_healthy(api_client):
    assert api_client.get("/health").json()["status"] == "healthy"


def test_health_response_matches_its_schema(api_client):
    # Round-tripping through the model proves the documented schema is the
    # shape actually served, not just the shape declared.
    payload = HealthResponse.model_validate(api_client.get("/health").json())
    assert payload.status == "healthy"


def test_health_reports_the_service_name(api_client):
    assert api_client.get("/health").json()["service"] == "resume-screening-api"


def test_health_reports_the_application_version(api_client):
    assert api_client.get("/health").json()["version"] == __version__


def test_health_reports_the_configured_provider(api_client):
    assert api_client.get("/health").json()["llm_provider"] == "fake"


def test_health_never_reports_a_key(api_client):
    body = api_client.get("/health").text.lower()
    assert "key" not in body
    assert "token" not in body


def test_health_works_without_a_resume_directory(api_service, tmp_path: Path):
    """Liveness must not depend on the resume directory being present."""
    settings = Settings(resume_dir=tmp_path / "does-not-exist")
    with build_test_client(settings, api_service) as client:
        assert client.get("/health").status_code == 200


def test_health_loads_no_model(api_settings, api_service):
    """A liveness check must stay cheap: no embedding, no parsing."""

    def explode(*args, **kwargs):
        raise AssertionError("/health must not touch the screening service")

    api_service.load_pool = explode
    api_service.matcher = explode

    with build_test_client(api_settings, api_service) as client:
        assert client.get("/health").status_code == 200


# --------------------------------------------------------------------------
# Generated documentation
# --------------------------------------------------------------------------


def test_swagger_ui_is_served(api_client):
    response = api_client.get("/docs")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_redoc_is_served(api_client):
    assert api_client.get("/redoc").status_code == 200


def test_openapi_schema_is_served(api_client):
    assert api_client.get("/openapi.json").status_code == 200


@pytest.mark.parametrize(
    "path",
    ["/health", "/upload-resume", "/match-candidates", "/analyze-candidate", "/candidates"],
)
def test_every_endpoint_is_documented(api_client, path: str):
    assert path in api_client.get("/openapi.json").json()["paths"]


def test_openapi_carries_project_metadata(api_client):
    info = api_client.get("/openapi.json").json()["info"]
    assert info["title"] == "Resume Screening & Candidate Matching API"
    assert info["version"] == __version__
    assert info["description"].strip()


def test_openapi_documents_that_the_score_is_not_a_probability(api_client):
    schema = api_client.get("/openapi.json").json()
    description = schema["info"]["description"].lower()
    assert "not a probability" in description


def test_every_operation_has_a_summary(api_client):
    schema = api_client.get("/openapi.json").json()
    for path, operations in schema["paths"].items():
        for method, operation in operations.items():
            assert operation.get("summary"), f"{method.upper()} {path} has no summary"
            assert operation.get("description"), f"{method.upper()} {path} has no description"


def test_operations_are_tagged(api_client):
    schema = api_client.get("/openapi.json").json()
    tags = {tag["name"] for tag in schema.get("tags", [])}
    assert {"system", "resumes", "matching", "analysis"} <= tags
