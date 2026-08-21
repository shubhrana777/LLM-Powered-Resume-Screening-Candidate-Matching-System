"""Tests for the shared error handling.

Two rules are enforced here, and both are about what a client is *not* told:

* Every error, whatever its origin, has the same ``{"detail", "code"}`` shape.
* No response ever carries a traceback, a filesystem path, an environment
  variable, or the text of a 5xx exception. Those go to the log instead.
"""

from __future__ import annotations

import pytest

from app.embeddings import EmbeddingError
from app.skill_taxonomy import TaxonomyError
from tests.conftest import ANALYST_JOB_DESCRIPTION

MATCH_BODY = {"job_description": ANALYST_JOB_DESCRIPTION}


# --------------------------------------------------------------------------
# Shape
# --------------------------------------------------------------------------


def test_an_unknown_route_is_a_404_in_the_standard_shape(api_client):
    response = api_client.get("/no-such-endpoint")
    assert response.status_code == 404
    assert set(response.json()) == {"detail", "code"}
    assert response.json()["code"] == "not_found"


def test_the_wrong_method_is_a_405(api_client):
    response = api_client.get("/match-candidates")
    assert response.status_code == 405
    assert response.json()["code"] == "method_not_allowed"


def test_a_domain_error_uses_the_standard_shape(api_client):
    response = api_client.post(
        "/analyze-candidate",
        json={"candidate": "who?", "job_description": ANALYST_JOB_DESCRIPTION},
    )
    assert response.status_code == 404
    assert set(response.json()) == {"detail", "code"}


def test_malformed_json_is_a_422(api_client):
    response = api_client.post(
        "/match-candidates",
        content=b"{not json at all",
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422
    assert response.json()["code"] == "validation_error"


def test_a_validation_error_lists_the_offending_fields(api_client):
    response = api_client.post("/match-candidates", json={"top_k": 3})
    body = response.json()

    assert body["code"] == "validation_error"
    assert body["errors"]
    assert set(body["errors"][0]) == {"field", "message", "type"}


def test_a_validation_error_does_not_echo_the_submitted_text(api_client):
    """Pydantic would normally echo the input; a job description can be long."""
    secret = "CONFIDENTIAL-INTERNAL-REQUISITION-9931"
    response = api_client.post(
        "/match-candidates", json={"job_description": secret, "top_k": 0}
    )

    assert response.status_code == 422
    assert secret not in response.text


# --------------------------------------------------------------------------
# Nothing leaks
# --------------------------------------------------------------------------


def test_an_unexpected_error_becomes_a_generic_500(lenient_api_client, api_service):
    def explode(*args, **kwargs):
        raise RuntimeError("connection string: postgres://admin:hunter2@10.0.0.4/prod")

    api_service.match = explode

    response = lenient_api_client.post("/match-candidates", json=MATCH_BODY)

    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error.", "code": "internal_error"}


def test_an_unexpected_error_does_not_leak_its_message(lenient_api_client, api_service):
    def explode(*args, **kwargs):
        raise RuntimeError("connection string: postgres://admin:hunter2@10.0.0.4/prod")

    api_service.match = explode

    assert "hunter2" not in lenient_api_client.post("/match-candidates", json=MATCH_BODY).text


def test_no_response_ever_contains_a_traceback(lenient_api_client, api_service):
    def explode(*args, **kwargs):
        raise RuntimeError("boom")

    api_service.match = explode

    body = lenient_api_client.post("/match-candidates", json=MATCH_BODY).text
    assert "Traceback" not in body
    assert "File \"" not in body


@pytest.mark.parametrize(
    "exception, expected_code",
    [
        (EmbeddingError("could not load C:\\models\\secret-weights\\model.bin"), "embedding_failed"),
        (TaxonomyError("bad taxonomy at C:\\config\\skills.json"), "taxonomy_failed"),
    ],
)
def test_a_server_side_domain_error_is_reported_without_its_detail(
    lenient_api_client, api_service, exception: Exception, expected_code: str
):
    def explode(*args, **kwargs):
        raise exception

    api_service.match = explode

    response = lenient_api_client.post("/match-candidates", json=MATCH_BODY)

    assert response.status_code == 500
    assert response.json()["code"] == expected_code
    assert "secret-weights" not in response.text
    assert "C:\\" not in response.text


def test_a_client_side_domain_error_keeps_its_useful_message(api_client):
    """4xx messages describe the request, so they are worth showing."""
    response = api_client.post(
        "/analyze-candidate",
        json={"candidate": "ghost", "job_description": ANALYST_JOB_DESCRIPTION},
    )
    assert "ghost" in response.json()["detail"]


def test_errors_are_logged_server_side(lenient_api_client, api_service, caplog):
    def explode(*args, **kwargs):
        raise RuntimeError("diagnostic detail the client never sees")

    api_service.match = explode

    with caplog.at_level("ERROR", logger="app.api.errors"):
        lenient_api_client.post("/match-candidates", json=MATCH_BODY)

    assert "diagnostic detail the client never sees" in caplog.text
