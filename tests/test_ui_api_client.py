"""Tests for the dashboard's API client.

Every test serves responses through an ``httpx`` mock transport, so the suite
exercises the real request-building and error-translation code without a server
and without a socket.

What matters here is that the three failure modes stay distinguishable. A
recruiter looking at "cannot reach the API" needs to start the backend; one
looking at "no candidate matching 'x'" needs to pick a different candidate. If
the client flattened both into one error the dashboard could not tell them apart.
"""

from __future__ import annotations

import json

import httpx
import pytest

from app.ui.api_client import (
    APIError,
    APITimeoutError,
    APIUnavailableError,
    ScreeningAPIClient,
)
from app.ui.config import UISettings

BASE_URL = "http://testserver:8000"


def make_client(handler, **overrides) -> ScreeningAPIClient:
    """Build a client whose requests are served by ``handler``."""
    settings = UISettings(api_base_url=BASE_URL, **overrides)
    return ScreeningAPIClient(settings, transport=httpx.MockTransport(handler))


def json_handler(payload, status_code: int = 200):
    """A handler returning one fixed JSON body."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, json=payload)

    return handler


def recording_handler(payload, status_code: int = 200):
    """A handler that records the requests it received."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        request.read()
        seen.append(request)
        return httpx.Response(status_code, json=payload)

    return handler, seen


# --------------------------------------------------------------------------
# Requests are built correctly
# --------------------------------------------------------------------------


def test_health_calls_the_health_endpoint():
    handler, seen = recording_handler({"status": "healthy"})
    with make_client(handler) as client:
        assert client.health() == {"status": "healthy"}

    assert seen[0].url.path == "/health"
    assert seen[0].method == "GET"


def test_the_base_url_is_used():
    handler, seen = recording_handler({"status": "healthy"})
    with make_client(handler) as client:
        client.health()

    assert str(seen[0].url).startswith(BASE_URL)


def test_a_trailing_slash_in_the_base_url_is_normalised():
    from app.ui.config import load_ui_settings

    import os

    os.environ["API_BASE_URL"] = "http://example.test:9000/"
    try:
        assert load_ui_settings().api_base_url == "http://example.test:9000"
    finally:
        del os.environ["API_BASE_URL"]


def test_list_candidates_calls_the_candidates_endpoint():
    handler, seen = recording_handler({"candidates": [], "count": 0})
    with make_client(handler) as client:
        client.list_candidates()

    assert seen[0].url.path == "/candidates"


def test_match_candidates_posts_the_job_description_and_top_k():
    handler, seen = recording_handler({"results": [], "count": 0})
    with make_client(handler) as client:
        client.match_candidates("Financial analyst wanted", 7)

    body = json.loads(seen[0].content)
    assert seen[0].url.path == "/match-candidates"
    assert body == {"job_description": "Financial analyst wanted", "top_k": 7}


def test_analyze_candidate_posts_the_candidate_and_job():
    handler, seen = recording_handler({"candidate_id": "sarah_wilson"})
    with make_client(handler) as client:
        client.analyze_candidate("sarah_wilson", "Financial analyst wanted")

    body = json.loads(seen[0].content)
    assert seen[0].url.path == "/analyze-candidate"
    assert body == {"candidate": "sarah_wilson", "job_description": "Financial analyst wanted"}


def test_upload_sends_multipart_with_the_store_flag():
    handler, seen = recording_handler({"filename": "cv.pdf", "stored": True})
    with make_client(handler) as client:
        client.upload_resume("cv.pdf", b"%PDF-1.4 fake", store=True)

    request = seen[0]
    assert request.url.path == "/upload-resume"
    assert "multipart/form-data" in request.headers["content-type"]
    assert b"%PDF-1.4 fake" in request.content
    assert b'name="store"' in request.content
    assert b"true" in request.content


def test_upload_can_ask_the_backend_not_to_store():
    handler, seen = recording_handler({"filename": "cv.pdf", "stored": False})
    with make_client(handler) as client:
        client.upload_resume("cv.pdf", b"%PDF-1.4 fake", store=False)

    assert b"false" in seen[0].content


def test_the_dashboard_stores_uploads_by_default():
    """A discarded upload could never be ranked, so storing is the default."""
    handler, seen = recording_handler({"filename": "cv.pdf", "stored": True})
    with make_client(handler) as client:
        client.upload_resume("cv.pdf", b"%PDF-1.4 fake")

    assert b"true" in seen[0].content


def test_analysis_uses_the_longer_timeout():
    """Analysis reaches a model; it must not share the quick request timeout."""
    seen: list[httpx.Timeout] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.extensions.get("timeout"))
        return httpx.Response(200, json={"candidate_id": "x"})

    with make_client(handler, timeout_seconds=5.0, analysis_timeout_seconds=99.0) as client:
        client.analyze_candidate("x", "job")

    assert seen[0]["read"] == 99.0


def test_ordinary_requests_use_the_short_timeout():
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.extensions.get("timeout"))
        return httpx.Response(200, json={"count": 0})

    with make_client(handler, timeout_seconds=5.0, analysis_timeout_seconds=99.0) as client:
        client.list_candidates()

    assert seen[0]["read"] == 5.0


# --------------------------------------------------------------------------
# Failures stay distinguishable
# --------------------------------------------------------------------------


def test_a_connection_failure_is_reported_as_unavailable():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    with make_client(handler) as client:
        with pytest.raises(APIUnavailableError) as info:
            client.health()

    assert BASE_URL in info.value.message


def test_a_timeout_is_reported_as_a_timeout():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("too slow", request=request)

    with make_client(handler) as client:
        with pytest.raises(APITimeoutError):
            client.list_candidates()


def test_a_timeout_explains_the_first_request_is_slow():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("too slow", request=request)

    with make_client(handler) as client:
        with pytest.raises(APITimeoutError) as info:
            client.list_candidates()

    assert "embedding model" in info.value.message


@pytest.mark.parametrize("status_code", [400, 404, 413, 500, 502])
def test_an_error_status_becomes_an_api_error(status_code: int):
    handler = json_handler({"detail": "something went wrong", "code": "oops"}, status_code)

    with make_client(handler) as client:
        with pytest.raises(APIError) as info:
            client.list_candidates()

    assert info.value.status_code == status_code
    assert info.value.code == "oops"


def test_the_backend_message_is_carried_through():
    """Backend 4xx messages are written for a human and contain no internals."""
    handler = json_handler(
        {"detail": "No candidate matching 'ghost'.", "code": "candidate_not_found"}, 404
    )

    with make_client(handler) as client:
        with pytest.raises(APIError) as info:
            client.analyze_candidate("ghost", "job")

    assert info.value.message == "No candidate matching 'ghost'."
    assert info.value.is_not_found is True


def test_a_non_404_is_not_reported_as_not_found():
    handler = json_handler({"detail": "bad", "code": "bad_request"}, 400)

    with make_client(handler) as client:
        with pytest.raises(APIError) as info:
            client.list_candidates()

    assert info.value.is_not_found is False


def test_validation_details_are_extracted():
    handler = json_handler(
        {
            "detail": "Request validation failed.",
            "code": "validation_error",
            "errors": [
                {"field": "body.job_description", "message": "Field required", "type": "missing"}
            ],
        },
        422,
    )

    with make_client(handler) as client:
        with pytest.raises(APIError) as info:
            client.match_candidates("", 5)

    assert info.value.details == ("body.job_description: Field required",)


def test_an_error_body_that_is_not_json_does_not_crash():
    """A proxy can return HTML; the client must not show it or fail on it."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(502, text="<html><body>Bad Gateway</body></html>")

    with make_client(handler) as client:
        with pytest.raises(APIError) as info:
            client.list_candidates()

    assert "502" in info.value.message
    assert "<html>" not in info.value.message


def test_a_success_body_that_is_not_json_is_reported():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="not json")

    with make_client(handler) as client:
        with pytest.raises(APIError) as info:
            client.health()

    assert info.value.code == "invalid_response"


def test_a_success_body_that_is_not_an_object_is_reported():
    with make_client(json_handler([1, 2, 3])) as client:
        with pytest.raises(APIError) as info:
            client.health()

    assert info.value.code == "invalid_response"


# --------------------------------------------------------------------------
# Availability probe
# --------------------------------------------------------------------------


def test_is_available_is_true_when_health_succeeds():
    with make_client(json_handler({"status": "healthy"})) as client:
        assert client.is_available() is True


def test_is_available_is_false_when_unreachable():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with make_client(handler) as client:
        assert client.is_available() is False


def test_is_available_is_false_on_an_error_status():
    with make_client(json_handler({"detail": "no"}, 500)) as client:
        assert client.is_available() is False


def test_is_available_never_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    with make_client(handler) as client:
        assert client.is_available() is False


# --------------------------------------------------------------------------
# The client is a client
# --------------------------------------------------------------------------


def test_the_ui_never_imports_backend_internals():
    """The dashboard must talk HTTP, not reach into the FastAPI service."""
    import pathlib

    forbidden = (
        "app.api.service",
        "app.api.dependencies",
        "app.matching",
        "app.rag_pipeline",
        "app.embeddings",
        "app.retriever",
        "app.llm",
        "app.resume_parser",
    )

    for path in pathlib.Path("app/ui").glob("*.py"):
        source = path.read_text(encoding="utf-8")
        for module in forbidden:
            assert f"import {module}" not in source, f"{path.name} imports {module}"
            assert f"from {module}" not in source, f"{path.name} imports from {module}"
