"""The only module in the dashboard that speaks HTTP.

Everything the UI knows about the backend passes through here. Pages call
methods and receive plain dictionaries matching the API's documented response
models; they never see a URL, a status code, a header or an ``httpx`` object.
That boundary is what keeps the dashboard a client rather than a second
implementation, and it is what makes the UI testable without a running server.

Errors
------
Three things can go wrong, and a recruiter needs to tell them apart:

:class:`APIUnavailableError`
    The backend could not be reached at all -- not started, wrong port, network
    gone. The dashboard shows how to start it.
:class:`APITimeoutError`
    It was reached but took too long. Usually the first request, which loads the
    embedding model, or a slow model provider.
:class:`APIError`
    It answered with a refusal. The backend's own ``detail`` and ``code`` are
    carried through, because those messages are written for a human and are
    already guaranteed to contain no paths or internals.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import httpx

from app.ui.config import UISettings, load_ui_settings

__all__ = [
    "APIClientError",
    "APIError",
    "APITimeoutError",
    "APIUnavailableError",
    "ScreeningAPIClient",
]

logger = logging.getLogger(__name__)


class APIClientError(Exception):
    """Base class for every failure this module reports."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class APIUnavailableError(APIClientError):
    """The backend could not be reached."""


class APITimeoutError(APIClientError):
    """The backend was reached but did not answer in time."""


class APIError(APIClientError):
    """The backend answered with an error status.

    Attributes:
        status_code: The HTTP status returned.
        code: The backend's machine-readable error code, when it sent one.
        details: Per-field validation errors, when the status was 422.
    """

    def __init__(
        self,
        message: str,
        status_code: int,
        code: str = "error",
        details: tuple[str, ...] = (),
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.details = details

    @property
    def is_not_found(self) -> bool:
        """Whether the backend reported the resource does not exist."""
        return self.status_code == 404


def _describe_error(response: httpx.Response) -> APIError:
    """Turn an error response into an :class:`APIError`.

    The backend answers every error as ``{"detail", "code"}``, adding an
    ``errors`` list on 422. Anything else -- a proxy's HTML error page, say --
    falls back to a generic message rather than showing the body, which could be
    anything.

    Args:
        response: The non-2xx response.

    Returns:
        The corresponding :class:`APIError`.
    """
    detail = f"The server returned HTTP {response.status_code}."
    code = "error"
    details: tuple[str, ...] = ()

    try:
        payload = response.json()
    except ValueError:
        payload = None

    if isinstance(payload, dict):
        detail = str(payload.get("detail") or detail)
        code = str(payload.get("code") or code)
        raw_errors = payload.get("errors")
        if isinstance(raw_errors, list):
            details = tuple(
                f"{item.get('field', '?')}: {item.get('message', '')}".strip()
                for item in raw_errors
                if isinstance(item, dict)
            )

    return APIError(detail, status_code=response.status_code, code=code, details=details)


class ScreeningAPIClient:
    """A typed-ish wrapper over the resume screening REST API.

    Args:
        settings: Where the backend is and how long to wait. Defaults to the
            environment.
        transport: Optional ``httpx`` transport, so tests can serve responses
            without a socket.
    """

    def __init__(
        self,
        settings: UISettings | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._settings = settings if settings is not None else load_ui_settings()
        self._client = httpx.Client(
            base_url=self._settings.api_base_url,
            timeout=self._settings.timeout_seconds,
            transport=transport,
        )

    @property
    def settings(self) -> UISettings:
        """The configuration this client was built with."""
        return self._settings

    @property
    def base_url(self) -> str:
        """The backend root this client talks to."""
        return self._settings.api_base_url

    def close(self) -> None:
        """Release the underlying connection pool."""
        self._client.close()

    def __enter__(self) -> ScreeningAPIClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        """Send one request and return its decoded body.

        Args:
            method: HTTP method.
            path: Path relative to the backend root.
            **kwargs: Passed through to ``httpx``.

        Returns:
            The decoded JSON object.

        Raises:
            APIUnavailableError: If the backend could not be reached.
            APITimeoutError: If it did not answer in time.
            APIError: If it answered with an error status or an unreadable body.
        """
        try:
            response = self._client.request(method, path, **kwargs)
        except httpx.TimeoutException as exc:
            logger.warning("%s %s timed out", method, path)
            raise APITimeoutError(
                "The server did not respond in time. The first request after "
                "start-up loads the embedding model, which can take a minute."
            ) from exc
        except httpx.RequestError as exc:
            logger.warning("%s %s could not be sent: %s", method, path, type(exc).__name__)
            raise APIUnavailableError(
                f"Could not reach the API at {self.base_url}."
            ) from exc

        if response.is_error:
            raise _describe_error(response)

        try:
            payload = response.json()
        except ValueError as exc:
            raise APIError(
                "The server sent a response that could not be read.",
                status_code=response.status_code,
                code="invalid_response",
            ) from exc

        if not isinstance(payload, dict):
            raise APIError(
                "The server sent a response in an unexpected shape.",
                status_code=response.status_code,
                code="invalid_response",
            )

        return payload

    # -- Endpoints ---------------------------------------------------------

    def health(self) -> dict[str, Any]:
        """``GET /health`` -- liveness and service information."""
        return self._request("GET", "/health")

    def is_available(self) -> bool:
        """Whether the backend answers a health check.

        Returns:
            ``True`` if it is up, ``False`` for any failure. Used for the status
            indicator, where the reason matters less than the fact.
        """
        try:
            self.health()
        except APIClientError:
            return False
        return True

    def list_candidates(self) -> dict[str, Any]:
        """``GET /candidates`` -- the pool the backend can rank."""
        return self._request("GET", "/candidates")

    def upload_resume(
        self,
        filename: str,
        content: bytes,
        store: bool = True,
    ) -> dict[str, Any]:
        """``POST /upload-resume`` -- add one PDF to the pool.

        Args:
            filename: Name to report; the backend derives a safe id from it.
            content: The PDF bytes.
            store: Keep the resume as a rankable candidate. The dashboard sends
                ``True``, since a discarded upload could never be screened.

        Returns:
            The upload result, including ``candidate_id`` when it was stored.
        """
        return self._request(
            "POST",
            "/upload-resume",
            files={"file": (filename, content, "application/pdf")},
            data={"store": str(bool(store)).lower()},
        )

    def delete_candidate(self, candidate: str) -> dict[str, Any]:
        """``DELETE /candidates/{id}`` -- remove one candidate from the pool.

        Args:
            candidate: Candidate id, display name or file name. Resolved by the
                backend against its own pool; never a path.

        Returns:
            What was deleted and how many candidates remain.
        """
        return self._request("DELETE", f"/candidates/{candidate}")

    def delete_candidates(self, candidates: Sequence[str]) -> dict[str, Any]:
        """``POST /candidates/delete`` -- remove several candidates.

        Args:
            candidates: Candidate ids, names or file names.

        Returns:
            What was deleted, what was not, and how many remain.
        """
        return self._request(
            "POST", "/candidates/delete", json={"candidates": list(candidates)}
        )

    def clear_candidates(self) -> dict[str, Any]:
        """``DELETE /candidates`` -- empty the candidate pool.

        Returns:
            Every id that was removed, and a remaining count.
        """
        return self._request("DELETE", "/candidates")

    def match_candidates(self, job_description: str, top_k: int) -> dict[str, Any]:
        """``POST /match-candidates`` -- rank the pool.

        Args:
            job_description: The job description text.
            top_k: How many candidates to return.

        Returns:
            The ranking, with the backend's note on what the score means.
        """
        return self._request(
            "POST",
            "/match-candidates",
            json={"job_description": job_description, "top_k": top_k},
        )

    def analyze_candidate(self, candidate: str, job_description: str) -> dict[str, Any]:
        """``POST /analyze-candidate`` -- full analysis of one candidate.

        Uses the longer analysis timeout, because this is the call that reaches
        a language model.

        Args:
            candidate: Candidate id, display name or file name.
            job_description: The job description text.

        Returns:
            The validated analysis, with the evidence it rests on.
        """
        return self._request(
            "POST",
            "/analyze-candidate",
            json={"candidate": candidate, "job_description": job_description},
            timeout=self._settings.analysis_timeout_seconds,
        )
