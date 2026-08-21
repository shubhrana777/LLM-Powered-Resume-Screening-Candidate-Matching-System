"""Consistent API error responses.

Every error the client sees has the same JSON shape::

    {"detail": "human-readable message", "code": "machine_readable_code"}

Three kinds of failure reach a client:

**Deliberate** -- :class:`APIError` and its subclasses, raised by the routes and
the service layer. The status code and the message are chosen on purpose.

**Domain** -- exceptions from the Phase 1-4 modules. They are translated by
handlers registered here rather than by ``try``/``except`` in every route, so
the routes stay thin and one exception cannot be mapped two different ways in
two different places.

**Unexpected** -- anything else. The client gets a fixed ``500`` message; the
traceback goes to the log and never into the response.

What is deliberately never sent to a client
-------------------------------------------
Tracebacks, filesystem paths (including temporary upload paths), API keys, and
the text of any 5xx exception. Server-side messages are logged in full; the
client gets a fixed sentence per error code. Messages for 4xx errors are the
exception's own text only where that text is known to describe the *request*
rather than the server -- the resume-parser errors all embed the file path they
were given, so they are given fixed wording instead.
"""

from __future__ import annotations

import logging
from typing import NamedTuple

from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.requests import Request
from starlette.responses import Response

from app.analysis_parser import AnalysisParseError
from app.chunker import ChunkingError
from app.embeddings import EmbeddingError
from app.llm import LLMCallError, LLMConfigurationError, LLMError
from app.matching import (
    DuplicateCandidateError,
    EmptyCandidateListError,
    EmptyJobDescriptionError,
    MatchingError,
    NoCandidatesIndexedError,
)
from app.models import InvalidCandidateError
from app.rag_context import ContextIsolationError
from app.resume_parser import (
    CorruptedPDFError,
    NoExtractableTextError,
    ResumeFileNotFoundError,
    ResumeParserError,
    UnsupportedFileTypeError,
)
from app.retriever import RetrievalError, UnknownCandidateError
from app.skill_taxonomy import TaxonomyError
from app.vector_store import VectorStoreError

__all__ = [
    "APIError",
    "BadRequestError",
    "NotFoundError",
    "PayloadTooLargeError",
    "UpstreamError",
    "install_error_handlers",
]

logger = logging.getLogger(__name__)

GENERIC_SERVER_MESSAGE = "Internal server error."


class APIError(Exception):
    """An error with a chosen HTTP status code and machine-readable code.

    Args:
        detail: Message for the client. Must be safe to show: no paths, no
            secrets, no exception internals.
        status_code: Override for the class default.
        code: Override for the class default.
    """

    status_code = 500
    code = "internal_error"

    def __init__(
        self,
        detail: str,
        *,
        status_code: int | None = None,
        code: str | None = None,
    ) -> None:
        super().__init__(detail)
        self.detail = detail
        if status_code is not None:
            self.status_code = status_code
        if code is not None:
            self.code = code


class BadRequestError(APIError):
    """The request itself is wrong: bad file, unusable content."""

    status_code = 400
    code = "bad_request"


class NotFoundError(APIError):
    """A requested candidate or resource does not exist."""

    status_code = 404
    code = "not_found"


class PayloadTooLargeError(APIError):
    """The upload exceeds the configured size ceiling."""

    status_code = 413
    code = "payload_too_large"


class UpstreamError(APIError):
    """A dependency the API calls out to failed, e.g. the LLM provider."""

    status_code = 502
    code = "upstream_error"


class _Mapping(NamedTuple):
    """How one domain exception becomes an HTTP response.

    Attributes:
        status_code: HTTP status to return.
        code: Machine-readable error code.
        message: Fixed client-facing message, or ``None`` to use the
            exception's own text. Only use ``None`` where that text is known to
            describe the request and to contain no server internals.
    """

    status_code: int
    code: str
    message: str | None


# Order matters: the first entry whose class appears in the exception's MRO
# wins, so subclasses must be listed before the bases they specialise.
_DOMAIN_ERRORS: tuple[tuple[type[Exception], _Mapping], ...] = (
    # --- Phase 1: resume parsing. Every message embeds the file path, so all
    # of them get fixed wording instead of the exception text.
    (
        UnsupportedFileTypeError,
        _Mapping(400, "unsupported_file_type", "Only PDF files are supported."),
    ),
    (
        CorruptedPDFError,
        _Mapping(400, "invalid_pdf", "The file could not be opened as a PDF."),
    ),
    (
        NoExtractableTextError,
        _Mapping(
            400,
            "no_extractable_text",
            "No selectable text could be extracted from the PDF. Scanned "
            "images need OCR, which this system does not do.",
        ),
    ),
    (
        ResumeFileNotFoundError,
        _Mapping(404, "not_found", "The requested resume file was not found."),
    ),
    (
        ResumeParserError,
        _Mapping(400, "resume_parse_failed", "The resume could not be read."),
    ),
    # --- Phase 2: matching. These messages describe the request.
    (EmptyJobDescriptionError, _Mapping(400, "empty_job_description", None)),
    (InvalidCandidateError, _Mapping(400, "invalid_candidate", None)),
    (EmptyCandidateListError, _Mapping(404, "no_candidates", None)),
    (
        DuplicateCandidateError,
        _Mapping(
            500,
            "duplicate_candidate",
            "The configured resume directory contains duplicate candidate ids.",
        ),
    ),
    (
        NoCandidatesIndexedError,
        _Mapping(500, "not_indexed", GENERIC_SERVER_MESSAGE),
    ),
    (MatchingError, _Mapping(400, "matching_failed", None)),
    (
        EmbeddingError,
        _Mapping(500, "embedding_failed", "The embedding model could not be used."),
    ),
    (
        VectorStoreError,
        _Mapping(500, "vector_store_failed", GENERIC_SERVER_MESSAGE),
    ),
    # --- Phase 3
    (
        TaxonomyError,
        _Mapping(500, "taxonomy_failed", "The skill taxonomy could not be loaded."),
    ),
    # --- Phase 4: RAG and LLM.
    (UnknownCandidateError, _Mapping(404, "candidate_not_found", None)),
    (
        RetrievalError,
        _Mapping(500, "retrieval_failed", "Evidence retrieval failed."),
    ),
    (ChunkingError, _Mapping(400, "chunking_failed", None)),
    (
        ContextIsolationError,
        _Mapping(
            500,
            "context_isolation",
            "The analysis was stopped because a candidate isolation check failed.",
        ),
    ),
    (
        AnalysisParseError,
        _Mapping(
            502,
            "llm_response_invalid",
            "The language model returned a response that could not be validated.",
        ),
    ),
    (
        LLMConfigurationError,
        _Mapping(
            500,
            "llm_not_configured",
            "The language model provider is not configured on this server.",
        ),
    ),
    (
        LLMCallError,
        _Mapping(502, "llm_call_failed", "The language model provider could not be reached."),
    ),
    (LLMError, _Mapping(502, "llm_error", "The language model call failed.")),
)

# Every domain exception class the handlers are registered for.
DOMAIN_EXCEPTION_TYPES: tuple[type[Exception], ...] = tuple(
    entry[0] for entry in _DOMAIN_ERRORS
)

_STATUS_CODES = {
    400: "bad_request",
    404: "not_found",
    405: "method_not_allowed",
    413: "payload_too_large",
    422: "validation_error",
    500: "internal_error",
}


def _mapping_for(exc: Exception) -> _Mapping:
    """Find the mapping for ``exc``, walking its MRO in registration order."""
    for exception_type, mapping in _DOMAIN_ERRORS:
        if isinstance(exc, exception_type):
            return mapping
    return _Mapping(500, "internal_error", GENERIC_SERVER_MESSAGE)


def _error_response(
    status_code: int,
    code: str,
    detail: str,
    extra: dict[str, object] | None = None,
) -> JSONResponse:
    """Build the standard error body."""
    payload: dict[str, object] = {"detail": detail, "code": code}
    if extra:
        payload.update(extra)
    return JSONResponse(status_code=status_code, content=payload)


def _log(request: Request, status_code: int, detail: str, exc: Exception) -> None:
    """Log a failure at a level matching its severity.

    5xx responses log the real exception text -- which the client never sees --
    so an operator can diagnose what the fixed message hides.
    """
    if status_code >= 500:
        logger.error(
            "%s %s -> %d: %s (%s: %s)",
            request.method,
            request.url.path,
            status_code,
            detail,
            type(exc).__name__,
            exc,
        )
    else:
        logger.info("%s %s -> %d: %s", request.method, request.url.path, status_code, detail)


async def api_error_handler(request: Request, exc: Exception) -> Response:
    """Render an :class:`APIError` raised by a route or the service layer."""
    error = exc if isinstance(exc, APIError) else APIError(GENERIC_SERVER_MESSAGE)
    _log(request, error.status_code, error.detail, exc)
    return _error_response(error.status_code, error.code, error.detail)


async def domain_error_handler(request: Request, exc: Exception) -> Response:
    """Translate a Phase 1-4 exception into the standard error body."""
    mapping = _mapping_for(exc)
    detail = mapping.message if mapping.message is not None else str(exc)
    _log(request, mapping.status_code, detail, exc)
    return _error_response(mapping.status_code, mapping.code, detail)


async def validation_error_handler(request: Request, exc: Exception) -> Response:
    """Render a Pydantic request-validation failure as ``422``.

    The per-field errors are reduced to ``loc``/``msg``/``type``: the ``input``
    Pydantic normally echoes back would repeat whatever the client sent, which
    for this API can be a whole job description.
    """
    raw = exc.errors() if isinstance(exc, RequestValidationError) else []
    errors = [
        {
            "field": ".".join(str(part) for part in item.get("loc", ())),
            "message": item.get("msg", ""),
            "type": item.get("type", ""),
        }
        for item in raw
    ]
    logger.info(
        "%s %s -> 422: %d validation error(s)", request.method, request.url.path, len(errors)
    )
    return _error_response(
        422,
        "validation_error",
        "Request validation failed.",
        {"errors": jsonable_encoder(errors)},
    )


async def http_exception_handler(request: Request, exc: Exception) -> Response:
    """Render Starlette's own errors -- unknown route, wrong method -- consistently."""
    status_code = getattr(exc, "status_code", 500)
    detail = str(getattr(exc, "detail", "") or GENERIC_SERVER_MESSAGE)
    code = _STATUS_CODES.get(status_code, "http_error")
    _log(request, status_code, detail, exc)
    return _error_response(status_code, code, detail)


async def unhandled_error_handler(request: Request, exc: Exception) -> Response:
    """Last resort: log the traceback, return a fixed message."""
    logger.exception(
        "Unhandled %s on %s %s", type(exc).__name__, request.method, request.url.path
    )
    return _error_response(500, "internal_error", GENERIC_SERVER_MESSAGE)


def install_error_handlers(app: FastAPI) -> None:
    """Register every error handler on ``app``.

    Args:
        app: The application to configure.
    """
    app.add_exception_handler(APIError, api_error_handler)
    for exception_type in DOMAIN_EXCEPTION_TYPES:
        app.add_exception_handler(exception_type, domain_error_handler)
    app.add_exception_handler(RequestValidationError, validation_error_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(Exception, unhandled_error_handler)
