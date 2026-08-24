"""HTTP endpoints.

Every route here does the same three things and nothing else: validate the
request (mostly via Pydantic), call one method on
:class:`~app.api.service.ScreeningService`, and render the result as a response
model. There is no PDF parsing, no similarity computation, no skill extraction,
no retrieval and no prompt in this module -- those live in the Phase 1-4
packages and are called, not copied.

Route functions for matching and analysis are declared ``def`` rather than
``async def`` on purpose: they call into blocking, CPU-bound code (embedding,
FAISS, the model), and FastAPI runs a synchronous route in a worker thread
instead of blocking the event loop.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, UploadFile
from starlette.concurrency import run_in_threadpool

from app import __version__
from app.api.dependencies import ServiceDep, SettingsDep
from app.api.errors import BadRequestError, PayloadTooLargeError
from app.api.schemas import (
    PREVIEW_CHARS,
    AnalysisResponse,
    AnalyzeRequest,
    CandidateListResponse,
    CandidateMatch,
    CandidateSummary,
    DeleteCandidatesRequest,
    DeleteCandidatesResponse,
    ErrorResponse,
    EvidenceItem,
    HealthResponse,
    MatchRequest,
    MatchResponse,
    UnreadableResume,
    UploadResumeResponse,
    ValidationErrorResponse,
)
from app.api.service import scrub_path
from app.models import CandidateAnalysis, MatchResult
from app.resume_parser import PDF_SUFFIX, extract_text_from_pdf

__all__ = ["router", "SERVICE_NAME"]

logger = logging.getLogger(__name__)

router = APIRouter()

SERVICE_NAME = "resume-screening-api"

# Uploads are streamed to disk in pieces so a large file never has to fit in
# memory before it can be rejected.
_UPLOAD_CHUNK_BYTES = 64 * 1024

# Every PDF starts with this. Checking it rejects a mislabelled file before
# anything is written, and gives a clearer error than PyMuPDF would.
_PDF_MAGIC = b"%PDF"

_ERROR_RESPONSES: dict[int | str, dict[str, object]] = {
    400: {"model": ErrorResponse, "description": "The request or the file is invalid."},
    404: {"model": ErrorResponse, "description": "The candidate or resource does not exist."},
    422: {"model": ValidationErrorResponse, "description": "Request validation failed."},
    500: {"model": ErrorResponse, "description": "Unexpected server-side error."},
}

_LLM_ERROR_RESPONSES: dict[int | str, dict[str, object]] = {
    **_ERROR_RESPONSES,
    502: {"model": ErrorResponse, "description": "The language model provider failed."},
}


# ---------------------------------------------------------------------------
# Rendering helpers -- dataclass in, response model out. No logic.
# ---------------------------------------------------------------------------


def _render_match(result: MatchResult) -> CandidateMatch:
    """Render one Phase 2 result.

    The score is rounded for readability only. It stays the raw cosine
    similarity FAISS returned; nothing is rescaled into a percentage.
    """
    return CandidateMatch(
        rank=result.rank,
        candidate=result.display_name,
        candidate_id=result.candidate_id,
        similarity_score=round(result.similarity_score, 4),
    )


def _render_analysis(analysis: CandidateAnalysis) -> AnalysisResponse:
    """Render one Phase 4 analysis, evidence included."""
    return AnalysisResponse(
        candidate=analysis.display_name,
        candidate_id=analysis.candidate_id,
        recommendation=analysis.recommendation,
        summary=analysis.summary,
        matched_skills=list(analysis.matched_skills),
        skill_gaps=list(analysis.skill_gaps),
        experience_assessment=analysis.experience_assessment,
        education=list(analysis.education),
        evidence=[
            EvidenceItem(
                candidate_id=item.candidate_id,
                chunk_id=item.chunk_id,
                text=item.text,
                retrieval_score=round(item.retrieval_score, 4),
            )
            for item in analysis.evidence
        ],
        limitations=list(analysis.limitations),
        warnings=list(analysis.warnings),
        is_grounded=analysis.is_grounded,
        model=analysis.model_name,
    )


# ---------------------------------------------------------------------------
# Upload handling
# ---------------------------------------------------------------------------


async def _stream_to_temp_file(upload: UploadFile, max_bytes: int) -> Path:
    """Write an upload to a temporary file, enforcing type and size as it goes.

    The temporary file is created by :mod:`tempfile` with a generated name. The
    client-supplied file name is never used to build a path, so a name like
    ``../../etc/passwd`` cannot escape anywhere.

    Args:
        upload: The incoming file.
        max_bytes: Size ceiling; the upload is abandoned as soon as it is passed.

    Returns:
        Path to the temporary file. The caller is responsible for deleting it.

    Raises:
        BadRequestError: If the file is empty or does not begin with ``%PDF``.
        PayloadTooLargeError: If it exceeds ``max_bytes``.
    """
    handle = tempfile.NamedTemporaryFile(suffix=PDF_SUFFIX, delete=False)
    temp_path = Path(handle.name)
    total = 0

    try:
        with handle:
            while chunk := await upload.read(_UPLOAD_CHUNK_BYTES):
                if total == 0 and not chunk.startswith(_PDF_MAGIC):
                    raise BadRequestError(
                        "The uploaded file is not a PDF: it does not start with %PDF.",
                        code="unsupported_file_type",
                    )

                total += len(chunk)
                if total > max_bytes:
                    raise PayloadTooLargeError(
                        f"The uploaded file exceeds the {max_bytes} byte limit."
                    )
                handle.write(chunk)

        if total == 0:
            raise BadRequestError("The uploaded file is empty.", code="empty_file")

        return temp_path
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["system"],
    summary="Liveness check",
    description=(
        "Reports that the service is up. Deliberately cheap: it loads no model, "
        "reads no resume, and never reveals whether an API key is configured."
    ),
)
def health(settings: SettingsDep) -> HealthResponse:
    """Return service liveness and identifying information."""
    return HealthResponse(
        service=SERVICE_NAME,
        version=__version__,
        llm_provider=settings.llm_provider,
    )


@router.post(
    "/upload-resume",
    response_model=UploadResumeResponse,
    tags=["resumes"],
    summary="Upload a PDF resume and extract its text",
    description=(
        "Validates and parses a PDF with the Phase 1 parser and reports what was "
        "extracted. The full text is not returned, only its length and a short "
        "preview. "
        "By default the file is written to a temporary location, parsed, and "
        "deleted: it is **not** stored and does not join the candidate pool. Send "
        "`store=true` to keep it -- the resume is copied into the server's resume "
        "directory under a generated name derived from the submitted one, and "
        "becomes rankable by `/match-candidates` and `/analyze-candidate`. "
        "Re-uploading a file whose name reduces to the same id replaces it."
    ),
    responses={**_ERROR_RESPONSES, 413: {"model": ErrorResponse, "description": "File too large."}},
)
async def upload_resume(
    settings: SettingsDep,
    service: ServiceDep,
    file: UploadFile = File(..., description="The PDF resume to upload."),
    store: bool = Form(
        False,
        description="Keep the resume as a rankable candidate instead of discarding it.",
    ),
) -> UploadResumeResponse:
    """Extract text from an uploaded PDF resume, optionally keeping it.

    Args:
        settings: API configuration, for the upload size ceiling.
        service: The shared screening service, used only when storing.
        file: The uploaded PDF.
        store: Whether the resume should join the candidate pool.

    Returns:
        Metadata about the extracted text, and the candidate id when stored.

    Raises:
        BadRequestError: If the file is missing a name, is not a PDF, or is empty.
        PayloadTooLargeError: If it exceeds the configured ceiling.
        app.resume_parser.ResumeParserError: If the PDF cannot be read. Mapped
            to a 400 by the registered error handlers.
    """
    # Keep only the file name: a client-supplied path component is never used.
    filename = Path(file.filename or "").name
    if not filename:
        raise BadRequestError("The upload has no file name.", code="missing_filename")

    if not filename.lower().endswith(PDF_SUFFIX):
        raise BadRequestError(
            f"Only {PDF_SUFFIX} files are supported; got {filename!r}.",
            code="unsupported_file_type",
        )

    logger.info("Upload received: %s (%s)", filename, file.content_type or "no content type")

    temp_path = await _stream_to_temp_file(file, settings.max_upload_bytes)
    candidate_id: str | None = None
    try:
        text = await run_in_threadpool(extract_text_from_pdf, temp_path)
        # Only a resume that parsed is worth keeping, so storing happens here
        # rather than while streaming.
        if store:
            candidate = await run_in_threadpool(service.store_resume, temp_path, filename)
            candidate_id = candidate.candidate_id
    except Exception as exc:
        logger.info("Upload %s could not be processed: %s", filename, type(exc).__name__)
        raise
    finally:
        temp_path.unlink(missing_ok=True)

    logger.info(
        "Upload %s parsed: %d characters extracted, stored=%s",
        filename,
        len(text),
        bool(candidate_id),
    )

    return UploadResumeResponse(
        filename=filename,
        text_length=len(text),
        word_count=len(text.split()),
        preview=text[:PREVIEW_CHARS],
        stored=candidate_id is not None,
        candidate_id=candidate_id,
    )


@router.post(
    "/match-candidates",
    response_model=MatchResponse,
    tags=["matching"],
    summary="Rank the resume pool against a job description",
    description=(
        "Embeds the job description and ranks the server's configured resumes by "
        "cosine similarity, using the Phase 2 matching engine. Resumes are parsed "
        "and embedded once and reused across requests. The score is a semantic "
        "similarity, not a probability."
    ),
    responses=_ERROR_RESPONSES,
)
def match_candidates(request: MatchRequest, service: ServiceDep) -> MatchResponse:
    """Rank the configured resume pool against a job description.

    Args:
        request: The job description and how many results to return.
        service: The shared screening service.

    Returns:
        Ranked candidates, best match first.

    Raises:
        app.api.errors.NotFoundError: If the resume directory is missing or empty.
    """
    logger.info(
        "Match request: top_k=%d, job description %d characters",
        request.top_k,
        len(request.job_description),
    )

    results, considered = service.match(request.job_description, request.top_k)

    logger.info("Match request returned %d of %d candidate(s)", len(results), considered)

    return MatchResponse(
        results=[_render_match(result) for result in results],
        count=len(results),
        candidates_considered=considered,
    )


@router.post(
    "/analyze-candidate",
    response_model=AnalysisResponse,
    tags=["analysis"],
    summary="Analyse one candidate against a job description",
    description=(
        "Runs the Phase 3 profile and the Phase 4 retrieval-augmented analysis for "
        "one candidate from the configured resume directory. The candidate's skills, "
        "experience and evidence are read from their resume; the request supplies "
        "only which candidate and which job. Retrieval is scoped to that candidate "
        "alone, and the model's response is validated against the deterministic "
        "profile before it is returned, so unsupported claims are corrected and "
        "reported in `warnings`."
    ),
    responses=_LLM_ERROR_RESPONSES,
)
def analyze_candidate(request: AnalyzeRequest, service: ServiceDep) -> AnalysisResponse:
    """Analyse one pooled candidate against a job description.

    Args:
        request: Which candidate, and the job description.
        service: The shared screening service.

    Returns:
        The validated analysis with the evidence it rests on.

    Raises:
        app.api.errors.NotFoundError: If the candidate is not in the pool.
        app.llm.LLMError: If the provider fails. Mapped to a 5xx by the handlers.
    """
    logger.info(
        "Analysis request for candidate %r, job description %d characters",
        request.candidate,
        len(request.job_description),
    )

    analysis = service.analyze(request.candidate, request.job_description)

    logger.info(
        "Analysis complete for %s: %s, grounded=%s, %d evidence passage(s), %d warning(s)",
        analysis.candidate_id,
        analysis.recommendation.value,
        analysis.is_grounded,
        len(analysis.evidence),
        len(analysis.warnings),
    )

    return _render_analysis(analysis)


@router.get(
    "/candidates",
    response_model=CandidateListResponse,
    tags=["resumes"],
    summary="List the candidates available on the server",
    description=(
        "Lists the resumes in the server's configured directory, parsed with the "
        "Phase 1 parser. Files that could not be read are reported under "
        "`unreadable` rather than silently omitted. An existing but empty "
        "directory returns an empty list; a missing directory is a 404."
    ),
    responses=_ERROR_RESPONSES,
)
def list_candidates(service: ServiceDep) -> CandidateListResponse:
    """List the candidates loaded from the configured resume directory.

    Args:
        service: The shared screening service.

    Returns:
        The available candidates and any files that could not be parsed.

    Raises:
        app.api.errors.NotFoundError: If the configured directory does not exist.
    """
    pool = service.load_pool()

    logger.info(
        "Candidate listing: %d available, %d unreadable",
        len(pool.candidates),
        len(pool.failures),
    )

    return CandidateListResponse(
        candidates=[
            CandidateSummary(
                candidate_id=candidate.candidate_id,
                name=candidate.display_name,
                filename=(
                    candidate.source_path.name
                    if candidate.source_path is not None
                    else f"{candidate.candidate_id}{PDF_SUFFIX}"
                ),
                text_length=len(candidate.resume_text),
            )
            for candidate in pool.candidates
        ],
        count=len(pool.candidates),
        unreadable=[
            UnreadableResume(
                filename=failure.path.name,
                reason=scrub_path(failure.reason, failure.path),
            )
            for failure in pool.failures
        ],
    )


@router.delete(
    "/candidates/{candidate_id}",
    response_model=DeleteCandidatesResponse,
    tags=["resumes"],
    summary="Remove one candidate from the pool",
    description=(
        "Deletes one resume from the server's configured resume directory and "
        "refreshes the pool, so the candidate stops appearing in rankings and can "
        "no longer be analysed.\n\n"
        "The path parameter names a candidate the server already holds; it is "
        "looked up in the pool and never used to build a path."
    ),
    responses=_ERROR_RESPONSES,
)
def delete_candidate(candidate_id: str, service: ServiceDep) -> DeleteCandidatesResponse:
    """Remove one pooled candidate.

    Args:
        candidate_id: Candidate id, display name, or resume file name.
        service: The shared screening service.

    Returns:
        What was removed and how many candidates remain.

    Raises:
        app.api.errors.NotFoundError: If the candidate is not in the pool.
    """
    logger.info("Delete request for candidate %r", candidate_id)

    removed = service.delete_candidate(candidate_id)
    remaining = len(service.load_pool().candidates)

    logger.info("Deleted %s; %d candidate(s) remain", removed, remaining)

    return DeleteCandidatesResponse(deleted=[removed], remaining=remaining)


@router.post(
    "/candidates/delete",
    response_model=DeleteCandidatesResponse,
    tags=["resumes"],
    summary="Remove several candidates from the pool",
    description=(
        "Deletes each named candidate and refreshes the pool. A candidate that "
        "cannot be removed is reported in `failed` and does not stop the rest of "
        "the batch.\n\n"
        "This is a POST rather than a DELETE because it carries a body, which "
        "many HTTP clients and proxies will not send on a DELETE."
    ),
    responses=_ERROR_RESPONSES,
)
def delete_candidates(
    request: DeleteCandidatesRequest, service: ServiceDep
) -> DeleteCandidatesResponse:
    """Remove several pooled candidates.

    Args:
        request: Which candidates to remove.
        service: The shared screening service.

    Returns:
        What was removed, what was not, and how many remain.
    """
    logger.info("Delete request for %d candidate(s)", len(request.candidates))

    deleted, failures = service.delete_candidates(request.candidates)
    remaining = len(service.load_pool().candidates)

    logger.info(
        "Deleted %d candidate(s), %d failed; %d remain",
        len(deleted),
        len(failures),
        remaining,
    )

    return DeleteCandidatesResponse(
        deleted=deleted,
        failed=[f"{reference}: {reason}" for reference, reason in failures],
        remaining=remaining,
    )


@router.delete(
    "/candidates",
    response_model=DeleteCandidatesResponse,
    tags=["resumes"],
    summary="Remove every candidate from the pool",
    description=(
        "Empties the candidate pool: every pooled resume is deleted and the "
        "indexes are rebuilt from nothing.\n\n"
        "Only files that are currently pooled candidates are touched -- anything "
        "else in the directory is left alone. Clearing an already-empty pool is "
        "not an error."
    ),
    responses=_ERROR_RESPONSES,
)
def clear_candidates(service: ServiceDep) -> DeleteCandidatesResponse:
    """Remove every pooled candidate.

    Args:
        service: The shared screening service.

    Returns:
        Everything that was removed, and a remaining count of zero.
    """
    logger.info("Clear request for the whole candidate pool")

    removed = service.clear_candidates()
    remaining = len(service.load_pool().candidates)

    logger.info("Cleared %d candidate(s); %d remain", len(removed), remaining)

    return DeleteCandidatesResponse(deleted=removed, remaining=remaining)
