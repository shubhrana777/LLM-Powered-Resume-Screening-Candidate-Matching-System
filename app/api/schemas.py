"""Pydantic request and response models.

These are the API's contract, and nothing more: they validate what arrives,
shape what leaves, and describe both in the generated OpenAPI document. No
business rule lives here -- skills, scores and recommendations are computed by
the Phase 1-4 modules and only rendered by these models.

Two notes travel with the data rather than only in the documentation, because
the numbers are easy to misread once they are out of context:

* ``similarity_score`` is cosine similarity, not a probability. See
  :data:`SCORE_NOTE`.
* ``recommendation`` is a coarse ordinal label, not a score. See
  :data:`RECOMMENDATION_NOTE`.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.models import Recommendation

__all__ = [
    "MAX_JOB_DESCRIPTION_CHARS",
    "MAX_TOP_K",
    "PREVIEW_CHARS",
    "SCORE_NOTE",
    "RECOMMENDATION_NOTE",
    "HealthResponse",
    "UploadResumeResponse",
    "MatchRequest",
    "CandidateMatch",
    "MatchResponse",
    "AnalyzeRequest",
    "EvidenceItem",
    "AnalysisResponse",
    "CandidateSummary",
    "UnreadableResume",
    "CandidateListResponse",
    "DeleteCandidatesRequest",
    "DeleteCandidatesResponse",
    "ErrorResponse",
    "ValidationErrorItem",
    "ValidationErrorResponse",
]

# A job description far longer than this is not a job description.
MAX_JOB_DESCRIPTION_CHARS = 20_000
MAX_TOP_K = 100
MAX_CANDIDATE_REF_CHARS = 200

# A delete batch larger than this is not a recruiter tidying up a pool.
MAX_DELETE_BATCH = 500

# How much extracted text POST /upload-resume echoes back, so a caller can see
# that extraction worked without the endpoint returning the whole resume.
PREVIEW_CHARS = 200

SCORE_NOTE = (
    "Cosine similarity between the job-description embedding and the resume "
    "embedding, in [-1.0, 1.0]. A semantic similarity score: not a probability "
    "of being hired, not a percentage of requirements met, and comparable only "
    "within a single ranking."
)

RECOMMENDATION_NOTE = (
    "A coarse ordinal label, not a score and not a probability. It describes "
    "how well the retrieved evidence supports the stated requirements, and is "
    "never a hiring decision."
)

# Rejected outright in a candidate reference: the pool is looked up by exact id
# or name and never by path, so these characters can only be an attempt at
# something else.
_UNSAFE_REFERENCE_CHARS = ("/", "\\", "\x00")


def _require_content(value: str, field_name: str) -> str:
    """Reject whitespace-only text.

    Args:
        value: The submitted string.
        field_name: Name used in the error message.

    Returns:
        ``value`` unchanged.

    Raises:
        ValueError: If the string holds no non-whitespace characters.
    """
    if not value.strip():
        raise ValueError(f"{field_name} must contain non-whitespace text")
    return value


class HealthResponse(BaseModel):
    """Liveness report. Cheap by design: it touches no model and no disk."""

    model_config = ConfigDict(json_schema_extra={"example": {"status": "healthy"}})

    status: Literal["healthy"] = Field(default="healthy", description="Always 'healthy'.")
    service: str = Field(description="Service name.")
    version: str = Field(description="Application version.")
    llm_provider: str = Field(
        description="Configured LLM provider name. Never a key, and never whether one is set."
    )


class UploadResumeResponse(BaseModel):
    """What was extracted from an uploaded PDF.

    The full resume text is never returned, only its size and a short preview.

    By default the upload is parsed in a temporary location and deleted, and
    does **not** join the candidate pool. Send ``store=true`` to keep it: the
    file is copied into the server's resume directory under a generated name and
    becomes rankable, and ``candidate_id`` names it.
    """

    filename: str = Field(description="The client-supplied file name, path stripped.")
    status: Literal["success"] = "success"
    text_length: int = Field(description="Characters of cleaned text extracted.")
    word_count: int = Field(description="Whitespace-separated tokens in the extracted text.")
    preview: str = Field(
        description=f"First {PREVIEW_CHARS} characters of the extracted text."
    )
    stored: bool = Field(
        default=False,
        description="Whether the resume joined the candidate pool. False unless store=true.",
    )
    candidate_id: str | None = Field(
        default=None,
        description="The id the resume was stored under; null when it was not stored.",
    )


class MatchRequest(BaseModel):
    """A job description to rank the configured resume pool against."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "job_description": "Financial analyst with 3+ years of experience. "
                "Strong Excel and SQL; Python for data analysis.",
                "top_k": 5,
            }
        }
    )

    job_description: str = Field(
        min_length=1,
        max_length=MAX_JOB_DESCRIPTION_CHARS,
        description="The job description text.",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=MAX_TOP_K,
        description="How many ranked candidates to return.",
    )

    @field_validator("job_description")
    @classmethod
    def _job_description_has_content(cls, value: str) -> str:
        return _require_content(value, "job_description")


class CandidateMatch(BaseModel):
    """One ranked candidate."""

    rank: int = Field(description="1-based position, where 1 is the closest match.")
    candidate: str = Field(description="Display name, falling back to the candidate id.")
    candidate_id: str = Field(description="Stable id, derived from the resume file name.")
    similarity_score: float = Field(description=SCORE_NOTE)


class MatchResponse(BaseModel):
    """A ranking of the configured resume pool."""

    results: list[CandidateMatch]
    count: int = Field(description="How many results were returned.")
    candidates_considered: int = Field(description="How many resumes were ranked.")
    score_type: Literal["cosine_similarity"] = "cosine_similarity"
    score_note: str = Field(default=SCORE_NOTE, description="What the score does and does not mean.")


class AnalyzeRequest(BaseModel):
    """A request to analyse one candidate from the pool against a job description.

    ``candidate`` identifies an existing candidate; it never carries candidate
    information. Skills, experience and evidence always come from the resume on
    disk, so a client cannot supply facts that bypass the grounding checks.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "candidate": "sarah_wilson",
                "job_description": "Financial analyst with 3+ years of experience. "
                "Strong Excel and SQL; Python for data analysis.",
            }
        }
    )

    candidate: str = Field(
        min_length=1,
        max_length=MAX_CANDIDATE_REF_CHARS,
        description="Candidate id, display name, or resume file name.",
    )
    job_description: str = Field(
        min_length=1,
        max_length=MAX_JOB_DESCRIPTION_CHARS,
        description="The job description text.",
    )

    @field_validator("job_description")
    @classmethod
    def _job_description_has_content(cls, value: str) -> str:
        return _require_content(value, "job_description")

    @field_validator("candidate")
    @classmethod
    def _candidate_is_a_reference_not_a_path(cls, value: str) -> str:
        _require_content(value, "candidate")
        if any(char in value for char in _UNSAFE_REFERENCE_CHARS):
            raise ValueError(
                "candidate must be an id or a name, not a path; "
                "remove any '/', '\\' or null characters"
            )
        return value


class EvidenceItem(BaseModel):
    """One verbatim resume passage that was put in front of the model.

    Source text only. The model's internal reasoning is neither requested nor
    returned. ``candidate_id`` is included so a client can verify for itself
    that every passage belongs to the candidate it asked about.
    """

    candidate_id: str
    chunk_id: str
    text: str
    retrieval_score: float = Field(
        description="Cosine similarity to the retrieval query, in [-1.0, 1.0]. "
        "A similarity score, not a confidence that the passage answers the query."
    )


class AnalysisResponse(BaseModel):
    """A validated, evidence-grounded analysis of one candidate.

    ``warnings`` lists claims the Phase 4 validator had to correct before the
    response was assembled, and ``is_grounded`` is false whenever it is
    non-empty. An analysis is meant to be checked against ``evidence``, not
    taken on trust.
    """

    candidate: str = Field(description="Display name, falling back to the candidate id.")
    candidate_id: str
    recommendation: Recommendation = Field(description=RECOMMENDATION_NOTE)
    recommendation_note: str = Field(default=RECOMMENDATION_NOTE)
    summary: str
    matched_skills: list[str] = Field(description="Required skills the evidence supports.")
    skill_gaps: list[str] = Field(description="Required skills the evidence does not support.")
    experience_assessment: str = Field(
        description="Prose comparison of stated experience against the requirement, "
        "or 'Not stated' when the resume gives none."
    )
    education: list[str] = Field(
        default_factory=list,
        description="Degrees found on the resume by the deterministic Phase 3 extractor. "
        "Extracted, never generated by the model.",
    )
    evidence: list[EvidenceItem]
    limitations: list[str] = Field(description="What this analysis could not determine.")
    warnings: list[str] = Field(description="Unsupported claims corrected during validation.")
    is_grounded: bool = Field(description="False when any claim had to be corrected.")
    model: str = Field(description="Provider/model identifier that produced the analysis.")


class CandidateSummary(BaseModel):
    """One candidate available in the configured resume directory."""

    candidate_id: str
    name: str
    filename: str
    text_length: int = Field(description="Characters of cleaned resume text.")


class UnreadableResume(BaseModel):
    """A file in the resume directory that could not be parsed."""

    filename: str
    reason: str = Field(description="Why it was skipped, with filesystem paths removed.")


class CandidateListResponse(BaseModel):
    """Everything the configured resume directory yielded."""

    candidates: list[CandidateSummary]
    count: int
    unreadable: list[UnreadableResume] = Field(
        default_factory=list,
        description="Files that were skipped, so a missing candidate is visible rather than silent.",
    )


class DeleteCandidatesRequest(BaseModel):
    """Which pooled candidates to remove.

    Each entry names a candidate the server already holds -- an id, a display
    name, or a resume file name. It is looked up in the pool, never treated as
    a path, so this cannot reach a file the pool does not contain.
    """

    model_config = ConfigDict(
        json_schema_extra={"example": {"candidates": ["sarah_wilson", "james_patel"]}}
    )

    candidates: list[str] = Field(
        min_length=1,
        max_length=MAX_DELETE_BATCH,
        description="Candidate ids, display names, or resume file names.",
    )

    @field_validator("candidates")
    @classmethod
    def _references_are_not_paths(cls, value: list[str]) -> list[str]:
        for item in value:
            _require_content(item, "candidates")
            if any(char in item for char in _UNSAFE_REFERENCE_CHARS):
                raise ValueError(
                    "a candidate must be an id or a name, not a path; "
                    "remove any '/', '\\' or null characters"
                )
        return value


class DeleteCandidatesResponse(BaseModel):
    """What a delete actually removed.

    One failure does not abandon the batch, so both lists can be non-empty:
    ``deleted`` names what went, ``failed`` says what stayed and why.
    """

    deleted: list[str] = Field(description="Ids of the candidates that were removed.")
    failed: list[str] = Field(
        default_factory=list,
        description="Entries that could not be removed, each with the reason.",
    )
    remaining: int = Field(description="How many candidates the pool still holds.")


class ErrorResponse(BaseModel):
    """The shape every error response takes."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {"detail": "Only PDF files are supported.", "code": "unsupported_file_type"}
        }
    )

    detail: str = Field(description="Human-readable message. Never a traceback or a path.")
    code: str = Field(description="Stable machine-readable error code.")


class ValidationErrorItem(BaseModel):
    """One field that failed validation."""

    field: str
    message: str
    type: str


class ValidationErrorResponse(ErrorResponse):
    """A ``422`` with per-field detail."""

    errors: list[ValidationErrorItem] = Field(default_factory=list)
