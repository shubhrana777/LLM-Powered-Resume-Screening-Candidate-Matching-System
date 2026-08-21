"""Lightweight typed records shared across the matching pipeline.

Plain frozen dataclasses are enough here. There is no validation framework and
no ORM: a candidate is text plus an identifier, and a match result is that
identifier plus a score.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "InvalidCandidateError",
    "Candidate",
    "MatchResult",
]


class InvalidCandidateError(ValueError):
    """A candidate record is missing an id or has no usable resume text."""


@dataclass(frozen=True, slots=True)
class Candidate:
    """One candidate to be ranked against a job description.

    Attributes:
        candidate_id: Stable identifier, unique within a matching run.
        resume_text: Cleaned resume text, typically from
            :func:`app.resume_parser.extract_text_from_pdf`.
        candidate_name: Human-readable name, when known.
        source_path: Path the resume was read from, when it came from a file.

    Raises:
        InvalidCandidateError: If the id or the resume text is empty.
    """

    candidate_id: str
    resume_text: str
    candidate_name: str | None = None
    source_path: Path | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_id, str) or not self.candidate_id.strip():
            raise InvalidCandidateError("candidate_id must be a non-empty string")

        if not isinstance(self.resume_text, str) or not self.resume_text.strip():
            raise InvalidCandidateError(
                f"resume_text for candidate {self.candidate_id!r} is empty or whitespace only"
            )

    @property
    def display_name(self) -> str:
        """The candidate name if known, otherwise the candidate id."""
        return self.candidate_name or self.candidate_id


@dataclass(frozen=True, slots=True)
class MatchResult:
    """One ranked candidate returned by the matching engine.

    Attributes:
        candidate_id: Identifier of the matched candidate.
        candidate_name: Candidate name, when known.
        similarity_score: Cosine similarity between the job-description
            embedding and the resume embedding, in ``[-1.0, 1.0]``. Higher means
            the two texts are closer in the embedding space. This is a
            **semantic similarity score**, not a probability of being hired and
            not a calibrated quality measure. See the module docstring of
            :mod:`app.matching` for the full definition.
        rank: 1-based position in the ranking, where 1 is the closest match.
        source_path: Path the resume was read from, when known.
    """

    candidate_id: str
    candidate_name: str | None
    similarity_score: float
    rank: int
    source_path: Path | None = None

    @property
    def display_name(self) -> str:
        """The candidate name if known, otherwise the candidate id."""
        return self.candidate_name or self.candidate_id
