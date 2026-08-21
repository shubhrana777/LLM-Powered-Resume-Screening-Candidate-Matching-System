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
    "SkillComparison",
    "EducationEntry",
    "JobRequirements",
    "CandidateProfile",
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


@dataclass(frozen=True, slots=True)
class SkillComparison:
    """The result of comparing required skills against candidate skills.

    Attributes:
        matched: Required skills the candidate has, in required-skill order.
        missing: Required skills the candidate lacks, in required-skill order.
        additional: Candidate skills the job did not ask for, in candidate order.
    """

    matched: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()
    additional: tuple[str, ...] = ()

    @property
    def required_count(self) -> int:
        """How many skills the job required."""
        return len(self.matched) + len(self.missing)

    @property
    def coverage(self) -> float | None:
        """Fraction of required skills the candidate has, in ``[0.0, 1.0]``.

        A plain count-based ratio -- ``len(matched) / required_count`` -- and
        nothing more. It is **not** a probability, a confidence, or a weighted
        score: every skill counts the same regardless of importance, and the
        figure only reflects skills present in the taxonomy.

        Returns:
            The ratio, or ``None`` when the job listed no recognised skills, in
            which case there is nothing to take a fraction of.
        """
        if self.required_count == 0:
            return None
        return len(self.matched) / self.required_count


@dataclass(frozen=True, slots=True)
class EducationEntry:
    """One education credential found in a resume.

    Attributes:
        degree: Normalized degree name, e.g. ``"Bachelor of Science"`` or ``"MBA"``.
        field: Field of study when one was stated nearby, e.g. ``"Finance"``.
        raw_text: The source line the entry came from, kept so a reviewer can
            check the extraction against the original resume.
    """

    degree: str
    field: str | None = None
    raw_text: str = ""

    def __str__(self) -> str:
        return f"{self.degree} - {self.field}" if self.field else self.degree


@dataclass(frozen=True, slots=True)
class JobRequirements:
    """Structured view of a job description.

    Only what the text states explicitly is recorded; nothing is inferred about
    requirements the description does not mention.

    Attributes:
        required_skills: Recognised skills named in the description, in
            taxonomy order.
        minimum_experience: Smallest explicitly stated number of years of
            experience, or ``None`` when the description does not state one.
        raw_text: The original job-description text.
    """

    required_skills: tuple[str, ...] = ()
    minimum_experience: float | None = None
    raw_text: str = ""


@dataclass(frozen=True, slots=True)
class CandidateProfile:
    """Structured analysis of one candidate against one job description.

    Attributes:
        candidate_id: Identifier of the candidate.
        candidate_name: Candidate name, when known.
        skills: Every recognised skill found on the resume, in taxonomy order.
        years_experience: Years of experience where explicitly stated on the
            resume, otherwise ``None``. Never inferred from dates or graduation
            years.
        education: Education entries found, possibly empty.
        matched_skills: Required skills the candidate has.
        missing_skills: Required skills the candidate lacks.
        additional_skills: Candidate skills the job did not ask for.
        semantic_match_score: Cosine similarity from Phase 2 matching, or
            ``None`` when the profile was built without semantic matching. A
            **semantic similarity score**, not a probability of being hired.
        rank: 1-based rank from semantic matching, when available.
        required_experience: Minimum experience the job stated, or ``None``.
        meets_experience_requirement: ``True``/``False`` when both the
            requirement and the candidate's experience are known, otherwise
            ``None``. Unknown is never treated as pass or fail.
        source_path: Path the resume was read from, when known.
    """

    candidate_id: str
    candidate_name: str | None = None
    skills: tuple[str, ...] = ()
    years_experience: float | None = None
    education: tuple[EducationEntry, ...] = ()
    matched_skills: tuple[str, ...] = ()
    missing_skills: tuple[str, ...] = ()
    additional_skills: tuple[str, ...] = ()
    semantic_match_score: float | None = None
    rank: int | None = None
    required_experience: float | None = None
    meets_experience_requirement: bool | None = None
    source_path: Path | None = None

    @property
    def display_name(self) -> str:
        """The candidate name if known, otherwise the candidate id."""
        return self.candidate_name or self.candidate_id

    @property
    def skill_comparison(self) -> SkillComparison:
        """The skill match rebuilt as a :class:`SkillComparison`."""
        return SkillComparison(
            matched=self.matched_skills,
            missing=self.missing_skills,
            additional=self.additional_skills,
        )
