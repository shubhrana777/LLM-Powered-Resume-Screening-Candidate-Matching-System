"""The records every layer of the system passes around.

Plain frozen dataclasses. There is no validation framework and no ORM: a
candidate is text plus an identifier, and a match result is that identifier
plus a score.

What each record is
-------------------
====================== =====================================================
:class:`Candidate`     One person's resume: id, text, name, source file.
:class:`JobRequirements`  A job description, plus the skills and minimum
                       experience read out of it.
:class:`MatchResult`   One candidate's position in a ranking (Phase 2).
:class:`SkillComparison`  Required skills split into matched / missing /
                       additional (Phase 3).
:class:`CandidateProfile` Everything the deterministic extractors found about
                       one candidate, optionally joined to their ranking.
:class:`CandidateAnalysis` The LLM's reading of one candidate, after it has
                       been validated against their profile (Phase 4).
:class:`EducationEntry` One degree found on a resume.
:class:`Recommendation` The controlled vocabulary an analysis may conclude with.
====================== =====================================================

The four measures, and why they are not interchangeable
-------------------------------------------------------
A candidate ends up carrying four numbers or labels. They come from different
places, mean different things, and regularly disagree with each other. Treating
any one as a stand-in for another is the most damaging mistake this codebase
could make, so each is named separately and never derived from the others.

``semantic_match_score`` -- **similarity**
    Cosine similarity between the job-description embedding and the whole
    resume embedding. *Raw model output*: an embedding-space distance, computed
    by :mod:`app.matching`. It reflects how alike two documents read, which is
    a useful way to order a pile of resumes and nothing more. It is not a
    probability, not a percentage of requirements met, and comparable only
    within a single ranking. The UI renders it as a percentage of the cosine
    scale for readability; the stored value is untouched.

``matched_skills`` / ``missing_skills`` -- **skill coverage**
    A count, not a score: how many of the skills named in the job description
    appear on the resume. *Derived deterministically* by
    :mod:`app.skill_extractor` from a fixed taxonomy. Exact, explainable, and
    blind to any skill the taxonomy does not know. A candidate can read as
    highly similar and still cover few required skills; that disagreement is
    information, not an error.

``meets_experience_requirement`` -- **experience**
    A three-way comparison: ``True``, ``False``, or ``None`` when either the
    resume or the job description does not state a figure. *Derived
    deterministically* by :mod:`app.candidate_analyzer`. Unknown is never
    resolved into a pass or a fail.

``recommendation`` -- **assessment**
    A coarse ordinal label from :class:`Recommendation`. *LLM-generated*, then
    checked against the deterministic profile by :mod:`app.analysis_parser` and
    replaced with ``INSUFFICIENT_INFORMATION`` if it cannot be trusted. It is
    not a score, not a ranking key, and never a hiring decision.

Provenance, in one line each: **similarity** is measured, **coverage** and
**experience** are extracted, **recommendation** is generated and then
validated. Only the last one can be wrong in an interesting way, which is why
:class:`CandidateAnalysis` carries the evidence it was based on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

__all__ = [
    "InvalidCandidateError",
    "Candidate",
    "MatchResult",
    "SkillComparison",
    "EducationEntry",
    "JobRequirements",
    "CandidateProfile",
    "NOT_STATED",
    "Recommendation",
    "CandidateAnalysis",
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
        matched_skills: **Skill coverage** -- required skills the candidate
            has. Extracted deterministically, so this is a count of named
            skills rather than a score.
        missing_skills: Required skills the candidate lacks.
        additional_skills: Candidate skills the job did not ask for.
        semantic_match_score: **Similarity** -- cosine similarity from Phase 2
            matching, or ``None`` when the profile was built without semantic
            matching. Raw model output: an embedding-space distance, not a
            probability of being hired and not a share of requirements met.
            Unrelated to ``matched_skills``; see the module docstring.
        rank: 1-based rank from semantic matching, when available.
        required_experience: Minimum experience the job stated, or ``None``.
        meets_experience_requirement: **Experience** -- ``True``/``False`` when
            both the requirement and the candidate's experience are known,
            otherwise ``None``. Derived deterministically, and unknown is never
            treated as pass or fail.
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


# The exact wording the LLM must use when a fact is absent from the evidence.
# Kept as a constant so the prompt, the parser and the tests cannot drift apart.
NOT_STATED = "Not stated"


class Recommendation(str, Enum):
    """Controlled vocabulary for an LLM hiring-review recommendation.

    A coarse, ordinal label -- deliberately not a score and not a probability.
    It expresses how well the retrieved evidence supports the stated
    requirements, nothing more, and never constitutes a hiring decision.

    ``INSUFFICIENT_INFORMATION`` is the correct answer whenever the evidence
    does not support any judgement, and is the safe fallback when an analysis
    cannot be trusted.
    """

    STRONG_MATCH = "STRONG_MATCH"
    GOOD_MATCH = "GOOD_MATCH"
    PARTIAL_MATCH = "PARTIAL_MATCH"
    WEAK_MATCH = "WEAK_MATCH"
    INSUFFICIENT_INFORMATION = "INSUFFICIENT_INFORMATION"

    @classmethod
    def values(cls) -> tuple[str, ...]:
        """Every permitted value, in descending order of strength."""
        return tuple(member.value for member in cls)

    @classmethod
    def parse(cls, value: object) -> Recommendation | None:
        """Coerce ``value`` to a member, or ``None`` if it is not one.

        Args:
            value: Candidate value, typically a string from LLM output.

        Returns:
            The matching member, or ``None``. Comparison ignores case and
            surrounding whitespace; anything else is rejected rather than
            guessed at.
        """
        if isinstance(value, cls):
            return value
        if not isinstance(value, str):
            return None
        try:
            return cls(value.strip().upper().replace(" ", "_").replace("-", "_"))
        except ValueError:
            return None


@dataclass(frozen=True, slots=True)
class CandidateAnalysis:
    """An LLM analysis of one candidate, grounded in retrieved resume evidence.

    Every field is either supported by the supplied evidence and profile, or
    reports absence explicitly. Nothing here is inferred from outside the
    material the model was given.

    Attributes:
        candidate_id: Candidate this analysis is about.
        candidate_name: Candidate name, when known.
        summary: **LLM-generated** prose summary of the candidate against the
            role. The only free text here, and the only field that can be
            subtly wrong in a way no check catches -- read ``evidence``.
        recommendation: **Assessment** -- LLM-generated controlled-vocabulary
            label, validated before it reaches this record. A coarse ordinal
            label, never a score and never a hiring decision. See
            :class:`Recommendation`.
        matched_skills: Required skills the evidence supports.
        skill_gaps: Required skills not supported by the evidence.
        experience_assessment: Prose comparison of stated experience against the
            stated requirement, or ``"Not stated"`` when the resume gives none.
        evidence: **Source text** -- the verbatim resume passages supplied to
            the model, retained so a reviewer can check any claim against what
            the resume actually says. Excerpts only, never the model's internal
            reasoning, and always scoped to this candidate alone.
        limitations: What this analysis could not determine, in the model's own
            words plus any caveats added by the parser.
        education: Degrees found on the resume by the Phase 3 extractor, as
            display strings. Deterministic -- extracted, never generated, and
            never subject to the model's claims.
        model_name: Identifier of the provider/model that produced it.
        warnings: Grounding problems detected while validating the response,
            such as a claimed skill absent from the candidate profile. A
            non-empty list means the raw output was corrected before use.
    """

    candidate_id: str
    candidate_name: str | None = None
    summary: str = NOT_STATED
    recommendation: Recommendation = Recommendation.INSUFFICIENT_INFORMATION
    matched_skills: tuple[str, ...] = ()
    skill_gaps: tuple[str, ...] = ()
    experience_assessment: str = NOT_STATED
    evidence: tuple[object, ...] = ()
    limitations: tuple[str, ...] = ()
    model_name: str = "unknown"
    warnings: tuple[str, ...] = field(default_factory=tuple)
    education: tuple[str, ...] = ()

    @property
    def display_name(self) -> str:
        """The candidate name if known, otherwise the candidate id."""
        return self.candidate_name or self.candidate_id

    @property
    def is_grounded(self) -> bool:
        """Whether validation found no unsupported claims in the raw output."""
        return not self.warnings
