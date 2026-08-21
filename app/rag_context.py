"""Assemble the grounded context handed to the LLM.

Three clearly separated sections, so the model can tell what kind of claim each
piece of information supports:

* **JOB DESCRIPTION** -- what the role asks for.
* **CANDIDATE PROFILE** -- the Phase 3 deterministic extraction. Reliable and
  checkable, but shallow.
* **RETRIEVED RESUME EVIDENCE** -- verbatim resume passages selected by
  retrieval. This is what the model may quote and reason over.

The rendered profile lines use a fixed format on purpose: the parser validates
LLM claims against them, and the offline fake provider reads them, so drift in
this layout would silently weaken both.

Candidate isolation is enforced here as well as in the retriever. Building a
context whose evidence belongs to a different candidate raises rather than
quietly producing a prompt that would make the LLM describe the wrong person.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from app.models import NOT_STATED, CandidateProfile, JobRequirements
from app.retriever import RetrievedEvidence

__all__ = [
    "ContextIsolationError",
    "RagContext",
    "build_rag_context",
]

MAX_EVIDENCE_CHARS = 1200


class ContextIsolationError(Exception):
    """Evidence from one candidate was about to be used for another."""


@dataclass(frozen=True, slots=True)
class RagContext:
    """The material supplied to the LLM for one candidate.

    Attributes:
        candidate_id: Candidate this context describes.
        profile: The Phase 3 structured profile.
        requirements: The structured job requirements.
        evidence: Retrieved resume passages, best match first.
        text: The rendered context string sent to the model.
    """

    candidate_id: str
    profile: CandidateProfile
    requirements: JobRequirements
    evidence: tuple[RetrievedEvidence, ...]
    text: str

    @property
    def evidence_chunk_ids(self) -> frozenset[str]:
        """Chunk ids the model was shown, used to validate cited evidence."""
        return frozenset(item.chunk_id for item in self.evidence)


def _format_skills(skills: Sequence[str]) -> str:
    """Render a skill list, or a clear marker when it is empty."""
    return ", ".join(skills) if skills else "none identified"


def _render_profile(profile: CandidateProfile) -> str:
    """Render the deterministic Phase 3 profile as fixed-format lines.

    Args:
        profile: The candidate profile.

    Returns:
        The CANDIDATE PROFILE section body.
    """
    experience = (
        NOT_STATED
        if profile.years_experience is None
        else f"{profile.years_experience:g} years (stated on resume)"
    )
    required = (
        NOT_STATED
        if profile.required_experience is None
        else f"{profile.required_experience:g} years"
    )
    verdict = {
        True: "yes",
        False: "no",
        None: "unknown - not enough information to say",
    }[profile.meets_experience_requirement]

    education = (
        "; ".join(str(entry) for entry in profile.education)
        if profile.education
        else NOT_STATED
    )

    coverage = profile.skill_comparison.coverage
    coverage_line = (
        "Skill coverage: not applicable (job listed no recognised skills)"
        if coverage is None
        else (
            f"Skill coverage: {len(profile.matched_skills)}"
            f"/{profile.skill_comparison.required_count} required skills"
        )
    )

    return "\n".join(
        [
            f"Candidate: {profile.display_name}",
            f"Skills found on resume: {_format_skills(profile.skills)}",
            f"Matched skills: {_format_skills(profile.matched_skills)}",
            f"Missing skills: {_format_skills(profile.missing_skills)}",
            coverage_line,
            f"Experience stated on resume: {experience}",
            f"Experience required by job: {required}",
            f"Meets stated experience requirement: {verdict}",
            f"Education: {education}",
        ]
    )


def _render_evidence(evidence: Sequence[RetrievedEvidence]) -> str:
    """Render retrieved passages as numbered, attributed blocks."""
    if not evidence:
        return (
            "No resume passages were retrieved for this candidate. "
            "Treat every candidate-specific fact as not stated."
        )

    blocks = []
    for position, item in enumerate(evidence, start=1):
        text = item.text.strip()
        if len(text) > MAX_EVIDENCE_CHARS:
            text = text[:MAX_EVIDENCE_CHARS].rstrip() + " ..."

        blocks.append(
            f"[Chunk {position}] (chunk_id={item.chunk_id}, "
            f"similarity={item.retrieval_score:.4f})\n{text}"
        )

    return "\n\n".join(blocks)


def build_rag_context(
    profile: CandidateProfile,
    requirements: JobRequirements,
    evidence: Sequence[RetrievedEvidence] = (),
) -> RagContext:
    """Assemble the LLM context for one candidate.

    Args:
        profile: Phase 3 structured profile for the candidate.
        requirements: Structured job requirements, carrying the original text.
        evidence: Retrieved resume passages. Every item must belong to
            ``profile.candidate_id``.

    Returns:
        The assembled :class:`RagContext`.

    Raises:
        ContextIsolationError: If any evidence item belongs to another
            candidate. This is a hard failure, never a silent filter.
        ValueError: If the job description text is empty.
    """
    items = tuple(evidence)

    foreign = sorted({item.candidate_id for item in items if item.candidate_id != profile.candidate_id})
    if foreign:
        raise ContextIsolationError(
            f"refusing to build context for candidate {profile.candidate_id!r} "
            f"using evidence from {foreign}; candidate evidence must never be mixed"
        )

    job_text = (requirements.raw_text or "").strip()
    if not job_text:
        raise ValueError("job requirements carry no job-description text")

    required = _format_skills(requirements.required_skills)
    minimum = (
        NOT_STATED
        if requirements.minimum_experience is None
        else f"{requirements.minimum_experience:g} years"
    )

    text = "\n".join(
        [
            "JOB DESCRIPTION:",
            job_text,
            "",
            f"Required skills identified in the job description: {required}",
            f"Minimum experience stated in the job description: {minimum}",
            "",
            "---",
            "",
            "CANDIDATE PROFILE:",
            "(extracted deterministically from this candidate's resume)",
            _render_profile(profile),
            "",
            "---",
            "",
            "RETRIEVED RESUME EVIDENCE:",
            f"(verbatim passages from {profile.display_name}'s resume only)",
            "",
            _render_evidence(items),
        ]
    )

    return RagContext(
        candidate_id=profile.candidate_id,
        profile=profile,
        requirements=requirements,
        evidence=items,
        text=text,
    )
