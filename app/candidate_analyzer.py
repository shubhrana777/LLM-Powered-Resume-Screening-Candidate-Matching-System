"""Candidate analysis: combine semantic matching with structured extraction.

This module is composition, not new machinery. PDF parsing comes from
:mod:`app.resume_parser` (Phase 1), ranking from :mod:`app.matching` (Phase 2),
and the structured fields from the Phase 3 extractors. Nothing here re-reads a
PDF or re-implements an embedding.

::

    resume text ──┬─► embeddings ─► FAISS ─► semantic rank ──┐
                  │                                          ├─► CandidateProfile
                  └─► skills / experience / education ───────┘

Everything except the semantic score is deterministic: the same resume and the
same job description always yield the same skills, experience and education.
"""

from __future__ import annotations

from typing import Sequence

from app.embeddings import TextEmbedder
from app.education_extractor import extract_education
from app.experience_extractor import extract_minimum_experience, extract_years_of_experience
from app.matching import CandidateMatcher, validate_job_description
from app.models import Candidate, CandidateProfile, JobRequirements, MatchResult
from app.skill_extractor import (
    DEFAULT_EXTRACTOR,
    SkillExtractor,
    compare_skills,
    extract_required_skills,
    extract_skills,
)

__all__ = [
    "analyze_job_description",
    "compare_experience",
    "build_candidate_profile",
    "analyze_candidates",
    "analyze_candidates_for_job",
]


def analyze_job_description(
    job_description: str,
    extractor: SkillExtractor = DEFAULT_EXTRACTOR,
) -> JobRequirements:
    """Turn job-description text into structured requirements.

    Only explicitly stated requirements are captured. A description that never
    names a skill produces no required skills, and one that never states a
    duration produces ``minimum_experience=None`` -- neither is guessed at.

    Args:
        job_description: The job-description text.
        extractor: Skill extractor supplying the taxonomy.

    Returns:
        The structured :class:`~app.models.JobRequirements`.

    Raises:
        app.matching.EmptyJobDescriptionError: If the text has no content.
    """
    text = validate_job_description(job_description)

    return JobRequirements(
        required_skills=extract_required_skills(text, extractor),
        minimum_experience=extract_minimum_experience(text),
        raw_text=text,
    )


def compare_experience(
    required_years: float | None,
    candidate_years: float | None,
) -> bool | None:
    """Decide whether a candidate meets a stated experience requirement.

    Unknown information is never resolved into a pass or a fail: if either side
    is missing there is no basis for a verdict, so the answer is ``None``.

    Args:
        required_years: Minimum years the job stated, or ``None``.
        candidate_years: Years the resume stated, or ``None``.

    Returns:
        ``True`` if the candidate meets or exceeds the requirement, ``False`` if
        they fall short, ``None`` if either figure is unknown.
    """
    if required_years is None or candidate_years is None:
        return None
    return candidate_years >= required_years


def build_candidate_profile(
    candidate: Candidate,
    requirements: JobRequirements | None = None,
    match_result: MatchResult | None = None,
    extractor: SkillExtractor = DEFAULT_EXTRACTOR,
) -> CandidateProfile:
    """Build a structured profile for one candidate.

    Args:
        candidate: The candidate record, carrying the parsed resume text.
        requirements: Structured job requirements. When ``None``, skills and
            experience are still extracted but nothing is compared.
        match_result: The candidate's Phase 2 ranking, when semantic matching
            has been run. When ``None`` the semantic fields stay ``None``.
        extractor: Skill extractor supplying the taxonomy.

    Returns:
        The :class:`~app.models.CandidateProfile`.

    Raises:
        TypeError: If ``candidate`` is not a :class:`~app.models.Candidate`.
    """
    if not isinstance(candidate, Candidate):
        raise TypeError(f"expected a Candidate, got {type(candidate).__name__}")

    skills = extract_skills(candidate.resume_text, extractor)
    years_experience = extract_years_of_experience(candidate.resume_text)
    education = extract_education(candidate.resume_text)

    required_skills = requirements.required_skills if requirements else ()
    required_experience = requirements.minimum_experience if requirements else None
    comparison = compare_skills(required_skills, skills, extractor)

    return CandidateProfile(
        candidate_id=candidate.candidate_id,
        candidate_name=candidate.candidate_name,
        skills=skills,
        years_experience=years_experience,
        education=education,
        matched_skills=comparison.matched,
        missing_skills=comparison.missing,
        additional_skills=comparison.additional,
        semantic_match_score=match_result.similarity_score if match_result else None,
        rank=match_result.rank if match_result else None,
        required_experience=required_experience,
        meets_experience_requirement=compare_experience(required_experience, years_experience),
        source_path=candidate.source_path,
    )


def analyze_candidates(
    candidates: Sequence[Candidate],
    requirements: JobRequirements | None = None,
    match_results: Sequence[MatchResult] | None = None,
    extractor: SkillExtractor = DEFAULT_EXTRACTOR,
) -> list[CandidateProfile]:
    """Build profiles for several candidates.

    Args:
        candidates: The candidate records.
        requirements: Structured job requirements, or ``None`` to skip comparison.
        match_results: Phase 2 results. Matched to candidates by
            ``candidate_id``, so the two sequences need not be in the same
            order and ``match_results`` may cover only some candidates (for
            example when ``top_k`` was used).
        extractor: Skill extractor supplying the taxonomy.

    Returns:
        Profiles ordered by semantic rank when ranking is available, otherwise
        in the order the candidates were given. Candidates with no matching
        result keep ``rank=None`` and sort last.
    """
    by_id = {result.candidate_id: result for result in (match_results or ())}

    profiles = [
        build_candidate_profile(
            candidate,
            requirements=requirements,
            match_result=by_id.get(candidate.candidate_id),
            extractor=extractor,
        )
        for candidate in candidates
    ]

    if by_id:
        # Unranked candidates sort after ranked ones, keeping input order.
        profiles.sort(key=lambda profile: (profile.rank is None, profile.rank or 0))

    return profiles


def analyze_candidates_for_job(
    candidates: Sequence[Candidate],
    job_description: str,
    top_k: int | None = None,
    embedder: TextEmbedder | None = None,
    extractor: SkillExtractor = DEFAULT_EXTRACTOR,
) -> tuple[JobRequirements, list[CandidateProfile]]:
    """Run the whole pipeline: rank candidates, then analyse them.

    Reuses :class:`app.matching.CandidateMatcher` for the semantic step rather
    than re-implementing embedding or search.

    Args:
        candidates: Non-empty sequence of candidate records.
        job_description: The job-description text.
        top_k: Limit on how many candidates to return. Defaults to all.
        embedder: Optional embedder override, mainly for tests.
        extractor: Skill extractor supplying the taxonomy.

    Returns:
        A ``(requirements, profiles)`` pair, profiles ordered best-match first.

    Raises:
        app.matching.EmptyJobDescriptionError: If the job description is empty.
        app.matching.EmptyCandidateListError: If no candidates are supplied.
    """
    requirements = analyze_job_description(job_description, extractor)

    matcher = CandidateMatcher(embedder=embedder)
    matcher.index_candidates(candidates)
    results = matcher.match(requirements.raw_text, top_k=top_k)

    ranked_ids = {result.candidate_id for result in results}
    ranked_candidates = [c for c in candidates if c.candidate_id in ranked_ids]

    profiles = analyze_candidates(
        ranked_candidates,
        requirements=requirements,
        match_results=results,
        extractor=extractor,
    )
    return requirements, profiles
