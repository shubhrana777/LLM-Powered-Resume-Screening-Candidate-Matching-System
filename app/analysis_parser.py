"""Parse and validate LLM output into a grounded :class:`CandidateAnalysis`.

This module is the enforcement half of the hallucination safeguards. The prompt
in :mod:`app.prompts` *asks* the model to stay grounded; this module *checks*
whether it did, and corrects the result when it did not.

What is verified against the deterministic Phase 3 profile:

* ``recommendation`` is in the controlled vocabulary, else it becomes
  ``INSUFFICIENT_INFORMATION``.
* ``matched_skills`` contains only skills actually extracted from this
  candidate's resume. A model claiming AWS for a resume that never mentions AWS
  has the claim removed, not merely flagged.
* ``skill_gaps`` contains only skills the job actually asked for.
* ``experience_assessment`` may not assert a number of years when the resume
  states none; such an assessment is replaced with "Not stated".
* Degrees named in the prose must appear in the extracted education, reusing the
  Phase 3 extractor to spot invented credentials.

Evidence is never taken from the model. The pipeline attaches the passages that
were actually retrieved, so fabricated citations are structurally impossible.

Every correction is recorded in ``CandidateAnalysis.warnings``. An analysis with
warnings was repaired, and ``is_grounded`` is ``False``.

This reduces hallucination; it does not eliminate it. Free prose can still be
subtly wrong in ways no automated check catches, which is why the retrieved
evidence travels with the analysis for a human to read.
"""

from __future__ import annotations

import json
import re
from typing import Any, Sequence

from app.education_extractor import extract_education
from app.models import (
    NOT_STATED,
    CandidateAnalysis,
    CandidateProfile,
    Recommendation,
)

__all__ = [
    "AnalysisParseError",
    "extract_json_object",
    "parse_candidate_analysis",
]

# "4 years", "4.5 yrs", "10 year" -- a claim about a duration.
_YEARS_CLAIM = re.compile(r"(?<!\d)(\d{1,2}(?:\.\d+)?)(?!\d)\s*\+?\s*(?:years?|yrs?)\b", re.IGNORECASE)

_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


class AnalysisParseError(ValueError):
    """The model response could not be read as an analysis object."""


def extract_json_object(raw: str) -> dict[str, Any]:
    """Pull a single JSON object out of a model response.

    Tolerates the usual wrappers -- a markdown fence, or a sentence before or
    after the object -- because those are formatting noise rather than a
    substantive failure. Anything else is rejected instead of guessed at.

    Args:
        raw: The raw text returned by the provider.

    Returns:
        The decoded JSON object.

    Raises:
        AnalysisParseError: If no JSON object can be found or decoded, or the
            top-level value is not an object.
    """
    if not isinstance(raw, str) or not raw.strip():
        raise AnalysisParseError("the model returned an empty response")

    text = raw.strip()

    fenced = _JSON_FENCE.search(text)
    if fenced:
        text = fenced.group(1).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise AnalysisParseError(
            f"no JSON object found in the model response: {raw[:200]!r}"
        )

    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise AnalysisParseError(f"the model response is not valid JSON ({exc})") from exc

    if not isinstance(payload, dict):
        raise AnalysisParseError(
            f"expected a JSON object, got {type(payload).__name__}"
        )

    return payload


def _as_text(value: Any, fallback: str = NOT_STATED) -> str:
    """Coerce a JSON value to a trimmed string, or the fallback when unusable."""
    if isinstance(value, str) and value.strip():
        return value.strip()
    return fallback


def _as_string_list(value: Any) -> list[str]:
    """Coerce a JSON value to a list of non-empty strings."""
    if isinstance(value, str) or not isinstance(value, (list, tuple)):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def _restrict_to(
    claimed: Sequence[str],
    allowed: Sequence[str],
    label: str,
    warnings: list[str],
) -> tuple[str, ...]:
    """Keep only claimed entries that appear in ``allowed``, warning about the rest.

    Args:
        claimed: Entries the model returned.
        allowed: Entries the deterministic profile supports.
        label: Field name, used in the warning text.
        warnings: Collector appended to in place.

    Returns:
        The supported entries, using the canonical spelling from ``allowed``.
    """
    canonical = {entry.strip().lower(): entry for entry in allowed}

    kept: list[str] = []
    dropped: list[str] = []
    seen: set[str] = set()

    for entry in claimed:
        key = entry.strip().lower()
        if key in canonical:
            if key not in seen:
                seen.add(key)
                kept.append(canonical[key])
        else:
            dropped.append(entry)

    if dropped:
        warnings.append(
            f"removed {len(dropped)} unsupported entr"
            f"{'y' if len(dropped) == 1 else 'ies'} from {label}: {', '.join(dropped)}"
        )

    return tuple(kept)


def _check_experience_claim(
    assessment: str,
    profile: CandidateProfile,
    warnings: list[str],
) -> str:
    """Reject an experience assessment that invents a duration.

    Args:
        assessment: The model's experience text.
        profile: The deterministic profile, holding what the resume states.
        warnings: Collector appended to in place.

    Returns:
        The assessment, or a safe replacement when it asserts years the resume
        never stated.
    """
    if profile.years_experience is not None:
        return assessment

    required = profile.required_experience
    invented = [
        value
        for value in (float(match) for match in _YEARS_CLAIM.findall(assessment))
        if required is None or value != required
    ]

    if not invented:
        return assessment

    warnings.append(
        "replaced an experience assessment that asserted "
        f"{invented[0]:g} years; the resume states no number of years"
    )

    requirement = (
        "The job description states no minimum either."
        if required is None
        else f"The job asks for {required:g} years."
    )
    return (
        f"{NOT_STATED}. The resume does not state a number of years of experience. "
        f"{requirement}"
    )


def _check_education_claims(
    text: str,
    profile: CandidateProfile,
    warnings: list[str],
) -> None:
    """Warn when prose names a degree the resume does not contain.

    Reuses the Phase 3 education extractor on the model's own words, so an
    invented credential is caught by the same rules that read the resume.

    Args:
        text: Model prose to scan.
        profile: The deterministic profile.
        warnings: Collector appended to in place.
    """
    supported = {entry.degree.lower() for entry in profile.education}
    claimed = {entry.degree for entry in extract_education(text)}

    invented = sorted(degree for degree in claimed if degree.lower() not in supported)
    if invented:
        warnings.append(
            f"analysis mentions {', '.join(invented)}, which the resume does not state"
        )


def parse_candidate_analysis(
    raw: str,
    profile: CandidateProfile,
    evidence: Sequence[object] = (),
    model_name: str = "unknown",
) -> CandidateAnalysis:
    """Parse a model response and ground it against the candidate profile.

    Args:
        raw: The provider's raw text response.
        profile: The Phase 3 profile for this candidate, used as ground truth.
        evidence: The passages actually retrieved. These are attached as-is;
            the model does not get to supply its own evidence.
        model_name: Identifier of the provider and model, recorded on the result.

    Returns:
        A validated :class:`~app.models.CandidateAnalysis`. Any correction made
        during validation is listed in ``warnings``.

    Raises:
        AnalysisParseError: If the response is not a readable JSON object.
    """
    payload = extract_json_object(raw)
    warnings: list[str] = []

    recommendation = Recommendation.parse(payload.get("recommendation"))
    if recommendation is None:
        warnings.append(
            f"unrecognised recommendation {payload.get('recommendation')!r}; "
            "defaulted to INSUFFICIENT_INFORMATION"
        )
        recommendation = Recommendation.INSUFFICIENT_INFORMATION

    summary = _as_text(payload.get("summary"))
    if summary == NOT_STATED:
        warnings.append("the response contained no usable summary")

    matched = _restrict_to(
        _as_string_list(payload.get("matched_skills")),
        profile.skills,
        "matched_skills",
        warnings,
    )
    gaps = _restrict_to(
        _as_string_list(payload.get("skill_gaps")),
        tuple(profile.missing_skills) + tuple(profile.matched_skills),
        "skill_gaps",
        warnings,
    )

    experience = _check_experience_claim(
        _as_text(payload.get("experience_assessment")), profile, warnings
    )

    _check_education_claims(f"{summary}\n{experience}", profile, warnings)

    if profile.years_experience is None and _YEARS_CLAIM.search(summary):
        stated = {
            float(match)
            for match in _YEARS_CLAIM.findall(summary)
            if profile.required_experience is None
            or float(match) != profile.required_experience
        }
        if stated:
            warnings.append(
                "the summary mentions a number of years that the resume does not state"
            )

    limitations = list(_as_string_list(payload.get("limitations")))
    if warnings:
        limitations.append(
            "Parts of the model response were not supported by the resume and were "
            "corrected during validation; see warnings."
        )

    return CandidateAnalysis(
        candidate_id=profile.candidate_id,
        candidate_name=profile.candidate_name,
        summary=summary,
        recommendation=recommendation,
        matched_skills=matched,
        skill_gaps=gaps,
        experience_assessment=experience,
        evidence=tuple(evidence),
        limitations=tuple(limitations),
        model_name=model_name,
        warnings=tuple(warnings),
        # Straight from the deterministic profile: the model is never asked
        # about education, so there is nothing here to validate or correct.
        education=tuple(str(entry) for entry in profile.education),
    )
