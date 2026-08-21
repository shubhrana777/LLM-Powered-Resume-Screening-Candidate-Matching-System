"""Pure display helpers.

No Streamlit, no HTTP, no state -- just values in and display values out, which
is why this module carries most of the dashboard's test coverage.

Two rules come from the backend's own contract and are easy to break in a UI:

* A similarity score is shown as a **percentage of the cosine scale** for
  readability -- ``0.5935`` reads as ``59.35%`` -- and is never described as a
  probability, a confidence, or a share of requirements met. The wording that
  travels with it says so explicitly; see :data:`SIMILARITY_MEANING`. The
  backend value is untouched: this module only formats.
* Unknown is **never** rendered as zero or as a failure. A candidate who has
  not been through the analysis step reads :data:`NOT_ANALYZED` -- "Not analyzed
  yet" -- which describes a step not taken, not a step that failed.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, NamedTuple

__all__ = [
    "NOT_ANALYZED",
    "NOT_ANALYZED_HINT",
    "SIMILARITY_MEANING",
    "Band",
    "score_band",
    "format_similarity",
    "format_similarity_raw",
    "to_percent",
    "recommendation_label",
    "recommendation_tone",
    "RECOMMENDATION_ORDER",
    "skill_coverage",
    "format_coverage",
    "experience_status",
    "grounding_label",
    "truncate",
    "plural",
    "candidate_display_name",
]

# "Not analyzed yet" rather than "Not analysed": ranking and analysis are two
# separate operations, and this label has to read as a step the recruiter has
# not run, never as a step the system tried and failed.
NOT_ANALYZED = "Not analyzed yet"

NOT_ANALYZED_HINT = (
    "Run candidate analysis to calculate skill coverage, experience fit, "
    "and recommendation."
)

SIMILARITY_MEANING = (
    "Semantic similarity between the job description and candidate resume "
    "embeddings. This is a ranking signal, not a hiring probability or "
    "percentage of requirements met."
)

# Retained under its old name so nothing that imports it breaks; the value is
# the new wording.
NOT_AVAILABLE = NOT_ANALYZED

NOT_STATED = "Not stated"


class Band(NamedTuple):
    """A qualitative reading of a similarity score.

    Attributes:
        label: Word for the band, so meaning never rests on colour alone.
        tone: Key into :data:`app.ui.theme.TONES`.
    """

    label: str
    tone: str


# Thresholds for the bundled MiniLM model, taken from the ranges the project's
# own README documents on real sample data: a strong resume/job pair lands
# around 0.70-0.80, a loosely related one 0.35-0.50, unrelated near 0.15.
# They describe *this* embedding model and are not a general scale.
STRONG_SCORE = 0.55
MODERATE_SCORE = 0.35


def score_band(score: float | None) -> Band:
    """Describe a similarity score in words.

    This is a reading aid, not a verdict: the bands describe how close two texts
    sit in the embedding space, nothing about a candidate's suitability.

    Args:
        score: Cosine similarity, or ``None`` when there is no score.

    Returns:
        The :class:`Band` for that score.
    """
    if score is None:
        return Band("No score", "neutral")
    if score >= STRONG_SCORE:
        return Band("Strong similarity", "positive")
    if score >= MODERATE_SCORE:
        return Band("Moderate similarity", "caution")
    return Band("Low similarity", "neutral")


def to_percent(score: float | None) -> float | None:
    """Scale a cosine similarity onto 0-100 for display.

    A pure rescaling of the same number -- ``0.5935`` becomes ``59.35`` -- so
    the value can be sorted numerically in a table while being rendered with a
    ``%`` suffix. It does not change what the number measures.

    Args:
        score: Cosine similarity, or ``None``.

    Returns:
        The scaled value, or ``None``.
    """
    if score is None:
        return None
    return score * 100.0


def format_similarity(score: float | None) -> str:
    """Render a similarity score as a percentage of the cosine scale.

    ``0.5935`` renders as ``"59.35%"``. The percent sign is a readability
    convention for a value that runs 0-1, **not** a claim about probability:
    every place this appears is accompanied by :data:`SIMILARITY_MEANING`.

    Args:
        score: Cosine similarity, or ``None``.

    Returns:
        The formatted score, or :data:`NOT_ANALYZED`.
    """
    if score is None:
        return NOT_ANALYZED
    return f"{score * 100:.2f}%"


def format_similarity_raw(score: float | None) -> str:
    """Render the underlying cosine value, for tooltips and provenance.

    Shown beside the percentage so the number the backend actually returned is
    never hidden from the recruiter.

    Args:
        score: Cosine similarity, or ``None``.

    Returns:
        The raw value to four decimals, or :data:`NOT_ANALYZED`.
    """
    if score is None:
        return NOT_ANALYZED
    return f"{score:.4f}"


# Retained under its old name so existing callers keep working.
format_score = format_similarity


RECOMMENDATION_ORDER: tuple[str, ...] = (
    "STRONG_MATCH",
    "GOOD_MATCH",
    "PARTIAL_MATCH",
    "WEAK_MATCH",
    "INSUFFICIENT_INFORMATION",
)

_RECOMMENDATION_LABELS: dict[str, str] = {
    "STRONG_MATCH": "Strong match",
    "GOOD_MATCH": "Good match",
    "PARTIAL_MATCH": "Partial match",
    "WEAK_MATCH": "Weak match",
    "INSUFFICIENT_INFORMATION": "Insufficient information",
}

_RECOMMENDATION_TONES: dict[str, str] = {
    "STRONG_MATCH": "positive",
    "GOOD_MATCH": "positive",
    "PARTIAL_MATCH": "caution",
    "WEAK_MATCH": "critical",
    "INSUFFICIENT_INFORMATION": "neutral",
}


def recommendation_label(value: str | None) -> str:
    """Render a recommendation as readable text.

    Args:
        value: A backend recommendation value, or ``None``.

    Returns:
        A human label. An unrecognised value is returned unchanged rather than
        being mapped to the nearest-looking one.
    """
    if not value:
        return NOT_ANALYZED
    return _RECOMMENDATION_LABELS.get(value, value.replace("_", " ").capitalize())


def recommendation_tone(value: str | None) -> str:
    """Return the tone key for a recommendation.

    Args:
        value: A backend recommendation value, or ``None``.

    Returns:
        A key into :data:`app.ui.theme.TONES`, defaulting to ``"neutral"``.
    """
    if not value:
        return "neutral"
    return _RECOMMENDATION_TONES.get(value, "neutral")


def skill_coverage(matched: Iterable[str] | None, gaps: Iterable[str] | None) -> float | None:
    """Fraction of the job's required skills the evidence supports.

    A plain count ratio and nothing more -- every skill counts the same, and only
    skills in the backend's taxonomy are counted at all.

    Args:
        matched: Skills the analysis supported.
        gaps: Required skills it did not.

    Returns:
        The ratio in ``[0.0, 1.0]``, or ``None`` when the job named no
        recognised skills, in which case there is nothing to take a fraction of.
    """
    have = len(list(matched or ()))
    missing = len(list(gaps or ()))
    total = have + missing
    if total == 0:
        return None
    return have / total


def format_coverage(matched: Iterable[str] | None, gaps: Iterable[str] | None) -> str:
    """Render skill coverage as a count, e.g. ``"12 / 13"``.

    A count rather than a percentage, so it reads as what it is: how many named
    skills were found, out of how many were asked for.

    Args:
        matched: Skills the analysis supported.
        gaps: Required skills it did not.

    Returns:
        The formatted coverage, or a note that no skills were recognised.
    """
    have = len(list(matched or ()))
    missing = len(list(gaps or ()))
    total = have + missing
    if total == 0:
        return "No skills named"
    return f"{have} / {total}"


def experience_status(assessment: str | None) -> Band:
    """Summarise the backend's experience assessment as a status.

    The backend writes this assessment in a fixed form, so reading it is
    reliable; but an unrecognised phrasing is reported as unknown rather than
    guessed at, because treating unknown as a pass or a fail is exactly the
    mistake the earlier phases were careful to avoid.

    Args:
        assessment: The ``experience_assessment`` string, or ``None``.

    Returns:
        A :class:`Band` whose label is one of "Requirement met", "Below
        requirement", "Not stated" or "Not analysed".
    """
    if not assessment:
        return Band(NOT_ANALYZED, "neutral")

    text = assessment.strip()
    if text.startswith(NOT_STATED):
        return Band(NOT_STATED, "neutral")

    lowered = text.casefold()
    if "requirement met: yes" in lowered:
        return Band("Requirement met", "positive")
    if "requirement met: no" in lowered:
        return Band("Below requirement", "critical")
    return Band("Not stated", "neutral")


def grounding_label(is_grounded: bool | None) -> Band:
    """Describe whether an analysis needed correcting.

    Args:
        is_grounded: The backend's ``is_grounded`` flag.

    Returns:
        A :class:`Band` labelling the grounding outcome.
    """
    if is_grounded is None:
        return Band(NOT_ANALYZED, "neutral")
    if is_grounded:
        return Band("Grounded", "positive")
    return Band("Corrected claims", "caution")


def truncate(text: str | None, limit: int = 160) -> str:
    """Shorten text for a table cell, marking that it was shortened.

    Args:
        text: The text, or ``None``.
        limit: Maximum characters before an ellipsis is added.

    Returns:
        The text, shortened at a word boundary where possible.
    """
    if not text:
        return ""
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    clipped = collapsed[:limit].rsplit(" ", 1)[0] or collapsed[:limit]
    return f"{clipped}…"


def plural(count: int, singular: str, plural_form: str | None = None) -> str:
    """Render a count with a correctly pluralised noun.

    Args:
        count: How many.
        singular: The singular noun.
        plural_form: The plural, defaulting to ``singular + "s"``.

    Returns:
        For example ``"1 candidate"`` or ``"3 candidates"``.
    """
    word = singular if count == 1 else (plural_form or f"{singular}s")
    return f"{count} {word}"


def candidate_display_name(record: Mapping[str, Any]) -> str:
    """Pick the best available name from a candidate-ish payload.

    The API uses ``name`` in listings and ``candidate`` in match and analysis
    responses; both carry ``candidate_id`` as a fallback.

    Args:
        record: Any candidate, match or analysis payload.

    Returns:
        A display name, falling back to the id and then to a placeholder.
    """
    for key in ("name", "candidate", "candidate_id"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return "Unknown candidate"
