"""Conservative extraction of stated years of experience.

Deliberately narrow: a number is returned only when the text *says* it in so
many words, close to the word "experience".

What is **not** done, on purpose:

* No inference from a graduation year ("B.S. 2018" says nothing about experience).
* No summing or differencing of employment dates ("2019-2025" is ignored).
* No guessing from seniority words ("Senior" is not evidence of a number).

When nothing is stated the answer is ``None``, which downstream code must treat
as *unknown* -- never as zero and never as a failure to meet a requirement.

Recognised shapes include ``"4 years of experience"``, ``"3+ years experience"``,
``"2.5 years of professional experience"``, ``"over 7 years of hands-on
experience"``, ``"3-5 years of experience"`` (the lower bound is taken), and
``"Experience: 5 years"``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = [
    "ExperienceMention",
    "find_experience_mentions",
    "extract_years_of_experience",
    "extract_minimum_experience",
]

# Nobody has more than this; a larger number means the text was misread.
MAX_PLAUSIBLE_YEARS = 60.0

# The digit guards stop a fragment of a longer number being read as a duration:
# without them "2015 years" would match the "15".
_NUMBER = r"(?<!\d)(\d{1,2}(?:\.\d+)?)(?!\d)"
_YEARS = r"(?:years?|yrs?)"

# "4 years of experience", "3+ years experience", "3-5 years of experience".
# The gap before "experience" allows qualifiers such as "of professional" but
# excludes '.' and newlines, so it cannot bridge two separate sentences.
_YEARS_THEN_EXPERIENCE = re.compile(
    rf"{_NUMBER}\s*\+?\s*(?:-\s*\d{{1,2}}\s*)?{_YEARS}\b[^.\n]{{0,40}}?\bexperience",
    re.IGNORECASE,
)

# "Experience: 5 years", "experience of 5 years".
_EXPERIENCE_THEN_YEARS = re.compile(
    rf"\bexperience\b[^.\n]{{0,20}}?{_NUMBER}\s*\+?\s*{_YEARS}\b",
    re.IGNORECASE,
)

_PATTERNS = (_YEARS_THEN_EXPERIENCE, _EXPERIENCE_THEN_YEARS)


@dataclass(frozen=True, slots=True)
class ExperienceMention:
    """One explicit statement of years of experience.

    Attributes:
        years: The number of years stated.
        matched_text: The exact substring it came from, kept so a reviewer can
            check the extraction against the resume.
        start: Start offset of the match in the source text.
    """

    years: float
    matched_text: str
    start: int


def find_experience_mentions(text: str) -> tuple[ExperienceMention, ...]:
    """Find every explicit statement of years of experience in ``text``.

    Args:
        text: Resume or job-description text. Non-string or empty input yields
            no mentions.

    Returns:
        Mentions ordered by position in the text. Overlapping matches from
        different patterns at the same offset are reported once.
    """
    if not isinstance(text, str) or not text.strip():
        return ()

    mentions: dict[int, ExperienceMention] = {}

    for pattern in _PATTERNS:
        for match in pattern.finditer(text):
            try:
                years = float(match.group(1))
            except (TypeError, ValueError):  # pragma: no cover - regex guarantees digits
                continue

            if not 0.0 < years <= MAX_PLAUSIBLE_YEARS:
                continue

            mentions.setdefault(
                match.start(),
                ExperienceMention(years=years, matched_text=match.group(0), start=match.start()),
            )

    return tuple(mentions[start] for start in sorted(mentions))


def extract_years_of_experience(text: str) -> float | None:
    """Extract a candidate's years of experience from resume text.

    When a resume states several figures -- an overall total plus per-technology
    ones such as "3 years of experience with Kubernetes" -- the **largest** is
    returned, since that is the one describing overall experience.

    Args:
        text: Cleaned resume text.

    Returns:
        The number of years, or ``None`` if the resume never states one.
    """
    mentions = find_experience_mentions(text)
    return max(mention.years for mention in mentions) if mentions else None


def extract_minimum_experience(text: str) -> float | None:
    """Extract the minimum experience a job description asks for.

    The mirror image of :func:`extract_years_of_experience`: where a posting
    states several figures ("3+ years required, 5+ preferred"), the **smallest**
    is returned, because that is the bar a candidate has to clear.

    Args:
        text: Job-description text.

    Returns:
        The minimum number of years, or ``None`` if none is stated.
    """
    mentions = find_experience_mentions(text)
    return min(mention.years for mention in mentions) if mentions else None
