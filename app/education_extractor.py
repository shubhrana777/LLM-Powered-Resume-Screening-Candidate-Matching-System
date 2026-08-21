"""Basic deterministic extraction of education credentials.

Scans line by line for a known degree pattern and, when the same line names a
field of study, captures that too. No semantic reasoning: a line either matches
a declared degree spelling or it does not.

Deliberate omissions
--------------------
* Bare ``BS`` and ``MS`` are **not** recognised, because "MS Excel" and "MS
  Office" appear on far more resumes than "MS Physics". The dotted forms
  (``M.S.``), the ``MSc``/``BSc`` forms, and the spelled-out forms are.
* A degree and its field must be on the same line; nothing is stitched across
  line breaks.
* No institution parsing, no graduation dates, no ranking of credentials.
"""

from __future__ import annotations

import re

from app.models import EducationEntry

__all__ = [
    "DEGREE_PATTERNS",
    "extract_education",
    "extract_highest_degree",
]

# Longest, most specific spellings first: "Bachelor of Science" must win over
# "Bachelor" on the same line. Each entry is (canonical degree, alternatives).
_DEGREE_SPELLINGS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Master of Business Administration", ("Master of Business Administration",)),
    ("MBA", ("M.B.A.", "MBA")),
    ("Bachelor of Science", ("Bachelor of Science", "B.Sc.", "BSc", "B.S.", "B.Sc")),
    ("Bachelor of Arts", ("Bachelor of Arts", "B.A.", "BA.")),
    ("Bachelor of Technology", ("Bachelor of Technology", "B.Tech.", "B.Tech", "BTech")),
    ("Bachelor of Engineering", ("Bachelor of Engineering", "B.E.", "B.Eng.", "B.Eng", "BEng")),
    ("Bachelor of Commerce", ("Bachelor of Commerce", "B.Com.", "B.Com", "BCom")),
    ("Master of Science", ("Master of Science", "M.Sc.", "MSc", "M.S.", "M.Sc")),
    ("Master of Arts", ("Master of Arts", "M.A.")),
    ("Master of Technology", ("Master of Technology", "M.Tech.", "M.Tech", "MTech")),
    ("Master of Engineering", ("Master of Engineering", "M.E.", "M.Eng.", "M.Eng", "MEng")),
    ("PhD", ("Ph.D.", "PhD", "Ph.D", "Doctor of Philosophy", "Doctorate")),
    ("Bachelor's Degree", ("Bachelor's", "Bachelors", "Bachelor")),
    ("Master's Degree", ("Master's", "Masters", "Master")),
    ("Associate Degree", ("Associate's Degree", "Associate Degree", "Associate's")),
    ("Diploma", ("Diploma",)),
)

# Ordering by descending literal length inside the compiled alternation makes
# the regex prefer the most specific spelling at any given position.
_BOUNDARY_LEFT = r"(?<![A-Za-z0-9])"
_BOUNDARY_RIGHT = r"(?![A-Za-z0-9])"


def _compile_degree(alternatives: tuple[str, ...]) -> re.Pattern[str]:
    """Compile one degree's spellings into a single boundary-guarded regex."""
    ordered = sorted(alternatives, key=len, reverse=True)
    alternation = "|".join(re.escape(name) for name in ordered)
    return re.compile(rf"{_BOUNDARY_LEFT}(?:{alternation}){_BOUNDARY_RIGHT}", re.IGNORECASE)


DEGREE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (canonical, _compile_degree(alternatives))
    for canonical, alternatives in _DEGREE_SPELLINGS
)

# Connectors between a degree and its field: "B.S. in Physics", "MBA - Finance".
_FIELD_LEAD = re.compile(r"^\s*(?:in|of|:|-|--|–|—|,)\s*", re.IGNORECASE)

# A field ends at an institution, a bracket, or a year.
_FIELD_STOP = re.compile(
    r"\s*(?:,|\(|\||–|—|\bat\b|\bfrom\b|\d{4}).*$",
    re.IGNORECASE,
)

MAX_FIELD_LENGTH = 60


def _extract_field(remainder: str) -> str | None:
    """Pull a field of study out of the text following a degree on the same line.

    Args:
        remainder: Whatever followed the degree match on that line.

    Returns:
        The field of study, or ``None`` when nothing usable follows.
    """
    text = _FIELD_LEAD.sub("", remainder, count=1)
    text = _FIELD_STOP.sub("", text).strip(" .,-–—:")

    if not text or len(text) > MAX_FIELD_LENGTH or not re.search(r"[A-Za-z]", text):
        return None

    return " ".join(text.split())


def extract_education(text: str) -> tuple[EducationEntry, ...]:
    """Extract education credentials from resume text.

    Args:
        text: Cleaned resume text. Non-string or empty input yields an empty
            tuple, since a resume may legitimately list no education.

    Returns:
        One :class:`~app.models.EducationEntry` per distinct degree/field pair,
        in order of appearance. Empty when nothing matched -- absence of a
        credential is reported as absence, never guessed at.
    """
    if not isinstance(text, str) or not text.strip():
        return ()

    entries: list[EducationEntry] = []
    seen: set[tuple[str, str | None]] = set()

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        best: tuple[int, str, re.Match[str]] | None = None
        for canonical, pattern in DEGREE_PATTERNS:
            match = pattern.search(stripped)
            if match is None:
                continue
            # Prefer the earliest match, and the longest one at that position,
            # so "Bachelor of Science" beats "Bachelor".
            candidate = (match.start(), canonical, match)
            if (
                best is None
                or candidate[0] < best[0]
                or (candidate[0] == best[0] and len(match.group(0)) > len(best[2].group(0)))
            ):
                best = candidate

        if best is None:
            continue

        _, canonical, match = best
        field = _extract_field(stripped[match.end() :])

        key = (canonical, field)
        if key not in seen:
            seen.add(key)
            entries.append(EducationEntry(degree=canonical, field=field, raw_text=stripped))

    return tuple(entries)


def extract_highest_degree(text: str) -> EducationEntry | None:
    """Return the most advanced degree found, or ``None``.

    "Most advanced" is decided by a fixed ranking of degree levels, not by any
    judgement about institutions or fields.

    Args:
        text: Cleaned resume text.

    Returns:
        The highest-ranked entry, or ``None`` when no degree was found. Where
        several entries tie, the first one in the text wins.
    """
    entries = extract_education(text)
    if not entries:
        return None

    return max(entries, key=lambda entry: _DEGREE_RANK.get(entry.degree, 0))


# Higher means more advanced. Used only for `extract_highest_degree`.
_DEGREE_RANK: dict[str, int] = {
    "Diploma": 1,
    "Associate Degree": 2,
    "Bachelor's Degree": 3,
    "Bachelor of Arts": 3,
    "Bachelor of Science": 3,
    "Bachelor of Commerce": 3,
    "Bachelor of Engineering": 3,
    "Bachelor of Technology": 3,
    "Master's Degree": 4,
    "Master of Arts": 4,
    "Master of Science": 4,
    "Master of Engineering": 4,
    "Master of Technology": 4,
    "MBA": 4,
    "Master of Business Administration": 4,
    "PhD": 5,
}
