"""Deterministic skill extraction from resume and job-description text.

No model and no LLM: extraction is exact alias matching against
:mod:`app.skill_taxonomy`, so the same text always produces the same skills and
every hit can be traced back to the substring that produced it.

Why not plain substring matching
--------------------------------
``"SQL" in text`` reports SQL for *PostgreSQL*, *MySQL* and *sqlalchemy*; ``"R"``
reports R for every word containing the letter. Each alias is therefore compiled
into a regex guarded by custom boundaries:

* **Left**  ``(?<![A-Za-z0-9+#&])`` -- no alphanumeric, ``+``, ``#`` or ``&``
  immediately before. Stops *MySQL* matching ``SQL`` and *R&D* matching ``R``.
* **Right** ``(?![A-Za-z0-9+#&])`` -- likewise after. Stops ``C`` matching
  *C++*, and *sqlalchemy* matching ``SQL``.
* **Right** ``(?!\\.\\w)`` -- a dot followed by a word character is treated as
  part of a longer token, so ``Node`` does not match inside *Node.js* while
  ``Python`` still matches in *"Python."* at the end of a sentence.

``\\b`` alone cannot do this: it is defined in terms of ``\\w``, so it behaves
incorrectly around ``+``, ``#`` and ``.``, which are exactly the characters that
appear in real skill names.

Two further rules keep matching predictable:

* Matching is case-insensitive, **except for single-character skills** such as
  ``C`` and ``R``, which must be capitalised in the source text. Without this a
  list item like ``"c) managed the team"`` would register the C language.
* Spaces, hyphens and underscores inside an alias match any run of the same, so
  ``"Power BI"`` also matches ``power-bi`` and a ``Power``/``BI`` line break.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Sequence

from app.models import SkillComparison
from app.skill_taxonomy import DEFAULT_TAXONOMY, SkillDefinition, SkillTaxonomy

__all__ = [
    "SkillMention",
    "SkillExtractor",
    "DEFAULT_EXTRACTOR",
    "extract_skills",
    "extract_required_skills",
    "compare_skills",
]

# Characters that, when adjacent to a match, mean it is part of a longer token.
_BOUNDARY_CHARS = r"A-Za-z0-9+#&"
_LEFT_BOUNDARY = rf"(?<![{_BOUNDARY_CHARS}])"
_RIGHT_BOUNDARY = rf"(?![{_BOUNDARY_CHARS}])(?!\.\w)"

# Inside an alias, any of these separates words and matches any run of them.
_SEPARATOR = r"[\s\-_]*"
_SEPARATOR_SPLIT = re.compile(r"[\s\-_]+")


def _compile_alias(alias: str) -> re.Pattern[str]:
    """Compile one alias into a boundary-guarded regex.

    Args:
        alias: A skill name or alias, e.g. ``"C++"`` or ``"Power BI"``.

    Returns:
        The compiled pattern. Single-character aliases are case-sensitive; all
        others are case-insensitive.
    """
    parts = [re.escape(part) for part in _SEPARATOR_SPLIT.split(alias.strip()) if part]
    core = _SEPARATOR.join(parts)
    pattern = f"{_LEFT_BOUNDARY}{core}{_RIGHT_BOUNDARY}"

    # "c)" or "r." should not register the C or R languages, so a bare letter
    # must be capitalised to count.
    flags = 0 if len(alias.strip()) == 1 else re.IGNORECASE
    return re.compile(pattern, flags)


@dataclass(frozen=True, slots=True)
class SkillMention:
    """One occurrence of a skill in a piece of text.

    Attributes:
        skill: Canonical skill name.
        category: Taxonomy category the skill belongs to.
        matched_text: The exact substring that matched, kept for explainability.
        start: Start offset of the match in the source text.
        end: End offset of the match in the source text.
    """

    skill: str
    category: str
    matched_text: str
    start: int
    end: int


class SkillExtractor:
    """Finds taxonomy skills in free text.

    Alias patterns are compiled once at construction, so a single extractor
    should be reused across many documents.

    Args:
        taxonomy: Vocabulary to recognise. Defaults to
            :data:`app.skill_taxonomy.DEFAULT_TAXONOMY`.
    """

    def __init__(self, taxonomy: SkillTaxonomy = DEFAULT_TAXONOMY) -> None:
        self._taxonomy = taxonomy
        self._patterns: tuple[tuple[SkillDefinition, re.Pattern[str]], ...] = tuple(
            (definition, _compile_alias(alias))
            for definition in taxonomy
            for alias in definition.all_names
        )
        # Declaration order gives a stable, category-grouped output ordering
        # that does not depend on how the source text happens to be phrased.
        self._order: dict[str, int] = {
            definition.name: position for position, definition in enumerate(taxonomy)
        }

    @property
    def taxonomy(self) -> SkillTaxonomy:
        """The taxonomy backing this extractor."""
        return self._taxonomy

    def find_mentions(self, text: str) -> tuple[SkillMention, ...]:
        """Find every occurrence of every known skill in ``text``.

        Args:
            text: Resume or job-description text. Non-string or empty input
                yields no mentions.

        Returns:
            Mentions sorted by position in the text, then by skill name. A skill
            appearing several times produces several mentions.
        """
        if not isinstance(text, str) or not text.strip():
            return ()

        mentions = [
            SkillMention(
                skill=definition.name,
                category=definition.category,
                matched_text=match.group(0),
                start=match.start(),
                end=match.end(),
            )
            for definition, pattern in self._patterns
            for match in pattern.finditer(text)
        ]

        mentions.sort(key=lambda mention: (mention.start, mention.skill))
        return tuple(mentions)

    def extract(self, text: str) -> tuple[str, ...]:
        """Extract the distinct skills present in ``text``.

        Args:
            text: Resume or job-description text. Non-string or empty input
                yields an empty result rather than an error, since a resume
                legitimately may list no known skills.

        Returns:
            Canonical skill names, deduplicated, in taxonomy declaration order.
        """
        found = {mention.skill for mention in self.find_mentions(text)}
        return tuple(sorted(found, key=lambda name: self._order[name]))

    def extract_by_category(self, text: str) -> dict[str, tuple[str, ...]]:
        """Group the extracted skills by taxonomy category.

        Args:
            text: Resume or job-description text.

        Returns:
            A mapping of category to canonical skill names. Categories with no
            matches are omitted.
        """
        grouped: dict[str, list[str]] = {}
        for skill in self.extract(text):
            category = self._taxonomy.category_of(skill)
            if category is not None:
                grouped.setdefault(category, []).append(skill)
        return {category: tuple(skills) for category, skills in grouped.items()}

    def order_key(self, skill: str) -> int:
        """Sort position of ``skill`` in the taxonomy, for deterministic output."""
        return self._order.get(skill, len(self._order))


DEFAULT_EXTRACTOR = SkillExtractor()


def extract_skills(text: str, extractor: SkillExtractor = DEFAULT_EXTRACTOR) -> tuple[str, ...]:
    """Extract skills from resume text.

    Args:
        text: Cleaned resume text.
        extractor: Extractor to use. Defaults to the shared one.

    Returns:
        Canonical skill names in taxonomy order.
    """
    return extractor.extract(text)


def extract_required_skills(
    text: str, extractor: SkillExtractor = DEFAULT_EXTRACTOR
) -> tuple[str, ...]:
    """Extract required skills from job-description text.

    Identical mechanics to :func:`extract_skills`; the separate name documents
    intent at call sites. Only skills explicitly named in the text are returned
    -- nothing is inferred about what the role "probably" needs.

    Args:
        text: Job-description text.
        extractor: Extractor to use. Defaults to the shared one.

    Returns:
        Canonical skill names in taxonomy order.
    """
    return extractor.extract(text)


def compare_skills(
    required_skills: Sequence[str],
    candidate_skills: Sequence[str],
    extractor: SkillExtractor = DEFAULT_EXTRACTOR,
) -> SkillComparison:
    """Compare a candidate's skills against a job's required skills.

    Comparison is by canonical name, so an alias on one side still matches the
    canonical form on the other. Unknown names are compared case-insensitively
    and passed through unchanged, which keeps the function usable with a
    hand-written skill list.

    Args:
        required_skills: Skills the job asks for.
        candidate_skills: Skills found on the resume.
        extractor: Supplies the taxonomy used for canonicalisation and ordering.

    Returns:
        A :class:`~app.models.SkillComparison`. ``matched`` and ``missing``
        preserve the order of ``required_skills``; ``additional`` preserves the
        order of ``candidate_skills``.
    """
    taxonomy = extractor.taxonomy

    def canonical(name: str) -> str:
        return taxonomy.canonical_name(name) or name.strip()

    def key(name: str) -> str:
        return canonical(name).lower()

    required_canonical: list[str] = []
    seen_required: set[str] = set()
    for name in _clean_names(required_skills):
        if key(name) not in seen_required:
            seen_required.add(key(name))
            required_canonical.append(canonical(name))

    candidate_canonical: list[str] = []
    seen_candidate: set[str] = set()
    for name in _clean_names(candidate_skills):
        if key(name) not in seen_candidate:
            seen_candidate.add(key(name))
            candidate_canonical.append(canonical(name))

    candidate_keys = {name.lower() for name in candidate_canonical}
    required_keys = {name.lower() for name in required_canonical}

    return SkillComparison(
        matched=tuple(n for n in required_canonical if n.lower() in candidate_keys),
        missing=tuple(n for n in required_canonical if n.lower() not in candidate_keys),
        additional=tuple(n for n in candidate_canonical if n.lower() not in required_keys),
    )


def _clean_names(names: Iterable[str]) -> list[str]:
    """Drop non-string and blank entries from a skill list."""
    if isinstance(names, str):
        raise TypeError("expected a sequence of skill names, not a single string")
    return [name for name in names if isinstance(name, str) and name.strip()]
