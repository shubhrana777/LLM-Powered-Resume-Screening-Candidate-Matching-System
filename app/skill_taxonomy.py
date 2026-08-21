"""Configurable skill taxonomy.

Every skill the system can recognise is declared here, in one place, rather than
being scattered through the extraction code. A skill has a canonical name, a
category, and any number of aliases (spellings, abbreviations, plurals).

Extending the taxonomy
----------------------
Three options, in increasing order of separation from the code:

1. Add an entry to :data:`DEFAULT_SKILL_DEFINITIONS` below.
2. Build a taxonomy at runtime::

       taxonomy = DEFAULT_TAXONOMY.extended([
           SkillDefinition("Rust", "Programming", ("rustlang",)),
       ])

3. Load one from a JSON file, keeping the vocabulary out of the codebase::

       taxonomy = SkillTaxonomy.from_json_file("my_skills.json")

   The JSON format is ``{"Category": {"Canonical Name": ["alias", ...]}}``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

__all__ = [
    "TaxonomyError",
    "SkillDefinition",
    "SkillTaxonomy",
    "DEFAULT_SKILL_DEFINITIONS",
    "DEFAULT_TAXONOMY",
    "normalize_skill_name",
]


class TaxonomyError(ValueError):
    """The taxonomy definition is malformed."""


def normalize_skill_name(name: str) -> str:
    """Normalize a skill name into a stable lookup key.

    Lower-cases and collapses whitespace, hyphens and underscores, so that
    ``"Scikit-Learn"``, ``"scikit learn"`` and ``"SCIKIT_LEARN"`` share a key.
    Characters that carry meaning in a skill name -- ``+`` in ``C++``, ``#`` in
    ``C#``, ``.`` in ``B.S.`` -- are preserved.

    Args:
        name: A skill name or alias.

    Returns:
        The normalized lookup key.

    Raises:
        TaxonomyError: If ``name`` is not a string or has no content.
    """
    if not isinstance(name, str) or not name.strip():
        raise TaxonomyError(f"skill name must be a non-empty string, got {name!r}")

    key = name.strip().lower()
    for separator in ("-", "_"):
        key = key.replace(separator, " ")
    return " ".join(key.split())


@dataclass(frozen=True, slots=True)
class SkillDefinition:
    """One recognisable skill.

    Attributes:
        name: Canonical display name, e.g. ``"Power BI"``. This is what the
            system reports, whichever alias was found in the text.
        category: Grouping such as ``"Programming"`` or ``"Finance"``.
        aliases: Alternative spellings. The canonical name is always matched and
            does not need to be repeated here.
    """

    name: str
    category: str
    aliases: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise TaxonomyError("skill name must be a non-empty string")
        if not isinstance(self.category, str) or not self.category.strip():
            raise TaxonomyError(f"category for skill {self.name!r} must be a non-empty string")

    @property
    def all_names(self) -> tuple[str, ...]:
        """The canonical name followed by every alias."""
        return (self.name, *self.aliases)


# The initial vocabulary. Aliases exist to catch real spellings found in
# resumes, including plurals, which are matched literally rather than by
# stemming so that behaviour stays predictable.
DEFAULT_SKILL_DEFINITIONS: tuple[SkillDefinition, ...] = (
    # -- Programming -------------------------------------------------------
    SkillDefinition("Python", "Programming", ("python3",)),
    SkillDefinition("Java", "Programming"),
    SkillDefinition("C++", "Programming", ("cpp",)),
    SkillDefinition("C#", "Programming", ("c sharp", "csharp")),
    SkillDefinition("C", "Programming"),
    SkillDefinition("JavaScript", "Programming", ("js", "ecmascript")),
    SkillDefinition("TypeScript", "Programming", ("ts",)),
    SkillDefinition("R", "Programming"),
    SkillDefinition("Go", "Programming", ("golang",)),
    SkillDefinition("Scala", "Programming"),
    # -- Data --------------------------------------------------------------
    SkillDefinition("SQL", "Data"),
    SkillDefinition("Excel", "Data", ("microsoft excel", "ms excel")),
    SkillDefinition("Power BI", "Data", ("powerbi",)),
    SkillDefinition("Tableau", "Data"),
    SkillDefinition("Pandas", "Data"),
    SkillDefinition("NumPy", "Data"),
    SkillDefinition("Data Analysis", "Data", ("data analytics", "analysing data")),
    SkillDefinition("Data Visualization", "Data", ("data visualisation",)),
    SkillDefinition("Statistics", "Data", ("statistical analysis",)),
    SkillDefinition("ETL", "Data"),
    SkillDefinition("Spark", "Data", ("apache spark", "pyspark")),
    # -- AI / ML -----------------------------------------------------------
    SkillDefinition("Machine Learning", "AI/ML", ("ml",)),
    SkillDefinition("Deep Learning", "AI/ML"),
    SkillDefinition("NLP", "AI/ML", ("natural language processing",)),
    SkillDefinition("Computer Vision", "AI/ML"),
    SkillDefinition("Scikit-learn", "AI/ML", ("sklearn",)),
    SkillDefinition("XGBoost", "AI/ML"),
    SkillDefinition("TensorFlow", "AI/ML"),
    SkillDefinition("PyTorch", "AI/ML", ("torch",)),
    SkillDefinition("Hugging Face", "AI/ML", ("huggingface",)),
    # -- Cloud / DevOps ----------------------------------------------------
    SkillDefinition("AWS", "Cloud/DevOps", ("amazon web services",)),
    SkillDefinition("Azure", "Cloud/DevOps", ("microsoft azure",)),
    SkillDefinition("GCP", "Cloud/DevOps", ("google cloud", "google cloud platform")),
    SkillDefinition("Docker", "Cloud/DevOps"),
    SkillDefinition("Kubernetes", "Cloud/DevOps", ("k8s",)),
    SkillDefinition("Terraform", "Cloud/DevOps"),
    SkillDefinition("CI/CD", "Cloud/DevOps", ("ci cd", "continuous integration")),
    SkillDefinition("Linux", "Cloud/DevOps"),
    # -- Backend -----------------------------------------------------------
    SkillDefinition("FastAPI", "Backend"),
    SkillDefinition("Flask", "Backend"),
    SkillDefinition("Django", "Backend"),
    SkillDefinition("REST API", "Backend", ("rest apis", "restful api", "restful apis")),
    SkillDefinition("GraphQL", "Backend"),
    SkillDefinition("Microservices", "Backend", ("microservice",)),
    SkillDefinition("PostgreSQL", "Backend", ("postgres",)),
    SkillDefinition("MySQL", "Backend"),
    SkillDefinition("MongoDB", "Backend"),
    SkillDefinition("Redis", "Backend"),
    # -- Finance -----------------------------------------------------------
    SkillDefinition("Financial Modeling", "Finance", ("financial modelling", "financial models")),
    SkillDefinition("Financial Analysis", "Finance", ("financial analyst",)),
    SkillDefinition("Forecasting", "Finance", ("forecast", "forecasts")),
    SkillDefinition("Risk Analysis", "Finance", ("risk management", "risk assessment")),
    SkillDefinition("Investment Analysis", "Finance", ("investment research",)),
    SkillDefinition("Budgeting", "Finance", ("budget planning",)),
    SkillDefinition("Valuation", "Finance"),
    SkillDefinition("Accounting", "Finance"),
)


class SkillTaxonomy:
    """An immutable collection of :class:`SkillDefinition` records.

    Args:
        definitions: The skills this taxonomy recognises.

    Raises:
        TaxonomyError: If two definitions collide on a canonical name or alias.
    """

    def __init__(self, definitions: Iterable[SkillDefinition]) -> None:
        self._definitions: tuple[SkillDefinition, ...] = tuple(definitions)
        self._by_key: dict[str, SkillDefinition] = {}

        for definition in self._definitions:
            if not isinstance(definition, SkillDefinition):
                raise TaxonomyError(
                    f"expected SkillDefinition, got {type(definition).__name__}"
                )
            for name in definition.all_names:
                key = normalize_skill_name(name)
                existing = self._by_key.get(key)
                if existing is not None and existing.name != definition.name:
                    raise TaxonomyError(
                        f"alias {name!r} is claimed by both {existing.name!r} "
                        f"and {definition.name!r}"
                    )
                self._by_key[key] = definition

    def __len__(self) -> int:
        """Number of distinct skills."""
        return len(self._definitions)

    def __iter__(self) -> Iterator[SkillDefinition]:
        """Iterate over the definitions in declaration order."""
        return iter(self._definitions)

    def __contains__(self, name: object) -> bool:
        """Whether ``name`` is a known canonical name or alias."""
        if not isinstance(name, str) or not name.strip():
            return False
        return normalize_skill_name(name) in self._by_key

    @property
    def definitions(self) -> tuple[SkillDefinition, ...]:
        """Every skill definition, in declaration order."""
        return self._definitions

    @property
    def categories(self) -> tuple[str, ...]:
        """Distinct categories, in first-appearance order."""
        seen: dict[str, None] = {}
        for definition in self._definitions:
            seen.setdefault(definition.category, None)
        return tuple(seen)

    def canonical_name(self, name: str) -> str | None:
        """Resolve any spelling of a skill to its canonical name.

        Args:
            name: A canonical name or alias, in any case.

        Returns:
            The canonical name, or ``None`` if the skill is unknown.
        """
        if not isinstance(name, str) or not name.strip():
            return None
        definition = self._by_key.get(normalize_skill_name(name))
        return definition.name if definition else None

    def get(self, name: str) -> SkillDefinition | None:
        """Return the definition for any spelling of a skill, or ``None``."""
        if not isinstance(name, str) or not name.strip():
            return None
        return self._by_key.get(normalize_skill_name(name))

    def category_of(self, name: str) -> str | None:
        """Return the category for any spelling of a skill, or ``None``."""
        definition = self.get(name)
        return definition.category if definition else None

    def by_category(self, category: str) -> tuple[SkillDefinition, ...]:
        """Every definition in ``category``, compared case-insensitively."""
        wanted = category.strip().lower()
        return tuple(d for d in self._definitions if d.category.lower() == wanted)

    def extended(self, definitions: Iterable[SkillDefinition]) -> SkillTaxonomy:
        """Return a new taxonomy with extra skills appended.

        Args:
            definitions: Additional skills.

        Returns:
            A new taxonomy; the original is left unchanged.

        Raises:
            TaxonomyError: If an added skill collides with an existing one.
        """
        return SkillTaxonomy((*self._definitions, *definitions))

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Mapping[str, Sequence[str]]]) -> SkillTaxonomy:
        """Build a taxonomy from ``{category: {skill: [aliases]}}``.

        Args:
            mapping: Nested mapping of categories to skills to aliases.

        Returns:
            The taxonomy.

        Raises:
            TaxonomyError: If the structure is not shaped as expected.
        """
        if not isinstance(mapping, Mapping):
            raise TaxonomyError(f"expected a mapping, got {type(mapping).__name__}")

        definitions: list[SkillDefinition] = []
        for category, skills in mapping.items():
            if not isinstance(skills, Mapping):
                raise TaxonomyError(
                    f"category {category!r} must map to a mapping of skills, "
                    f"got {type(skills).__name__}"
                )
            for skill, aliases in skills.items():
                if isinstance(aliases, str) or not isinstance(aliases, Sequence):
                    raise TaxonomyError(
                        f"aliases for skill {skill!r} must be a list of strings"
                    )
                definitions.append(SkillDefinition(skill, category, tuple(aliases)))

        return cls(definitions)

    @classmethod
    def from_json_file(cls, path: str | Path) -> SkillTaxonomy:
        """Load a taxonomy from a JSON file.

        Args:
            path: Path to a JSON file shaped ``{category: {skill: [aliases]}}``.

        Returns:
            The taxonomy.

        Raises:
            FileNotFoundError: If the file does not exist.
            TaxonomyError: If the file is not valid JSON or is malformed.
        """
        file_path = Path(path).expanduser().resolve()
        if not file_path.is_file():
            raise FileNotFoundError(f"taxonomy file not found: {file_path}")

        try:
            payload = json.loads(file_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise TaxonomyError(f"taxonomy file {file_path} is not valid JSON ({exc})") from exc

        return cls.from_mapping(payload)


DEFAULT_TAXONOMY = SkillTaxonomy(DEFAULT_SKILL_DEFINITIONS)
