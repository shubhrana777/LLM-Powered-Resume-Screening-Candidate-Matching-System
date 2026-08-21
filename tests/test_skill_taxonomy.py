"""Unit tests for app.skill_taxonomy."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.skill_taxonomy import (
    DEFAULT_TAXONOMY,
    SkillDefinition,
    SkillTaxonomy,
    TaxonomyError,
    normalize_skill_name,
)


class TestNormalizeSkillName:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("Python", "python"),
            ("  PYTHON  ", "python"),
            ("Power BI", "power bi"),
            ("Scikit-Learn", "scikit learn"),
            ("SCIKIT_LEARN", "scikit learn"),
            ("scikit   learn", "scikit learn"),
            ("C++", "c++"),
            ("C#", "c#"),
            ("B.S.", "b.s."),
        ],
    )
    def test_normalizes_to_a_stable_key(self, raw: str, expected: str) -> None:
        assert normalize_skill_name(raw) == expected

    def test_hyphen_and_space_forms_share_a_key(self) -> None:
        assert normalize_skill_name("scikit-learn") == normalize_skill_name("Scikit Learn")

    @pytest.mark.parametrize("bad", ["", "   ", None, 42])
    def test_invalid_input_raises(self, bad: object) -> None:
        with pytest.raises(TaxonomyError):
            normalize_skill_name(bad)  # type: ignore[arg-type]


class TestSkillDefinition:
    def test_all_names_starts_with_the_canonical_name(self) -> None:
        definition = SkillDefinition("Python", "Programming", ("python3",))
        assert definition.all_names == ("Python", "python3")

    def test_aliases_are_optional(self) -> None:
        assert SkillDefinition("Java", "Programming").all_names == ("Java",)

    @pytest.mark.parametrize("bad", ["", "   "])
    def test_blank_name_raises(self, bad: str) -> None:
        with pytest.raises(TaxonomyError):
            SkillDefinition(bad, "Programming")

    def test_blank_category_raises(self) -> None:
        with pytest.raises(TaxonomyError):
            SkillDefinition("Python", "")

    def test_is_immutable(self) -> None:
        definition = SkillDefinition("Python", "Programming")
        with pytest.raises(Exception):
            definition.name = "Java"  # type: ignore[misc]


class TestDefaultTaxonomy:
    @pytest.mark.parametrize(
        "skill",
        [
            "Python", "Java", "C++", "JavaScript", "TypeScript",
            "SQL", "Excel", "Power BI", "Tableau", "Pandas", "NumPy",
            "Data Analysis", "Statistics",
            "Machine Learning", "Deep Learning", "NLP", "Scikit-learn",
            "XGBoost", "TensorFlow", "PyTorch",
            "AWS", "Azure", "GCP", "Docker", "Kubernetes",
            "FastAPI", "Flask", "Django", "REST API",
            "Financial Modeling", "Financial Analysis", "Forecasting",
            "Risk Analysis", "Investment Analysis",
        ],
    )
    def test_contains_every_required_skill(self, skill: str) -> None:
        """Every skill named in the Phase 3 brief must be present."""
        assert skill in DEFAULT_TAXONOMY

    def test_covers_the_expected_categories(self) -> None:
        expected = {"Programming", "Data", "AI/ML", "Cloud/DevOps", "Backend", "Finance"}
        assert expected.issubset(set(DEFAULT_TAXONOMY.categories))

    def test_lookup_is_case_insensitive(self) -> None:
        assert DEFAULT_TAXONOMY.canonical_name("python") == "Python"
        assert DEFAULT_TAXONOMY.canonical_name("POWER BI") == "Power BI"

    def test_alias_resolves_to_the_canonical_name(self) -> None:
        assert DEFAULT_TAXONOMY.canonical_name("sklearn") == "Scikit-learn"
        assert DEFAULT_TAXONOMY.canonical_name("k8s") == "Kubernetes"
        assert DEFAULT_TAXONOMY.canonical_name("natural language processing") == "NLP"

    def test_unknown_skill_resolves_to_none(self) -> None:
        assert DEFAULT_TAXONOMY.canonical_name("Underwater Basket Weaving") is None

    @pytest.mark.parametrize("bad", ["", "   ", None, 42])
    def test_unusable_lookup_input_is_none_not_an_error(self, bad: object) -> None:
        assert DEFAULT_TAXONOMY.canonical_name(bad) is None  # type: ignore[arg-type]
        assert bad not in DEFAULT_TAXONOMY

    def test_category_lookup(self) -> None:
        assert DEFAULT_TAXONOMY.category_of("Python") == "Programming"
        assert DEFAULT_TAXONOMY.category_of("Forecasting") == "Finance"
        assert DEFAULT_TAXONOMY.category_of("nonexistent") is None

    def test_by_category_returns_that_category_only(self) -> None:
        finance = DEFAULT_TAXONOMY.by_category("Finance")
        assert finance
        assert all(definition.category == "Finance" for definition in finance)

    def test_by_category_is_case_insensitive(self) -> None:
        assert DEFAULT_TAXONOMY.by_category("finance") == DEFAULT_TAXONOMY.by_category("Finance")

    def test_is_iterable_and_sized(self) -> None:
        assert len(DEFAULT_TAXONOMY) == len(list(DEFAULT_TAXONOMY))
        assert len(DEFAULT_TAXONOMY) > 40

    def test_get_returns_the_definition(self) -> None:
        definition = DEFAULT_TAXONOMY.get("powerbi")
        assert definition is not None
        assert definition.name == "Power BI"


class TestExtensibility:
    def test_extended_adds_skills_without_mutating_the_original(self) -> None:
        original_size = len(DEFAULT_TAXONOMY)
        extended = DEFAULT_TAXONOMY.extended([SkillDefinition("Rust", "Programming", ("rustlang",))])

        assert extended.canonical_name("rustlang") == "Rust"
        assert len(extended) == original_size + 1
        assert len(DEFAULT_TAXONOMY) == original_size
        assert "Rust" not in DEFAULT_TAXONOMY

    def test_from_mapping_builds_a_taxonomy(self) -> None:
        taxonomy = SkillTaxonomy.from_mapping(
            {"Programming": {"Rust": ["rustlang"]}, "Data": {"DuckDB": []}}
        )
        assert taxonomy.canonical_name("rustlang") == "Rust"
        assert taxonomy.category_of("DuckDB") == "Data"
        assert len(taxonomy) == 2

    def test_from_json_file_round_trip(self, tmp_path: Path) -> None:
        path = tmp_path / "skills.json"
        path.write_text(
            json.dumps({"Programming": {"Rust": ["rustlang"]}}), encoding="utf-8"
        )

        taxonomy = SkillTaxonomy.from_json_file(path)
        assert taxonomy.canonical_name("Rust") == "Rust"

    def test_missing_json_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            SkillTaxonomy.from_json_file(tmp_path / "nope.json")

    def test_invalid_json_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "broken.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(TaxonomyError):
            SkillTaxonomy.from_json_file(path)

    def test_malformed_mapping_raises(self) -> None:
        with pytest.raises(TaxonomyError):
            SkillTaxonomy.from_mapping({"Programming": ["not", "a", "mapping"]})  # type: ignore[arg-type]

    def test_string_aliases_instead_of_a_list_raise(self) -> None:
        with pytest.raises(TaxonomyError):
            SkillTaxonomy.from_mapping({"Programming": {"Rust": "rustlang"}})  # type: ignore[dict-item]

    def test_conflicting_alias_raises(self) -> None:
        with pytest.raises(TaxonomyError, match="claimed by both"):
            SkillTaxonomy(
                [
                    SkillDefinition("Python", "Programming", ("py",)),
                    SkillDefinition("Pythonic", "Programming", ("py",)),
                ]
            )

    def test_non_definition_entry_raises(self) -> None:
        with pytest.raises(TaxonomyError):
            SkillTaxonomy(["Python"])  # type: ignore[list-item]
