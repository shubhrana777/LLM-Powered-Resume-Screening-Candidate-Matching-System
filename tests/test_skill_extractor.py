"""Unit tests for app.skill_extractor.

The false-positive cases matter most: naive substring matching would report SQL
for "PostgreSQL" and R for every word containing the letter, which on a
screening tool means fabricated skills.
"""

from __future__ import annotations

import pytest

from app.skill_extractor import (
    DEFAULT_EXTRACTOR,
    SkillExtractor,
    compare_skills,
    extract_required_skills,
    extract_skills,
)
from app.skill_taxonomy import DEFAULT_TAXONOMY, SkillDefinition, SkillTaxonomy


class TestBasicExtraction:
    def test_extracts_the_brief_example(self) -> None:
        text = (
            "Experienced financial analyst using Python, SQL,\n"
            "Excel and Power BI for forecasting and reporting."
        )
        skills = extract_skills(text)

        for expected in ("Python", "SQL", "Excel", "Power BI", "Forecasting"):
            assert expected in skills

    def test_extracts_a_single_skill(self) -> None:
        assert extract_skills("I write Python every day.") == ("Python",)

    def test_returns_canonical_names_not_the_matched_spelling(self) -> None:
        assert extract_skills("Experienced with sklearn and k8s") == (
            "Scikit-learn",
            "Kubernetes",
        )

    def test_deduplicates_repeated_skills(self) -> None:
        text = "Python, Python, and more Python. Python again."
        assert extract_skills(text) == ("Python",)

    def test_ignores_unknown_skills(self) -> None:
        assert extract_skills("Expert in Underwater Basket Weaving") == ()

    def test_output_is_deterministic(self) -> None:
        text = "Tableau, Python, Docker, SQL"
        assert extract_skills(text) == extract_skills(text)

    def test_ordering_follows_the_taxonomy_not_the_text(self) -> None:
        """Same skills in a different order produce the same output."""
        assert extract_skills("SQL then Python") == extract_skills("Python then SQL")

    @pytest.mark.parametrize("empty", ["", "   ", "\n\t", None, 42])
    def test_empty_or_invalid_input_yields_no_skills(self, empty: object) -> None:
        assert extract_skills(empty) == ()  # type: ignore[arg-type]

    def test_text_with_no_known_skills_yields_nothing(self) -> None:
        assert extract_skills("I enjoy long walks and gardening.") == ()


class TestCaseInsensitivity:
    @pytest.mark.parametrize("spelling", ["Python", "python", "PYTHON", "PyThOn"])
    def test_multi_character_skills_match_in_any_case(self, spelling: str) -> None:
        assert extract_skills(f"Skilled in {spelling}") == ("Python",)

    @pytest.mark.parametrize("spelling", ["Power BI", "power bi", "POWER BI", "PowerBI"])
    def test_multi_word_skills_match_in_any_case(self, spelling: str) -> None:
        assert "Power BI" in extract_skills(f"Dashboards in {spelling}")


class TestMultiWordAndSeparators:
    @pytest.mark.parametrize(
        "text, expected",
        [
            ("Built with Power BI", "Power BI"),
            ("Built with power-bi", "Power BI"),
            ("Built with power_bi", "Power BI"),
            ("Built with powerbi", "Power BI"),
            ("Applied machine learning", "Machine Learning"),
            ("Applied Machine-Learning", "Machine Learning"),
            ("Used scikit-learn", "Scikit-learn"),
            ("Used scikit learn", "Scikit-learn"),
            ("Did data analysis", "Data Analysis"),
            ("Did data analytics", "Data Analysis"),
        ],
    )
    def test_separator_variants_all_match(self, text: str, expected: str) -> None:
        assert expected in extract_skills(text)

    def test_multi_word_skill_matches_across_a_line_break(self) -> None:
        """PDF extraction frequently splits a phrase across lines."""
        assert "Power BI" in extract_skills("Reporting in Power\nBI dashboards")

    def test_multi_word_skill_is_preferred_over_its_parts(self) -> None:
        skills = extract_skills("Experienced in machine learning")
        assert "Machine Learning" in skills


class TestPunctuation:
    @pytest.mark.parametrize(
        "text",
        [
            "Python.",
            "Python,",
            "Python;",
            "Python!",
            "(Python)",
            "[Python]",
            "Python/Django",
            "- Python",
            "Skills: Python",
            "Python's ecosystem",
            "**Python**",
        ],
    )
    def test_surrounding_punctuation_does_not_block_a_match(self, text: str) -> None:
        assert "Python" in extract_skills(text)

    def test_comma_separated_list_extracts_every_entry(self) -> None:
        skills = extract_skills("Skills: Python, SQL, Docker, Tableau, AWS")
        assert set(skills) == {"Python", "SQL", "Docker", "Tableau", "AWS"}


class TestFalseSubstringMatches:
    """Naive `alias in text` matching would fail every case here."""

    @pytest.mark.parametrize(
        "text",
        ["MySQL database", "PostgreSQL cluster", "sqlalchemy ORM", "NoSQL stores"],
    )
    def test_sql_is_not_matched_inside_a_longer_word(self, text: str) -> None:
        assert "SQL" not in extract_skills(text)

    def test_sql_is_still_matched_when_standalone(self) -> None:
        assert "SQL" in extract_skills("Strong SQL skills")
        assert "SQL" in extract_skills("Wrote SQL, then reviewed it.")

    @pytest.mark.parametrize(
        "text",
        [
            "Great rapport with clients",
            "Regular reporting duties",
            "R&D department",
            "Ready to relocate",
            "Recruitment experience",
        ],
    )
    def test_r_is_not_matched_inside_words(self, text: str) -> None:
        assert "R" not in extract_skills(text)

    @pytest.mark.parametrize("text", ["Proficient in R.", "R, Python and SQL", "Uses R for stats"])
    def test_r_is_matched_when_standalone_and_capitalised(self, text: str) -> None:
        assert "R" in extract_skills(text)

    def test_lowercase_standalone_letter_is_not_a_skill(self) -> None:
        """A list marker like 'c)' must not register the C language."""
        assert extract_skills("c) delivered the project on time") == ()
        assert extract_skills("r) reviewed the results") == ()

    @pytest.mark.parametrize(
        "text",
        ["Managed the rest of the team", "Restructured the department", "Rested the servers"],
    )
    def test_rest_api_is_not_matched_by_the_word_rest(self, text: str) -> None:
        assert "REST API" not in extract_skills(text)

    def test_rest_api_is_matched_when_actually_present(self) -> None:
        assert "REST API" in extract_skills("Designed REST APIs")
        assert "REST API" in extract_skills("Built a RESTful API")

    def test_go_is_not_matched_inside_words(self) -> None:
        assert "Go" not in extract_skills("Going forward we algo trade")

    def test_java_is_not_matched_by_javascript(self) -> None:
        skills = extract_skills("Wrote JavaScript for the frontend")
        assert "JavaScript" in skills
        assert "Java" not in skills


class TestSpecialCharacterSkills:
    def test_cpp_is_extracted(self) -> None:
        assert "C++" in extract_skills("Systems programming in C++")

    def test_c_is_not_matched_by_cpp(self) -> None:
        assert extract_skills("Systems programming in C++") == ("C++",)

    def test_c_and_cpp_together(self) -> None:
        skills = extract_skills("Built with C++ and C")
        assert "C++" in skills
        assert "C" in skills

    def test_csharp_is_extracted_and_is_not_c(self) -> None:
        skills = extract_skills("Wrote C# services")
        assert "C#" in skills
        assert "C" not in skills

    def test_c_is_not_matched_inside_an_initialism(self) -> None:
        """'C.S.' is a subject abbreviation, not the C language."""
        assert "C" not in extract_skills("Studied C.S. at university")

    def test_cpp_alias_matches(self) -> None:
        assert "C++" in extract_skills("Experienced in cpp development")

    def test_ci_cd_with_a_slash_is_extracted(self) -> None:
        assert "CI/CD" in extract_skills("Owned the CI/CD pipeline")


class TestExtractionDetail:
    def test_find_mentions_records_the_matched_substring(self) -> None:
        mentions = DEFAULT_EXTRACTOR.find_mentions("Used PYTHON and sklearn")
        by_skill = {mention.skill: mention for mention in mentions}

        assert by_skill["Python"].matched_text == "PYTHON"
        assert by_skill["Scikit-learn"].matched_text == "sklearn"

    def test_find_mentions_records_offsets_that_point_at_the_text(self) -> None:
        text = "We use Python daily"
        mention = DEFAULT_EXTRACTOR.find_mentions(text)[0]
        assert text[mention.start : mention.end] == "Python"

    def test_repeated_skill_produces_several_mentions_but_one_skill(self) -> None:
        text = "Python and more Python"
        assert len(DEFAULT_EXTRACTOR.find_mentions(text)) == 2
        assert DEFAULT_EXTRACTOR.extract(text) == ("Python",)

    def test_mentions_are_ordered_by_position(self) -> None:
        mentions = DEFAULT_EXTRACTOR.find_mentions("Docker first, then Python, then SQL")
        assert [m.start for m in mentions] == sorted(m.start for m in mentions)

    def test_mentions_carry_the_category(self) -> None:
        mention = DEFAULT_EXTRACTOR.find_mentions("Uses Python")[0]
        assert mention.category == "Programming"

    def test_extract_by_category_groups_results(self) -> None:
        grouped = DEFAULT_EXTRACTOR.extract_by_category("Python, SQL and Docker")
        assert grouped["Programming"] == ("Python",)
        assert grouped["Data"] == ("SQL",)
        assert grouped["Cloud/DevOps"] == ("Docker",)

    def test_extract_by_category_omits_empty_categories(self) -> None:
        assert set(DEFAULT_EXTRACTOR.extract_by_category("Python")) == {"Programming"}


class TestCustomTaxonomy:
    def test_extractor_honours_a_custom_taxonomy(self) -> None:
        taxonomy = SkillTaxonomy([SkillDefinition("Rust", "Programming", ("rustlang",))])
        extractor = SkillExtractor(taxonomy)

        assert extractor.extract("Systems work in rustlang") == ("Rust",)
        assert extractor.extract("Python developer") == ()

    def test_extended_taxonomy_finds_both_old_and_new_skills(self) -> None:
        taxonomy = DEFAULT_TAXONOMY.extended([SkillDefinition("Rust", "Programming")])
        extractor = SkillExtractor(taxonomy)

        assert set(extractor.extract("Python and Rust")) == {"Python", "Rust"}

    def test_extract_required_skills_matches_extract_skills(self) -> None:
        text = "Looking for Python and SQL experience"
        assert extract_required_skills(text) == extract_skills(text)


class TestCompareSkills:
    def test_brief_example(self) -> None:
        comparison = compare_skills(
            ["Python", "SQL", "Excel", "Power BI", "Tableau"],
            ["Python", "SQL", "Power BI"],
        )
        assert comparison.matched == ("Python", "SQL", "Power BI")
        assert comparison.missing == ("Excel", "Tableau")

    def test_all_skills_matched(self) -> None:
        comparison = compare_skills(["Python", "SQL"], ["Python", "SQL", "Docker"])
        assert comparison.matched == ("Python", "SQL")
        assert comparison.missing == ()

    def test_no_skills_matched(self) -> None:
        comparison = compare_skills(["Python", "SQL"], ["Tableau"])
        assert comparison.matched == ()
        assert comparison.missing == ("Python", "SQL")

    def test_additional_skills_are_reported(self) -> None:
        comparison = compare_skills(["Python"], ["Python", "Docker", "AWS"])
        assert comparison.additional == ("Docker", "AWS")

    def test_empty_requirements(self) -> None:
        comparison = compare_skills([], ["Python", "SQL"])
        assert comparison.matched == ()
        assert comparison.missing == ()
        assert comparison.additional == ("Python", "SQL")

    def test_empty_candidate_skills(self) -> None:
        comparison = compare_skills(["Python", "SQL"], [])
        assert comparison.matched == ()
        assert comparison.missing == ("Python", "SQL")

    def test_both_empty(self) -> None:
        comparison = compare_skills([], [])
        assert comparison.matched == ()
        assert comparison.missing == ()
        assert comparison.additional == ()

    def test_ordering_follows_the_required_list(self) -> None:
        comparison = compare_skills(["Tableau", "Python", "SQL"], ["SQL", "Python", "Tableau"])
        assert comparison.matched == ("Tableau", "Python", "SQL")

    def test_comparison_is_case_insensitive(self) -> None:
        comparison = compare_skills(["python", "SQL"], ["PYTHON", "sql"])
        assert comparison.missing == ()
        assert len(comparison.matched) == 2

    def test_aliases_match_canonical_names(self) -> None:
        """A job asking for 'Kubernetes' is satisfied by a resume saying 'k8s'."""
        comparison = compare_skills(["Kubernetes"], ["k8s"])
        assert comparison.matched == ("Kubernetes",)
        assert comparison.missing == ()

    def test_duplicates_are_collapsed(self) -> None:
        comparison = compare_skills(["Python", "Python"], ["Python", "python"])
        assert comparison.matched == ("Python",)
        assert comparison.required_count == 1

    def test_unknown_skills_are_still_compared(self) -> None:
        comparison = compare_skills(["Basket Weaving"], ["Basket Weaving"])
        assert comparison.matched == ("Basket Weaving",)

    def test_blank_entries_are_ignored(self) -> None:
        comparison = compare_skills(["Python", "", "   "], ["Python"])
        assert comparison.matched == ("Python",)
        assert comparison.missing == ()

    def test_a_bare_string_is_rejected(self) -> None:
        """'Python' as a string would otherwise be read as six characters."""
        with pytest.raises(TypeError):
            compare_skills("Python", ["Python"])  # type: ignore[arg-type]


class TestSkillComparisonCoverage:
    def test_coverage_is_a_plain_ratio(self) -> None:
        comparison = compare_skills(["Python", "SQL", "Excel", "Power BI"], ["Python", "SQL"])
        assert comparison.required_count == 4
        assert comparison.coverage == 0.5

    def test_full_coverage(self) -> None:
        assert compare_skills(["Python"], ["Python"]).coverage == 1.0

    def test_zero_coverage(self) -> None:
        assert compare_skills(["Python"], ["SQL"]).coverage == 0.0

    def test_coverage_is_none_when_nothing_was_required(self) -> None:
        """No requirements means there is no fraction to take."""
        assert compare_skills([], ["Python"]).coverage is None
