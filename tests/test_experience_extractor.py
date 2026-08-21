"""Unit tests for app.experience_extractor.

The negative cases carry the weight: inventing years of experience a resume
never stated would be a fabrication about a real person, so "unknown" must stay
unknown.
"""

from __future__ import annotations

import pytest

from app.experience_extractor import (
    MAX_PLAUSIBLE_YEARS,
    extract_minimum_experience,
    extract_years_of_experience,
    find_experience_mentions,
)


class TestIntegerYears:
    @pytest.mark.parametrize(
        "text, expected",
        [
            ("4 years of experience", 4.0),
            ("4 years experience", 4.0),
            ("I have 6 years of experience in finance", 6.0),
            ("10 years of experience", 10.0),
            ("1 year of experience", 1.0),
            ("5 yrs of experience", 5.0),
            ("5 yr experience", 5.0),
        ],
    )
    def test_extracts_whole_years(self, text: str, expected: float) -> None:
        assert extract_years_of_experience(text) == expected


class TestDecimalYears:
    @pytest.mark.parametrize(
        "text, expected",
        [
            ("2.5 years experience", 2.5),
            ("2.5 years of professional experience", 2.5),
            ("1.5 years of experience", 1.5),
        ],
    )
    def test_extracts_fractional_years(self, text: str, expected: float) -> None:
        assert extract_years_of_experience(text) == expected


class TestPlusAndRanges:
    @pytest.mark.parametrize(
        "text, expected",
        [
            ("3+ years of experience", 3.0),
            ("3 + years of experience", 3.0),
            ("3+ years experience", 3.0),
            ("5+ years of professional experience", 5.0),
        ],
    )
    def test_plus_sign_takes_the_stated_number(self, text: str, expected: float) -> None:
        assert extract_years_of_experience(text) == expected

    def test_range_takes_the_lower_bound(self) -> None:
        assert extract_years_of_experience("3-5 years of experience") == 3.0


class TestQualifiedPhrasings:
    @pytest.mark.parametrize(
        "text, expected",
        [
            ("over 7 years of hands-on experience", 7.0),
            ("8 years of relevant experience", 8.0),
            ("4 years of industry experience", 4.0),
            ("Experience: 5 years", 5.0),
            ("6 years of experience in financial analysis", 6.0),
        ],
    )
    def test_extracts_through_qualifiers(self, text: str, expected: float) -> None:
        assert extract_years_of_experience(text) == expected

    def test_case_insensitive(self) -> None:
        assert extract_years_of_experience("4 YEARS OF EXPERIENCE") == 4.0


class TestNoFalseInference:
    """Nothing may be inferred that the text does not state."""

    @pytest.mark.parametrize(
        "text",
        [
            "Graduated in 2015",
            "B.S. Computer Science, State University (2018)",
            "Class of 2012",
            "Worked at Acme 2019-2025",
            "Acme Corp - Backend Engineer (2020-2024)",
            "Senior Software Engineer",
            "Lead Architect with deep expertise",
            "Veteran of the industry",
        ],
    )
    def test_dates_and_seniority_never_produce_a_number(self, text: str) -> None:
        assert extract_years_of_experience(text) is None

    def test_years_without_the_word_experience_is_not_counted(self) -> None:
        """Conservative by design: 'years building X' is not an explicit claim."""
        assert extract_years_of_experience("8 years building Python services") is None

    def test_years_of_education_is_not_experience(self) -> None:
        assert extract_years_of_experience("4 years of undergraduate study") is None

    @pytest.mark.parametrize("text", ["", "   ", "\n", None, 42])
    def test_empty_or_invalid_input_is_none(self, text: object) -> None:
        assert extract_years_of_experience(text) is None  # type: ignore[arg-type]

    def test_resume_with_no_experience_statement_is_none(self) -> None:
        resume = (
            "Nina Volkov\nGraphic Designer\n"
            "Led brand identity projects.\nSKILLS\nIllustrator, Photoshop"
        )
        assert extract_years_of_experience(resume) is None

    def test_does_not_bridge_two_sentences(self) -> None:
        """'5 years' and 'experience' in separate sentences are unrelated."""
        assert extract_years_of_experience("The project ran 5 years. I gained experience.") is None

    def test_implausibly_large_values_are_rejected(self) -> None:
        assert extract_years_of_experience("100 years of experience") is None
        assert extract_years_of_experience("2015 years of experience") is None

    def test_boundary_of_plausibility(self) -> None:
        assert extract_years_of_experience(f"{int(MAX_PLAUSIBLE_YEARS)} years of experience") == 60.0

    def test_zero_years_is_not_reported(self) -> None:
        assert extract_years_of_experience("0 years of experience") is None


class TestMultipleMentions:
    def test_candidate_extraction_takes_the_largest(self) -> None:
        """The overall figure, not a per-technology one."""
        text = "8 years of experience overall, including 3 years of experience with Kubernetes."
        assert extract_years_of_experience(text) == 8.0

    def test_job_extraction_takes_the_smallest(self) -> None:
        """The minimum bar a candidate must clear."""
        text = "3+ years of experience required. 5+ years of experience preferred."
        assert extract_minimum_experience(text) == 3.0

    def test_both_read_the_same_single_statement_identically(self) -> None:
        text = "4 years of experience"
        assert extract_years_of_experience(text) == extract_minimum_experience(text) == 4.0

    def test_minimum_is_none_when_nothing_is_stated(self) -> None:
        assert extract_minimum_experience("We are hiring an analyst.") is None


class TestMentions:
    def test_mentions_record_the_source_text(self) -> None:
        mentions = find_experience_mentions("I have 4 years of experience in finance")
        assert len(mentions) == 1
        assert mentions[0].years == 4.0
        assert "4 years of experience" in mentions[0].matched_text

    def test_mentions_point_at_the_right_offset(self) -> None:
        text = "Summary: 6 years of experience"
        mention = find_experience_mentions(text)[0]
        assert text[mention.start :].startswith("6 years")

    def test_several_mentions_are_all_returned_in_order(self) -> None:
        text = "8 years of experience overall. 3 years of experience with Docker."
        mentions = find_experience_mentions(text)
        assert [m.years for m in mentions] == [8.0, 3.0]

    def test_no_mentions_for_plain_text(self) -> None:
        assert find_experience_mentions("A resume with no numbers") == ()

    def test_overlapping_patterns_report_once_per_position(self) -> None:
        mentions = find_experience_mentions("Experience: 5 years of experience")
        assert len({m.start for m in mentions}) == len(mentions)
