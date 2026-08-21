"""Unit tests for app.education_extractor."""

from __future__ import annotations

import pytest

from app.education_extractor import extract_education, extract_highest_degree


class TestDegreeFormats:
    @pytest.mark.parametrize(
        "text, degree",
        [
            ("Bachelor of Science in Physics", "Bachelor of Science"),
            ("B.S. Computer Science", "Bachelor of Science"),
            ("BSc Mathematics", "Bachelor of Science"),
            ("B.Sc. Chemistry", "Bachelor of Science"),
            ("Bachelor of Arts in History", "Bachelor of Arts"),
            ("B.A. Economics", "Bachelor of Arts"),
            ("B.Tech Computer Science", "Bachelor of Technology"),
            ("Bachelor of Technology in IT", "Bachelor of Technology"),
            ("B.E. Mechanical Engineering", "Bachelor of Engineering"),
            ("B.Eng Information Systems", "Bachelor of Engineering"),
            ("B.Com Accounting", "Bachelor of Commerce"),
            ("Master of Science in Statistics", "Master of Science"),
            ("M.S. Data Science", "Master of Science"),
            ("MSc Computer Science", "Master of Science"),
            ("M.Tech Software Engineering", "Master of Technology"),
            ("MBA in Finance", "MBA"),
            ("M.B.A. Marketing", "MBA"),
            ("PhD in Statistics", "PhD"),
            ("Ph.D. Machine Learning", "PhD"),
            ("Doctorate in Physics", "PhD"),
            ("Diploma in Professional Patisserie", "Diploma"),
            ("Associate Degree in Nursing", "Associate Degree"),
        ],
    )
    def test_recognises_common_degree_spellings(self, text: str, degree: str) -> None:
        entries = extract_education(text)
        assert entries, f"no degree found in {text!r}"
        assert entries[0].degree == degree

    @pytest.mark.parametrize(
        "text, degree",
        [
            ("Bachelor's Degree in Marketing", "Bachelor's Degree"),
            ("Bachelors in Marketing", "Bachelor's Degree"),
            ("Master's Degree in Education", "Master's Degree"),
            ("Masters in Education", "Master's Degree"),
        ],
    )
    def test_recognises_generic_degree_words(self, text: str, degree: str) -> None:
        assert extract_education(text)[0].degree == degree

    def test_specific_spelling_wins_over_the_generic_one(self) -> None:
        """'Bachelor of Science' must not be reduced to 'Bachelor's Degree'."""
        assert extract_education("Bachelor of Science in Physics")[0].degree == (
            "Bachelor of Science"
        )

    def test_matching_is_case_insensitive(self) -> None:
        assert extract_education("bachelor of science in physics")[0].degree == (
            "Bachelor of Science"
        )


class TestFieldOfStudy:
    @pytest.mark.parametrize(
        "text, field",
        [
            ("B.S. Computer Science, State University (2018)", "Computer Science"),
            ("Master of Science in Computer Science, University of Toronto", "Computer Science"),
            ("MBA in Finance", "Finance"),
            ("MBA - Finance", "Finance"),
            ("MBA: Finance", "Finance"),
            ("PhD in Statistics", "Statistics"),
            ("B.Com Accounting and Finance, Aston University (2023)", "Accounting and Finance"),
            ("Diploma in Graphic Design, Prague College of Art", "Graphic Design"),
        ],
    )
    def test_captures_the_field_of_study(self, text: str, field: str) -> None:
        assert extract_education(text)[0].field == field

    def test_field_stops_before_the_institution(self) -> None:
        entry = extract_education("B.Tech Computer Science, Indian Institute of Technology")[0]
        assert entry.field == "Computer Science"

    def test_field_stops_before_a_year(self) -> None:
        entry = extract_education("Bachelor of Science Physics 2019")[0]
        assert entry.field == "Physics"

    def test_field_is_none_when_the_degree_stands_alone(self) -> None:
        assert extract_education("MBA")[0].field is None

    def test_raw_text_is_preserved_for_review(self) -> None:
        line = "B.S. Computer Science, State University (2018)"
        assert extract_education(line)[0].raw_text == line

    def test_str_renders_degree_and_field(self) -> None:
        assert str(extract_education("MBA in Finance")[0]) == "MBA - Finance"

    def test_str_renders_degree_alone(self) -> None:
        assert str(extract_education("MBA")[0]) == "MBA"


class TestMultipleEntries:
    def test_extracts_several_degrees(self) -> None:
        resume = (
            "EDUCATION\n"
            "MBA in Finance, Manchester Business School (2021)\n"
            "B.Com Accounting, University of Leeds (2018)\n"
        )
        entries = extract_education(resume)
        assert [entry.degree for entry in entries] == ["MBA", "Bachelor of Commerce"]

    def test_preserves_order_of_appearance(self) -> None:
        resume = "B.S. Physics\nPhD in Physics\n"
        assert [entry.degree for entry in extract_education(resume)] == [
            "Bachelor of Science",
            "PhD",
        ]

    def test_deduplicates_identical_entries(self) -> None:
        resume = "MBA in Finance\nMBA in Finance\n"
        assert len(extract_education(resume)) == 1

    def test_same_degree_with_different_fields_is_kept_separately(self) -> None:
        resume = "M.S. Physics\nM.S. Mathematics\n"
        assert len(extract_education(resume)) == 2


class TestMissingEducation:
    @pytest.mark.parametrize("text", ["", "   ", "\n\t", None, 42])
    def test_empty_or_invalid_input_yields_nothing(self, text: object) -> None:
        assert extract_education(text) == ()  # type: ignore[arg-type]

    def test_resume_without_education_yields_nothing(self) -> None:
        resume = "Jane Doe\nSenior Engineer\nSKILLS\nPython, SQL, Docker"
        assert extract_education(resume) == ()

    def test_ms_office_is_not_mistaken_for_a_masters_degree(self) -> None:
        """Bare 'MS' is excluded precisely because of this case."""
        assert extract_education("Proficient in MS Excel and MS Office") == ()

    def test_the_word_bachelor_party_is_still_matched(self) -> None:
        """A documented limitation: the extractor keys on the word, not context."""
        assert extract_education("Organised a bachelor party")[0].degree == "Bachelor's Degree"


class TestHighestDegree:
    def test_picks_the_most_advanced(self) -> None:
        resume = "B.Com Accounting\nMBA in Finance\n"
        highest = extract_highest_degree(resume)
        assert highest is not None
        assert highest.degree == "MBA"

    def test_phd_outranks_a_masters(self) -> None:
        resume = "M.S. Physics\nPhD in Physics\n"
        highest = extract_highest_degree(resume)
        assert highest is not None
        assert highest.degree == "PhD"

    def test_bachelors_outranks_a_diploma(self) -> None:
        resume = "Diploma in Design\nB.A. History\n"
        highest = extract_highest_degree(resume)
        assert highest is not None
        assert highest.degree == "Bachelor of Arts"

    def test_none_when_no_education_is_present(self) -> None:
        assert extract_highest_degree("No degrees listed here") is None

    def test_single_entry_is_returned(self) -> None:
        highest = extract_highest_degree("PhD in Statistics")
        assert highest is not None
        assert highest.field == "Statistics"
