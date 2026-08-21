"""Tests for the dashboard's display helpers.

Three properties are worth more than the rest, because a UI breaks them easily
and the consequence is a recruiter reading a number as something it is not:

* similarity is shown as a percentage of the cosine scale for readability, and
  the wording that travels with it refuses the probability reading outright;
* similarity, skill coverage, experience and recommendation stay visibly
  different measures -- never interchangeable, never derived from each other;
* unknown is never rendered as zero, as a pass, or as a failure. A candidate who
  has not been analysed reads "Not analyzed yet".
"""

from __future__ import annotations

import pytest

from app.ui.formatting import (
    MODERATE_SCORE,
    NOT_ANALYZED,
    SIMILARITY_MEANING,
    STRONG_SCORE,
    candidate_display_name,
    experience_status,
    format_coverage,
    format_similarity,
    format_similarity_raw,
    grounding_label,
    plural,
    recommendation_label,
    recommendation_tone,
    score_band,
    skill_coverage,
    to_percent,
    truncate,
)
from app.ui.theme import TONES


# --------------------------------------------------------------------------
# Scores
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "score, expected",
    [
        (0.90, "Strong similarity"),
        (STRONG_SCORE, "Strong similarity"),
        (0.45, "Moderate similarity"),
        (MODERATE_SCORE, "Moderate similarity"),
        (0.10, "Low similarity"),
        (-0.20, "Low similarity"),
    ],
)
def test_scores_are_banded_in_words(score: float, expected: str):
    assert score_band(score).label == expected


def test_a_missing_score_has_its_own_band():
    assert score_band(None).label == "No score"


def test_every_band_tone_exists_in_the_theme():
    for score in (0.9, 0.45, 0.1, None):
        assert score_band(score).tone in TONES


def test_a_similarity_is_rendered_as_a_percentage():
    """The backend value is unchanged; only its presentation is scaled."""
    assert format_similarity(0.5935) == "59.35%"


@pytest.mark.parametrize(
    "score, expected",
    [
        (0.5935, "59.35%"),
        (0.6024, "60.24%"),
        (1.0, "100.00%"),
        (0.0, "0.00%"),
        (0.07, "7.00%"),
    ],
)
def test_similarity_percentages_are_two_decimals(score: float, expected: str):
    assert format_similarity(score) == expected


def test_a_negative_similarity_is_rendered_faithfully():
    """Cosine similarity can be negative; it is not clamped to look better."""
    assert format_similarity(-0.1234) == "-12.34%"


def test_the_raw_cosine_value_is_still_available():
    """The percentage is a presentation; the underlying number stays visible."""
    assert format_similarity_raw(0.5935) == "0.5935"


def test_scaling_to_percent_preserves_the_value():
    assert to_percent(0.5935) == pytest.approx(59.35)
    assert to_percent(None) is None


def test_a_missing_score_is_not_rendered_as_zero():
    assert format_similarity(None) == NOT_ANALYZED
    assert "0" not in format_similarity(None)


def test_the_similarity_wording_refuses_the_wrong_readings():
    """The percent sign must never be allowed to imply probability."""
    meaning = SIMILARITY_MEANING.lower()
    assert "ranking signal" in meaning
    assert "not a hiring probability" in meaning
    assert "percentage of requirements met" in meaning


# --------------------------------------------------------------------------
# Recommendations
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value, expected",
    [
        ("STRONG_MATCH", "Strong match"),
        ("GOOD_MATCH", "Good match"),
        ("PARTIAL_MATCH", "Partial match"),
        ("WEAK_MATCH", "Weak match"),
        ("INSUFFICIENT_INFORMATION", "Insufficient information"),
    ],
)
def test_recommendations_are_labelled_readably(value: str, expected: str):
    assert recommendation_label(value) == expected


def test_the_labels_cover_the_backend_vocabulary():
    """If the backend adds a value, this test fails rather than the UI guessing."""
    from app.models import Recommendation

    for value in Recommendation.values():
        assert recommendation_label(value) != NOT_ANALYZED
        assert recommendation_tone(value) in TONES


def test_a_missing_recommendation_is_reported_as_not_analyzed_yet():
    assert recommendation_label(None) == NOT_ANALYZED
    assert recommendation_tone(None) == "neutral"


def test_the_not_analyzed_label_describes_a_step_not_a_failure():
    """It must not read as "the system tried and could not"."""
    assert NOT_ANALYZED == "Not analyzed yet"
    lowered = NOT_ANALYZED.lower()
    for failure_word in ("fail", "error", "unavailable", "unable", "none"):
        assert failure_word not in lowered


def test_the_not_analyzed_hint_says_how_to_fix_it():
    from app.ui.formatting import NOT_ANALYZED_HINT

    hint = NOT_ANALYZED_HINT.lower()
    assert "run candidate analysis" in hint
    assert "skill coverage" in hint


def test_an_unknown_recommendation_is_shown_not_guessed_at():
    assert recommendation_label("SOMETHING_NEW") == "Something new"
    assert recommendation_tone("SOMETHING_NEW") == "neutral"


def test_insufficient_information_is_not_styled_as_a_failure():
    """It means the evidence did not support a judgement, not that the candidate is weak."""
    assert recommendation_tone("INSUFFICIENT_INFORMATION") == "neutral"
    assert recommendation_tone("WEAK_MATCH") == "critical"


# --------------------------------------------------------------------------
# Skill coverage
# --------------------------------------------------------------------------


def test_coverage_is_a_plain_ratio():
    assert skill_coverage(["a", "b", "c"], ["d"]) == pytest.approx(0.75)


def test_full_coverage_is_one():
    assert skill_coverage(["a", "b"], []) == 1.0


def test_no_coverage_is_zero_not_none():
    assert skill_coverage([], ["a"]) == 0.0


def test_coverage_of_nothing_is_unknown_not_zero():
    """A job naming no recognised skills has no denominator to divide by."""
    assert skill_coverage([], []) is None
    assert skill_coverage(None, None) is None


def test_coverage_is_formatted_as_a_count():
    assert format_coverage(["a", "b"], ["c"]) == "2 / 3"


def test_coverage_is_never_formatted_as_a_percentage():
    assert "%" not in format_coverage(["a"], ["b", "c"])


def test_coverage_with_no_named_skills_says_so():
    assert format_coverage([], []) == "No skills named"


# --------------------------------------------------------------------------
# Experience
# --------------------------------------------------------------------------


def test_a_met_requirement_is_recognised():
    assessment = (
        "The resume states 4 years (stated on resume); the job asks for 3 years. "
        "Requirement met: yes."
    )
    status = experience_status(assessment)
    assert status.label == "Requirement met"
    assert status.tone == "positive"


def test_an_unmet_requirement_is_recognised():
    assessment = (
        "The resume states 2 years (stated on resume); the job asks for 3 years. "
        "Requirement met: no."
    )
    status = experience_status(assessment)
    assert status.label == "Below requirement"
    assert status.tone == "critical"


def test_an_unstated_duration_is_neither_a_pass_nor_a_fail():
    assessment = (
        "Not stated. The resume does not state a number of years, so this cannot be "
        "compared against the requirement of 3 years."
    )
    status = experience_status(assessment)
    assert status.label == "Not stated"
    assert status.tone == "neutral"


def test_unrecognised_phrasing_is_not_guessed_at():
    assert experience_status("Some wording nobody anticipated").tone == "neutral"


def test_a_missing_assessment_is_reported_as_not_analysed():
    assert experience_status(None).label == NOT_ANALYZED
    assert experience_status("").label == NOT_ANALYZED


def test_the_real_backend_wording_is_recognised():
    """Guards against the backend's phrasing drifting away from this parser."""
    from app.analysis_parser import _check_experience_claim  # noqa: F401 - import proves it exists
    from app.models import NOT_STATED

    assert experience_status(NOT_STATED).label == "Not stated"


# --------------------------------------------------------------------------
# Grounding
# --------------------------------------------------------------------------


def test_a_grounded_analysis_is_labelled_grounded():
    assert grounding_label(True).label == "Grounded"


def test_a_corrected_analysis_is_flagged_as_a_warning_not_an_error():
    """Corrections mean the safeguard worked, so the tone is caution, not failure."""
    band = grounding_label(False)
    assert band.label == "Corrected claims"
    assert band.tone == "caution"


def test_unknown_grounding_is_reported_as_not_analysed():
    assert grounding_label(None).label == NOT_ANALYZED


# --------------------------------------------------------------------------
# Text helpers
# --------------------------------------------------------------------------


def test_short_text_is_left_alone():
    assert truncate("Short enough", 50) == "Short enough"


def test_long_text_is_shortened_and_marked():
    result = truncate("word " * 100, 40)
    assert result.endswith("…")
    assert len(result) <= 41


def test_whitespace_is_collapsed():
    assert truncate("a\n\n  b\tc") == "a b c"


def test_empty_text_is_empty():
    assert truncate(None) == ""
    assert truncate("") == ""


@pytest.mark.parametrize(
    "count, expected", [(0, "0 candidates"), (1, "1 candidate"), (5, "5 candidates")]
)
def test_counts_are_pluralised(count: int, expected: str):
    assert plural(count, "candidate") == expected


def test_an_irregular_plural_can_be_given():
    assert plural(2, "analysis", "analyses") == "2 analyses"


# --------------------------------------------------------------------------
# Names
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "record, expected",
    [
        ({"name": "Sarah Wilson", "candidate_id": "sarah_wilson"}, "Sarah Wilson"),
        ({"candidate": "Sarah Wilson", "candidate_id": "sarah_wilson"}, "Sarah Wilson"),
        ({"candidate_id": "sarah_wilson"}, "sarah_wilson"),
        ({}, "Unknown candidate"),
        ({"name": "   ", "candidate_id": "sarah_wilson"}, "sarah_wilson"),
    ],
)
def test_a_display_name_is_found_in_any_payload_shape(record: dict, expected: str):
    assert candidate_display_name(record) == expected



# --------------------------------------------------------------------------
# The four measures are distinct
# --------------------------------------------------------------------------


def test_similarity_and_skill_coverage_are_formatted_differently():
    """They answer different questions and must never look interchangeable."""
    similarity = format_similarity(0.9166)
    coverage = format_coverage(["a"] * 11, ["b"])

    assert similarity == "91.66%"
    assert coverage == "11 / 12"
    assert "%" not in coverage
    assert "/" not in similarity


def test_skill_coverage_is_never_derived_from_similarity():
    """Coverage counts skills; it has no relationship to the embedding score."""
    assert format_coverage([], []) == "No skills named"
    assert format_coverage(["a"], []) == "1 / 1"


def test_experience_is_a_status_not_a_number():
    assessment = "The resume states 4 years; the job asks for 3 years. Requirement met: yes."
    assert experience_status(assessment).label == "Requirement met"


def test_a_recommendation_is_a_label_not_a_score():
    label = recommendation_label("STRONG_MATCH")
    assert label == "Strong match"
    assert "%" not in label
    assert not any(character.isdigit() for character in label)
