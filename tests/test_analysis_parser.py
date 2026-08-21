"""Unit tests for app.analysis_parser.

Two groups: reading a response at all, and refusing to believe one. The second
group is the real hallucination safeguard, so it feeds deliberately fabricated
model output and asserts the claims do not survive.
"""

from __future__ import annotations

import json

import pytest

from app.analysis_parser import (
    AnalysisParseError,
    extract_json_object,
    parse_candidate_analysis,
)
from app.models import NOT_STATED, CandidateProfile, EducationEntry, Recommendation


@pytest.fixture
def profile() -> CandidateProfile:
    """A candidate with known skills, experience and education."""
    return CandidateProfile(
        candidate_id="c-1",
        candidate_name="Sarah Wilson",
        skills=("Python", "SQL", "Excel"),
        years_experience=4.0,
        education=(EducationEntry("MBA", "Finance"),),
        matched_skills=("Python", "SQL"),
        missing_skills=("Tableau",),
        required_experience=3.0,
        meets_experience_requirement=True,
    )


@pytest.fixture
def unknown_experience_profile() -> CandidateProfile:
    """A candidate whose resume states no years and no education."""
    return CandidateProfile(
        candidate_id="c-2",
        candidate_name="Nina Volkov",
        skills=("Photoshop",),
        years_experience=None,
        education=(),
        matched_skills=(),
        missing_skills=("Python", "SQL"),
        required_experience=3.0,
        meets_experience_requirement=None,
    )


def response(**overrides) -> str:
    """Build a well-formed model response, with optional field overrides."""
    payload = {
        "summary": "Sarah has relevant analytical experience.",
        "recommendation": "GOOD_MATCH",
        "matched_skills": ["Python", "SQL"],
        "skill_gaps": ["Tableau"],
        "experience_assessment": "The resume states 4 years against a requirement of 3.",
        "limitations": ["Could not verify employment dates."],
    }
    payload.update(overrides)
    return json.dumps(payload)


class TestExtractJsonObject:
    def test_plain_json(self) -> None:
        assert extract_json_object('{"a": 1}') == {"a": 1}

    def test_markdown_fenced_json(self) -> None:
        assert extract_json_object('```json\n{"a": 1}\n```') == {"a": 1}

    def test_bare_fence(self) -> None:
        assert extract_json_object('```\n{"a": 1}\n```') == {"a": 1}

    def test_prose_before_and_after(self) -> None:
        raw = 'Here is the analysis:\n{"a": 1}\nLet me know if you need more.'
        assert extract_json_object(raw) == {"a": 1}

    def test_nested_objects_survive(self) -> None:
        assert extract_json_object('{"a": {"b": [1, 2]}}') == {"a": {"b": [1, 2]}}

    @pytest.mark.parametrize("bad", ["", "   ", None, 42])
    def test_empty_response_raises(self, bad: object) -> None:
        with pytest.raises(AnalysisParseError):
            extract_json_object(bad)  # type: ignore[arg-type]

    def test_no_json_raises(self) -> None:
        with pytest.raises(AnalysisParseError, match="no JSON object"):
            extract_json_object("I am unable to help with that request.")

    def test_malformed_json_raises(self) -> None:
        with pytest.raises(AnalysisParseError, match="not valid JSON"):
            extract_json_object('{"a": 1,,,}')

    def test_json_array_raises(self) -> None:
        with pytest.raises(AnalysisParseError):
            extract_json_object("[1, 2, 3]")


class TestValidResponse:
    def test_parses_every_field(self, profile: CandidateProfile) -> None:
        analysis = parse_candidate_analysis(response(), profile)

        assert analysis.candidate_id == "c-1"
        assert analysis.candidate_name == "Sarah Wilson"
        assert analysis.recommendation is Recommendation.GOOD_MATCH
        assert analysis.matched_skills == ("Python", "SQL")
        assert analysis.skill_gaps == ("Tableau",)
        assert "4 years" in analysis.experience_assessment

    def test_a_clean_response_produces_no_warnings(self, profile: CandidateProfile) -> None:
        analysis = parse_candidate_analysis(response(), profile)
        assert analysis.warnings == ()
        assert analysis.is_grounded

    def test_model_name_is_recorded(self, profile: CandidateProfile) -> None:
        analysis = parse_candidate_analysis(response(), profile, model_name="fake/x")
        assert analysis.model_name == "fake/x"

    def test_evidence_is_attached_from_the_caller(self, profile: CandidateProfile) -> None:
        """Evidence comes from retrieval, never from the model."""
        evidence = ("passage one", "passage two")
        analysis = parse_candidate_analysis(response(), profile, evidence=evidence)
        assert analysis.evidence == evidence

    def test_model_supplied_evidence_is_ignored(self, profile: CandidateProfile) -> None:
        raw = response(evidence=["a passage the model invented"])
        analysis = parse_candidate_analysis(raw, profile, evidence=("real passage",))
        assert analysis.evidence == ("real passage",)

    @pytest.mark.parametrize("value", Recommendation.values())
    def test_every_controlled_value_is_accepted(
        self, profile: CandidateProfile, value: str
    ) -> None:
        analysis = parse_candidate_analysis(response(recommendation=value), profile)
        assert analysis.recommendation.value == value

    def test_recommendation_is_case_insensitive(self, profile: CandidateProfile) -> None:
        analysis = parse_candidate_analysis(response(recommendation="strong match"), profile)
        assert analysis.recommendation is Recommendation.STRONG_MATCH


class TestInvalidAndMissingFields:
    def test_unparseable_response_raises(self, profile: CandidateProfile) -> None:
        with pytest.raises(AnalysisParseError):
            parse_candidate_analysis("not json at all", profile)

    def test_unknown_recommendation_falls_back_safely(self, profile: CandidateProfile) -> None:
        analysis = parse_candidate_analysis(response(recommendation="DEFINITELY_HIRE"), profile)

        assert analysis.recommendation is Recommendation.INSUFFICIENT_INFORMATION
        assert any("DEFINITELY_HIRE" in warning for warning in analysis.warnings)

    def test_missing_recommendation_falls_back_safely(self, profile: CandidateProfile) -> None:
        analysis = parse_candidate_analysis(json.dumps({"summary": "text"}), profile)
        assert analysis.recommendation is Recommendation.INSUFFICIENT_INFORMATION

    def test_numeric_recommendation_is_rejected(self, profile: CandidateProfile) -> None:
        """A score would be a probability by another name."""
        analysis = parse_candidate_analysis(response(recommendation=0.91), profile)
        assert analysis.recommendation is Recommendation.INSUFFICIENT_INFORMATION

    def test_missing_summary_is_reported(self, profile: CandidateProfile) -> None:
        analysis = parse_candidate_analysis(json.dumps({"recommendation": "GOOD_MATCH"}), profile)

        assert analysis.summary == NOT_STATED
        assert any("summary" in warning for warning in analysis.warnings)

    def test_empty_object_still_produces_a_safe_analysis(
        self, profile: CandidateProfile
    ) -> None:
        analysis = parse_candidate_analysis("{}", profile)

        assert analysis.recommendation is Recommendation.INSUFFICIENT_INFORMATION
        assert analysis.matched_skills == ()
        assert analysis.candidate_id == "c-1"

    @pytest.mark.parametrize("bad", ["a string", 42, None, {"a": 1}])
    def test_non_list_skill_fields_become_empty(
        self, profile: CandidateProfile, bad: object
    ) -> None:
        analysis = parse_candidate_analysis(response(matched_skills=bad), profile)
        assert analysis.matched_skills == ()

    def test_non_string_skill_entries_are_dropped(self, profile: CandidateProfile) -> None:
        raw = response(matched_skills=["Python", 42, None, "", "SQL"])
        analysis = parse_candidate_analysis(raw, profile)
        assert analysis.matched_skills == ("Python", "SQL")

    def test_non_string_experience_becomes_not_stated(self, profile: CandidateProfile) -> None:
        analysis = parse_candidate_analysis(response(experience_assessment=4), profile)
        assert analysis.experience_assessment == NOT_STATED


class TestHallucinationSafeguards:
    """Deliberately fabricated output must not survive validation."""

    def test_a_skill_absent_from_the_resume_is_removed(
        self, profile: CandidateProfile
    ) -> None:
        """The resume never mentions AWS, so the claim must not stand."""
        raw = response(matched_skills=["Python", "SQL", "AWS"])
        analysis = parse_candidate_analysis(raw, profile)

        assert "AWS" not in analysis.matched_skills
        assert analysis.matched_skills == ("Python", "SQL")
        assert any("AWS" in warning for warning in analysis.warnings)
        assert not analysis.is_grounded

    def test_several_invented_skills_are_all_removed(
        self, profile: CandidateProfile
    ) -> None:
        raw = response(matched_skills=["AWS", "Kubernetes", "Terraform"])
        analysis = parse_candidate_analysis(raw, profile)
        assert analysis.matched_skills == ()

    def test_a_gap_the_job_never_asked_for_is_removed(
        self, profile: CandidateProfile
    ) -> None:
        raw = response(skill_gaps=["Tableau", "Fortran"])
        analysis = parse_candidate_analysis(raw, profile)

        assert analysis.skill_gaps == ("Tableau",)
        assert any("Fortran" in warning for warning in analysis.warnings)

    def test_invented_experience_is_replaced_when_none_is_stated(
        self, unknown_experience_profile: CandidateProfile
    ) -> None:
        """Unknown experience must stay unknown, not become a number."""
        raw = response(experience_assessment="The candidate has 7 years of experience.")
        analysis = parse_candidate_analysis(raw, unknown_experience_profile)

        assert analysis.experience_assessment.startswith(NOT_STATED)
        assert "7 years" not in analysis.experience_assessment
        assert any("7 years" in warning for warning in analysis.warnings)

    def test_mentioning_the_required_years_is_not_treated_as_invention(
        self, unknown_experience_profile: CandidateProfile
    ) -> None:
        """Referring to the job's own requirement is legitimate."""
        raw = response(
            summary="A designer, assessed against an analyst role.",
            matched_skills=[],
            skill_gaps=["Python", "SQL"],
            experience_assessment=f"{NOT_STATED}. The job asks for 3 years.",
        )
        analysis = parse_candidate_analysis(raw, unknown_experience_profile)

        assert analysis.warnings == ()
        assert analysis.experience_assessment.endswith("The job asks for 3 years.")

    def test_stated_experience_may_be_discussed_freely(
        self, profile: CandidateProfile
    ) -> None:
        raw = response(experience_assessment="4 years stated, above the 3 years required.")
        analysis = parse_candidate_analysis(raw, profile)

        assert analysis.experience_assessment.startswith("4 years")
        assert analysis.warnings == ()

    def test_an_invented_degree_is_flagged(
        self, unknown_experience_profile: CandidateProfile
    ) -> None:
        raw = response(summary="The candidate holds a PhD in Statistics.")
        analysis = parse_candidate_analysis(raw, unknown_experience_profile)

        assert any("PhD" in warning for warning in analysis.warnings)

    def test_a_real_degree_is_not_flagged(self, profile: CandidateProfile) -> None:
        raw = response(summary="The candidate holds an MBA in Finance.")
        analysis = parse_candidate_analysis(raw, profile)

        assert not any("MBA" in warning for warning in analysis.warnings)

    def test_an_invented_degree_in_the_experience_text_is_flagged(
        self, unknown_experience_profile: CandidateProfile
    ) -> None:
        raw = response(experience_assessment="Holds a B.S. in Computer Science.")
        analysis = parse_candidate_analysis(raw, unknown_experience_profile)
        assert any("Bachelor of Science" in warning for warning in analysis.warnings)

    def test_invented_years_in_the_summary_are_flagged(
        self, unknown_experience_profile: CandidateProfile
    ) -> None:
        raw = response(summary="A designer with 9 years of experience in branding.")
        analysis = parse_candidate_analysis(raw, unknown_experience_profile)

        assert any("number of years" in warning for warning in analysis.warnings)

    def test_corrections_are_recorded_in_limitations(
        self, profile: CandidateProfile
    ) -> None:
        raw = response(matched_skills=["AWS"])
        analysis = parse_candidate_analysis(raw, profile)

        assert any("corrected during validation" in item for item in analysis.limitations)

    def test_skill_matching_is_case_insensitive(self, profile: CandidateProfile) -> None:
        raw = response(matched_skills=["python", "SQL"])
        analysis = parse_candidate_analysis(raw, profile)

        assert analysis.matched_skills == ("Python", "SQL")
        assert analysis.warnings == ()

    def test_duplicate_claims_are_collapsed(self, profile: CandidateProfile) -> None:
        raw = response(matched_skills=["Python", "Python", "python"])
        analysis = parse_candidate_analysis(raw, profile)
        assert analysis.matched_skills == ("Python",)

    def test_a_skill_on_the_resume_but_not_required_is_still_allowed(
        self, profile: CandidateProfile
    ) -> None:
        """Excel is on the resume, so claiming it is grounded even if unmatched."""
        raw = response(matched_skills=["Python", "Excel"])
        analysis = parse_candidate_analysis(raw, profile)

        assert "Excel" in analysis.matched_skills
        assert analysis.warnings == ()
