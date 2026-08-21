"""Unit tests for app.candidate_analyzer and the Phase 3 models."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.candidate_analyzer import (
    analyze_candidates,
    analyze_job_description,
    build_candidate_profile,
    compare_experience,
)
from app.matching import EmptyJobDescriptionError
from app.models import Candidate, CandidateProfile, EducationEntry, JobRequirements, MatchResult


class TestAnalyzeJobDescription:
    def test_brief_example(self) -> None:
        text = (
            "Looking for a financial analyst with 3+ years of experience "
            "using Python, SQL, Excel and Power BI."
        )
        requirements = analyze_job_description(text)

        for skill in ("Python", "SQL", "Excel", "Power BI"):
            assert skill in requirements.required_skills
        assert requirements.minimum_experience == 3.0

    def test_extracts_required_skills(self, analyst_job: str) -> None:
        requirements = analyze_job_description(analyst_job)
        assert {"Python", "SQL", "Excel", "Power BI", "Tableau"}.issubset(
            set(requirements.required_skills)
        )

    def test_extracts_minimum_experience(self, analyst_job: str) -> None:
        assert analyze_job_description(analyst_job).minimum_experience == 3.0

    def test_minimum_experience_is_none_when_unstated(self) -> None:
        requirements = analyze_job_description("We need a Python developer.")
        assert requirements.minimum_experience is None
        assert requirements.required_skills == ("Python",)

    def test_no_recognised_skills_yields_an_empty_tuple(self) -> None:
        requirements = analyze_job_description("We need a friendly, motivated person.")
        assert requirements.required_skills == ()

    def test_nothing_is_inferred_beyond_the_text(self) -> None:
        """A Django posting must not imply Python unless Python is written."""
        requirements = analyze_job_description("We need a Django developer.")
        assert requirements.required_skills == ("Django",)

    def test_raw_text_is_preserved(self) -> None:
        assert analyze_job_description("  Python developer  ").raw_text == "Python developer"

    @pytest.mark.parametrize("bad", ["", "   ", "\n\t"])
    def test_empty_job_description_raises(self, bad: str) -> None:
        with pytest.raises(EmptyJobDescriptionError):
            analyze_job_description(bad)


class TestCompareExperience:
    @pytest.mark.parametrize(
        "required, candidate, expected",
        [
            (3.0, 4.0, True),
            (3.0, 3.0, True),
            (3.0, 2.0, False),
            (3.0, None, None),
            (None, 4.0, None),
            (None, None, None),
            (0.5, 1.0, True),
        ],
    )
    def test_comparison_matrix(
        self, required: float | None, candidate: float | None, expected: bool | None
    ) -> None:
        assert compare_experience(required, candidate) is expected

    def test_unknown_is_never_a_failure(self) -> None:
        """The brief is explicit: unknown must not be treated as not meeting."""
        assert compare_experience(3.0, None) is not False

    def test_unknown_is_never_a_pass(self) -> None:
        assert compare_experience(3.0, None) is not True


class TestBuildCandidateProfile:
    def test_extracts_skills_experience_and_education(self, strong_candidate: Candidate) -> None:
        profile = build_candidate_profile(strong_candidate)

        assert "Python" in profile.skills
        assert profile.years_experience == 4.0
        assert profile.education[0].degree == "MBA"

    def test_preserves_candidate_identity(self, strong_candidate: Candidate) -> None:
        profile = build_candidate_profile(strong_candidate)
        assert profile.candidate_id == "c-strong"
        assert profile.candidate_name == "Sarah Wilson"
        assert profile.display_name == "Sarah Wilson"

    def test_matched_and_missing_skills_against_requirements(
        self, strong_candidate: Candidate
    ) -> None:
        requirements = JobRequirements(required_skills=("Python", "SQL", "Tableau"))
        profile = build_candidate_profile(strong_candidate, requirements)

        assert "Python" in profile.matched_skills
        assert "SQL" in profile.matched_skills
        assert "Tableau" in profile.missing_skills

    def test_partial_candidate_falls_short_on_experience(
        self, partial_candidate: Candidate
    ) -> None:
        requirements = JobRequirements(required_skills=("Excel",), minimum_experience=3.0)
        profile = build_candidate_profile(partial_candidate, requirements)

        assert profile.years_experience == 2.0
        assert profile.required_experience == 3.0
        assert profile.meets_experience_requirement is False

    def test_poor_candidate_has_unknown_experience(self, poor_candidate: Candidate) -> None:
        requirements = JobRequirements(required_skills=("Python",), minimum_experience=3.0)
        profile = build_candidate_profile(poor_candidate, requirements)

        assert profile.years_experience is None
        assert profile.meets_experience_requirement is None
        assert profile.missing_skills == ("Python",)

    def test_semantic_score_and_rank_are_preserved(self, strong_candidate: Candidate) -> None:
        result = MatchResult("c-strong", "Sarah Wilson", 0.9123, 1)
        profile = build_candidate_profile(strong_candidate, match_result=result)

        assert profile.semantic_match_score == 0.9123
        assert profile.rank == 1

    def test_semantic_fields_are_none_without_a_match_result(
        self, strong_candidate: Candidate
    ) -> None:
        profile = build_candidate_profile(strong_candidate)
        assert profile.semantic_match_score is None
        assert profile.rank is None

    def test_works_without_requirements(self, strong_candidate: Candidate) -> None:
        profile = build_candidate_profile(strong_candidate)
        assert profile.skills
        assert profile.matched_skills == ()
        assert profile.missing_skills == ()
        assert profile.required_experience is None

    def test_source_path_is_carried_through(self, tmp_path: Path) -> None:
        candidate = Candidate("c-1", "Python developer", "Dev", tmp_path / "cv.pdf")
        assert build_candidate_profile(candidate).source_path == tmp_path / "cv.pdf"

    def test_additional_skills_are_reported(self, strong_candidate: Candidate) -> None:
        requirements = JobRequirements(required_skills=("Python",))
        profile = build_candidate_profile(strong_candidate, requirements)
        assert "SQL" in profile.additional_skills

    def test_skill_comparison_property_round_trips(self, strong_candidate: Candidate) -> None:
        requirements = JobRequirements(required_skills=("Python", "Tableau"))
        profile = build_candidate_profile(strong_candidate, requirements)
        comparison = profile.skill_comparison

        assert comparison.matched == profile.matched_skills
        assert comparison.missing == profile.missing_skills
        assert comparison.required_count == 2

    def test_non_candidate_input_raises(self) -> None:
        with pytest.raises(TypeError):
            build_candidate_profile({"candidate_id": "c-1"})  # type: ignore[arg-type]

    def test_is_deterministic(self, strong_candidate: Candidate) -> None:
        requirements = JobRequirements(required_skills=("Python", "Tableau"))
        first = build_candidate_profile(strong_candidate, requirements)
        second = build_candidate_profile(strong_candidate, requirements)
        assert first == second


class TestAnalyzeCandidates:
    def test_builds_a_profile_per_candidate(self, analyst_candidates: list[Candidate]) -> None:
        profiles = analyze_candidates(analyst_candidates)
        assert len(profiles) == 3
        assert all(isinstance(profile, CandidateProfile) for profile in profiles)

    def test_preserves_input_order_without_ranking(
        self, analyst_candidates: list[Candidate]
    ) -> None:
        profiles = analyze_candidates(analyst_candidates)
        assert [p.candidate_id for p in profiles] == [c.candidate_id for c in analyst_candidates]

    def test_orders_by_rank_when_match_results_are_supplied(
        self, analyst_candidates: list[Candidate]
    ) -> None:
        results = [
            MatchResult("c-poor", "Nina Volkov", 0.10, 1),
            MatchResult("c-strong", "Sarah Wilson", 0.90, 2),
            MatchResult("c-partial", "James Patel", 0.50, 3),
        ]
        profiles = analyze_candidates(analyst_candidates, match_results=results)
        assert [p.candidate_id for p in profiles] == ["c-poor", "c-strong", "c-partial"]

    def test_match_results_are_paired_by_id_not_position(
        self, analyst_candidates: list[Candidate]
    ) -> None:
        results = [MatchResult("c-partial", "James Patel", 0.42, 1)]
        profiles = analyze_candidates(analyst_candidates, match_results=results)
        by_id = {profile.candidate_id: profile for profile in profiles}

        assert by_id["c-partial"].semantic_match_score == 0.42
        assert by_id["c-strong"].semantic_match_score is None

    def test_unranked_candidates_sort_last(self, analyst_candidates: list[Candidate]) -> None:
        results = [MatchResult("c-partial", "James Patel", 0.42, 1)]
        profiles = analyze_candidates(analyst_candidates, match_results=results)
        assert profiles[0].candidate_id == "c-partial"
        assert all(p.rank is None for p in profiles[1:])

    def test_requirements_apply_to_every_profile(
        self, analyst_candidates: list[Candidate]
    ) -> None:
        requirements = JobRequirements(required_skills=("Python",), minimum_experience=3.0)
        profiles = analyze_candidates(analyst_candidates, requirements=requirements)
        assert all(p.required_experience == 3.0 for p in profiles)

    def test_empty_candidate_list_yields_no_profiles(self) -> None:
        assert analyze_candidates([]) == []


class TestPhase3Models:
    def test_education_entry_str_with_and_without_field(self) -> None:
        assert str(EducationEntry("MBA", "Finance")) == "MBA - Finance"
        assert str(EducationEntry("MBA")) == "MBA"

    def test_job_requirements_defaults(self) -> None:
        requirements = JobRequirements()
        assert requirements.required_skills == ()
        assert requirements.minimum_experience is None

    def test_candidate_profile_display_name_falls_back_to_id(self) -> None:
        assert CandidateProfile("c-1").display_name == "c-1"

    def test_candidate_profile_is_immutable(self) -> None:
        profile = CandidateProfile("c-1")
        with pytest.raises(Exception):
            profile.candidate_id = "c-2"  # type: ignore[misc]

    def test_candidate_profile_defaults_are_conservative(self) -> None:
        """Unknown fields default to None/empty, never to a fabricated value."""
        profile = CandidateProfile("c-1")
        assert profile.years_experience is None
        assert profile.semantic_match_score is None
        assert profile.meets_experience_requirement is None
        assert profile.skills == ()
        assert profile.education == ()
