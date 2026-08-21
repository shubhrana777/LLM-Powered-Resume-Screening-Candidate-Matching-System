"""Unit tests for app.rag_context."""

from __future__ import annotations

import pytest

from app.models import NOT_STATED, CandidateProfile, EducationEntry, JobRequirements
from app.rag_context import ContextIsolationError, build_rag_context
from app.retriever import RetrievedEvidence

REQUIREMENTS = JobRequirements(
    required_skills=("Python", "SQL", "Tableau"),
    minimum_experience=3.0,
    raw_text="Financial analyst with 3+ years using Python, SQL and Tableau.",
)


def evidence(candidate_id: str, index: int = 0, text: str = "A resume passage.") -> RetrievedEvidence:
    """Build one evidence item for a candidate."""
    return RetrievedEvidence(
        candidate_id=candidate_id,
        chunk_id=f"{candidate_id}#{index}",
        text=text,
        retrieval_score=0.5,
        chunk_index=index,
    )


@pytest.fixture
def profile() -> CandidateProfile:
    """A profile with everything populated."""
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
def sparse_profile() -> CandidateProfile:
    """A profile with nothing known beyond an id."""
    return CandidateProfile(
        candidate_id="c-2",
        candidate_name="Nina Volkov",
        missing_skills=("Python", "SQL", "Tableau"),
        required_experience=3.0,
    )


class TestSections:
    def test_all_three_sections_are_present(self, profile: CandidateProfile) -> None:
        text = build_rag_context(profile, REQUIREMENTS, [evidence("c-1")]).text

        assert "JOB DESCRIPTION:" in text
        assert "CANDIDATE PROFILE:" in text
        assert "RETRIEVED RESUME EVIDENCE:" in text

    def test_sections_appear_in_order(self, profile: CandidateProfile) -> None:
        text = build_rag_context(profile, REQUIREMENTS, [evidence("c-1")]).text

        assert text.index("JOB DESCRIPTION:") < text.index("CANDIDATE PROFILE:")
        assert text.index("CANDIDATE PROFILE:") < text.index("RETRIEVED RESUME EVIDENCE:")

    def test_job_description_text_is_included(self, profile: CandidateProfile) -> None:
        assert REQUIREMENTS.raw_text in build_rag_context(profile, REQUIREMENTS).text

    def test_required_skills_are_listed(self, profile: CandidateProfile) -> None:
        text = build_rag_context(profile, REQUIREMENTS).text
        assert "Required skills identified in the job description: Python, SQL, Tableau" in text

    def test_minimum_experience_is_listed(self, profile: CandidateProfile) -> None:
        assert "Minimum experience stated in the job description: 3 years" in (
            build_rag_context(profile, REQUIREMENTS).text
        )

    def test_unstated_minimum_experience_is_marked(self, profile: CandidateProfile) -> None:
        requirements = JobRequirements(required_skills=("Python",), raw_text="Python developer.")
        text = build_rag_context(profile, requirements).text
        assert f"Minimum experience stated in the job description: {NOT_STATED}" in text


class TestProfileRendering:
    def test_profile_fields_are_rendered(self, profile: CandidateProfile) -> None:
        text = build_rag_context(profile, REQUIREMENTS).text

        assert "Candidate: Sarah Wilson" in text
        assert "Matched skills: Python, SQL" in text
        assert "Missing skills: Tableau" in text
        assert "Skill coverage: 2/3 required skills" in text
        assert "Experience stated on resume: 4 years (stated on resume)" in text
        assert "Experience required by job: 3 years" in text
        assert "Meets stated experience requirement: yes" in text
        assert "Education: MBA - Finance" in text

    def test_unknown_experience_is_marked_not_stated(
        self, sparse_profile: CandidateProfile
    ) -> None:
        text = build_rag_context(sparse_profile, REQUIREMENTS).text
        assert f"Experience stated on resume: {NOT_STATED}" in text

    def test_unknown_verdict_is_marked_unknown_not_no(
        self, sparse_profile: CandidateProfile
    ) -> None:
        """Unknown must never be rendered as a failure."""
        text = build_rag_context(sparse_profile, REQUIREMENTS).text
        assert "Meets stated experience requirement: unknown" in text
        assert "Meets stated experience requirement: no" not in text

    def test_missing_education_is_marked_not_stated(
        self, sparse_profile: CandidateProfile
    ) -> None:
        assert f"Education: {NOT_STATED}" in build_rag_context(sparse_profile, REQUIREMENTS).text

    def test_empty_skill_lists_are_marked(self, sparse_profile: CandidateProfile) -> None:
        text = build_rag_context(sparse_profile, REQUIREMENTS).text
        assert "Matched skills: none identified" in text

    def test_failing_verdict_is_rendered_as_no(self) -> None:
        profile = CandidateProfile(
            candidate_id="c-3",
            years_experience=2.0,
            required_experience=3.0,
            meets_experience_requirement=False,
        )
        assert "Meets stated experience requirement: no" in (
            build_rag_context(profile, REQUIREMENTS).text
        )


class TestEvidenceRendering:
    def test_evidence_text_is_included(self, profile: CandidateProfile) -> None:
        item = evidence("c-1", text="Automated reporting with Python and SQL.")
        text = build_rag_context(profile, REQUIREMENTS, [item]).text

        assert "Automated reporting with Python and SQL." in text

    def test_evidence_is_numbered_and_attributed(self, profile: CandidateProfile) -> None:
        text = build_rag_context(profile, REQUIREMENTS, [evidence("c-1", 0)]).text

        assert "[Chunk 1]" in text
        assert "chunk_id=c-1#0" in text
        assert "similarity=0.5000" in text

    def test_several_passages_are_numbered_in_order(self, profile: CandidateProfile) -> None:
        items = [evidence("c-1", index) for index in range(3)]
        text = build_rag_context(profile, REQUIREMENTS, items).text

        assert text.index("[Chunk 1]") < text.index("[Chunk 2]") < text.index("[Chunk 3]")

    def test_absent_evidence_is_stated_explicitly(self, profile: CandidateProfile) -> None:
        """Silence would invite the model to fill the gap."""
        text = build_rag_context(profile, REQUIREMENTS, []).text
        assert "No resume passages were retrieved" in text
        assert "not stated" in text.lower()

    def test_very_long_passages_are_truncated(self, profile: CandidateProfile) -> None:
        item = evidence("c-1", text="word " * 2000)
        text = build_rag_context(profile, REQUIREMENTS, [item]).text
        assert "..." in text

    def test_evidence_chunk_ids_are_exposed(self, profile: CandidateProfile) -> None:
        context = build_rag_context(profile, REQUIREMENTS, [evidence("c-1", 0), evidence("c-1", 1)])
        assert context.evidence_chunk_ids == frozenset({"c-1#0", "c-1#1"})


class TestCandidateIsolation:
    def test_foreign_evidence_raises(self, profile: CandidateProfile) -> None:
        """Building a context from someone else's resume is a hard failure."""
        with pytest.raises(ContextIsolationError, match="c-999"):
            build_rag_context(profile, REQUIREMENTS, [evidence("c-999")])

    def test_mixed_evidence_raises(self, profile: CandidateProfile) -> None:
        with pytest.raises(ContextIsolationError):
            build_rag_context(profile, REQUIREMENTS, [evidence("c-1"), evidence("c-999")])

    def test_the_error_names_every_foreign_candidate(self, profile: CandidateProfile) -> None:
        with pytest.raises(ContextIsolationError) as exc_info:
            build_rag_context(
                profile, REQUIREMENTS, [evidence("c-aaa"), evidence("c-bbb")]
            )
        assert "c-aaa" in str(exc_info.value)
        assert "c-bbb" in str(exc_info.value)

    def test_own_evidence_is_accepted(self, profile: CandidateProfile) -> None:
        context = build_rag_context(profile, REQUIREMENTS, [evidence("c-1")])
        assert len(context.evidence) == 1

    def test_foreign_text_never_reaches_the_context(self, profile: CandidateProfile) -> None:
        secret = "CANDIDATE B CONFIDENTIAL SALARY HISTORY"
        with pytest.raises(ContextIsolationError):
            build_rag_context(profile, REQUIREMENTS, [evidence("c-999", text=secret)])


class TestContextObject:
    def test_carries_its_inputs(self, profile: CandidateProfile) -> None:
        item = evidence("c-1")
        context = build_rag_context(profile, REQUIREMENTS, [item])

        assert context.candidate_id == "c-1"
        assert context.profile is profile
        assert context.requirements is REQUIREMENTS
        assert context.evidence == (item,)

    def test_empty_job_text_raises(self, profile: CandidateProfile) -> None:
        with pytest.raises(ValueError, match="job-description text"):
            build_rag_context(profile, JobRequirements(raw_text="   "))

    def test_rendering_is_deterministic(self, profile: CandidateProfile) -> None:
        item = evidence("c-1")
        first = build_rag_context(profile, REQUIREMENTS, [item]).text
        second = build_rag_context(profile, REQUIREMENTS, [item]).text
        assert first == second
