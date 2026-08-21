"""End-to-end tests for app.rag_pipeline and the ``rag`` CLI subcommand.

Everything here runs offline: the embedder is the deterministic fake from
conftest, and the model is the offline provider from :mod:`app.llm`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.analysis_parser import AnalysisParseError
from app.candidate_analyzer import analyze_job_description
from app.llm import FakeLLMProvider, ScriptedLLMProvider
from app.main import main
from app.matching import EmptyCandidateListError
from app.models import Candidate, CandidateAnalysis, Recommendation
from app.rag_context import ContextIsolationError
from app.rag_pipeline import RagConfig, RagPipeline
from app.retriever import UnknownCandidateError

from .conftest import FakeEmbedder

ANALYST_JOB = (
    "Financial analyst wanted. Requirements: Python for reporting automation, "
    "SQL for the data warehouse, Excel modelling, and Power BI dashboards. "
    "3+ years of experience required."
)


@pytest.fixture
def pipeline(fake_embedder: FakeEmbedder, analyst_candidates: list[Candidate]) -> RagPipeline:
    """A pipeline with the analyst candidates indexed."""
    instance = RagPipeline(
        embedder=fake_embedder,
        llm=FakeLLMProvider(),
        config=RagConfig(chunk_size=30, chunk_overlap=8, top_k=3),
    )
    instance.index_candidates(analyst_candidates)
    return instance


class TestIndexing:
    def test_indexing_reports_the_chunk_count(self, pipeline: RagPipeline) -> None:
        assert len(pipeline.retriever) == sum(
            pipeline.retriever.chunk_count(cid) for cid in pipeline.candidate_ids
        )

    def test_every_candidate_is_indexed(self, pipeline: RagPipeline) -> None:
        assert set(pipeline.candidate_ids) == {"c-strong", "c-partial", "c-poor"}

    def test_long_resumes_produce_several_chunks(
        self, fake_embedder: FakeEmbedder, long_resume_candidate: Candidate
    ) -> None:
        instance = RagPipeline(embedder=fake_embedder, llm=FakeLLMProvider())
        assert instance.index_candidates([long_resume_candidate]) > 1

    def test_indexing_nothing_raises(self, fake_embedder: FakeEmbedder) -> None:
        instance = RagPipeline(embedder=fake_embedder, llm=FakeLLMProvider())
        with pytest.raises(EmptyCandidateListError):
            instance.index_candidates([])

    def test_reindexing_replaces_the_previous_set(
        self, pipeline: RagPipeline, analyst_candidates: list[Candidate]
    ) -> None:
        pipeline.index_candidates(analyst_candidates[:1])
        assert pipeline.candidate_ids == ("c-strong",)


class TestContextBuilding:
    def test_context_contains_all_three_sections(self, pipeline: RagPipeline) -> None:
        requirements = analyze_job_description(ANALYST_JOB)
        context = pipeline.build_context("c-strong", requirements)

        assert "JOB DESCRIPTION:" in context.text
        assert "CANDIDATE PROFILE:" in context.text
        assert "RETRIEVED RESUME EVIDENCE:" in context.text

    def test_job_description_is_included(self, pipeline: RagPipeline) -> None:
        requirements = analyze_job_description(ANALYST_JOB)
        assert "Financial analyst wanted" in pipeline.build_context("c-strong", requirements).text

    def test_candidate_profile_is_included(self, pipeline: RagPipeline) -> None:
        requirements = analyze_job_description(ANALYST_JOB)
        context = pipeline.build_context("c-strong", requirements)

        assert context.profile.candidate_id == "c-strong"
        assert "Sarah Wilson" in context.text

    def test_evidence_is_retrieved(self, pipeline: RagPipeline) -> None:
        requirements = analyze_job_description(ANALYST_JOB)
        assert pipeline.build_context("c-strong", requirements).evidence

    def test_evidence_is_capped_by_top_k(self, pipeline: RagPipeline) -> None:
        requirements = analyze_job_description(ANALYST_JOB)
        assert len(pipeline.build_context("c-strong", requirements).evidence) <= 3

    def test_unknown_candidate_raises(self, pipeline: RagPipeline) -> None:
        requirements = analyze_job_description(ANALYST_JOB)
        with pytest.raises(UnknownCandidateError):
            pipeline.build_context("nobody", requirements)


class TestAnalysis:
    def test_returns_a_candidate_analysis(self, pipeline: RagPipeline) -> None:
        assert isinstance(pipeline.analyze_candidate("c-strong", ANALYST_JOB), CandidateAnalysis)

    def test_identity_is_preserved(self, pipeline: RagPipeline) -> None:
        analysis = pipeline.analyze_candidate("c-strong", ANALYST_JOB)
        assert analysis.candidate_id == "c-strong"
        assert analysis.candidate_name == "Sarah Wilson"

    def test_recommendation_is_from_the_controlled_vocabulary(
        self, pipeline: RagPipeline
    ) -> None:
        analysis = pipeline.analyze_candidate("c-strong", ANALYST_JOB)
        assert analysis.recommendation.value in Recommendation.values()

    def test_evidence_is_attached(self, pipeline: RagPipeline) -> None:
        analysis = pipeline.analyze_candidate("c-strong", ANALYST_JOB)

        assert analysis.evidence
        for item in analysis.evidence:
            assert item.candidate_id == "c-strong"
            assert item.chunk_id
            assert item.text
            assert -1.0 <= item.retrieval_score <= 1.0

    def test_model_name_is_recorded(self, pipeline: RagPipeline) -> None:
        assert pipeline.analyze_candidate("c-strong", ANALYST_JOB).model_name.startswith("fake/")

    def test_strong_candidate_outranks_poor_one(self, pipeline: RagPipeline) -> None:
        strong = pipeline.analyze_candidate("c-strong", ANALYST_JOB)
        poor = pipeline.analyze_candidate("c-poor", ANALYST_JOB)
        assert len(strong.matched_skills) > len(poor.matched_skills)

    def test_analysis_is_deterministic(self, pipeline: RagPipeline) -> None:
        first = pipeline.analyze_candidate("c-strong", ANALYST_JOB)
        second = pipeline.analyze_candidate("c-strong", ANALYST_JOB)
        assert first.summary == second.summary
        assert first.recommendation is second.recommendation

    def test_accepts_prebuilt_requirements(self, pipeline: RagPipeline) -> None:
        requirements = analyze_job_description(ANALYST_JOB)
        assert pipeline.analyze_candidate("c-strong", requirements).candidate_id == "c-strong"

    def test_analyze_all_covers_every_candidate(self, pipeline: RagPipeline) -> None:
        _requirements, analyses = pipeline.analyze_all(ANALYST_JOB)
        assert {a.candidate_id for a in analyses} == {"c-strong", "c-partial", "c-poor"}

    def test_analyze_all_honours_top_k(self, pipeline: RagPipeline) -> None:
        _requirements, analyses = pipeline.analyze_all(ANALYST_JOB, top_k_candidates=2)
        assert len(analyses) == 2

    def test_analyze_all_returns_the_requirements(self, pipeline: RagPipeline) -> None:
        requirements, _analyses = pipeline.analyze_all(ANALYST_JOB)
        assert requirements.minimum_experience == 3.0

    def test_analyze_all_before_indexing_raises(self, fake_embedder: FakeEmbedder) -> None:
        instance = RagPipeline(embedder=fake_embedder, llm=FakeLLMProvider())
        with pytest.raises(EmptyCandidateListError):
            instance.analyze_all(ANALYST_JOB)

    def test_one_model_call_per_candidate(
        self, fake_embedder: FakeEmbedder, analyst_candidates: list[Candidate]
    ) -> None:
        provider = FakeLLMProvider()
        instance = RagPipeline(embedder=fake_embedder, llm=provider)
        instance.index_candidates(analyst_candidates)
        instance.analyze_all(ANALYST_JOB)

        assert len(provider.calls) == len(analyst_candidates)

    def test_the_system_prompt_is_sent(self, pipeline: RagPipeline) -> None:
        pipeline.analyze_candidate("c-strong", ANALYST_JOB)
        _prompt, system = pipeline.llm.calls[-1]
        assert system is not None
        assert "Do not invent candidate information." in system


class TestCandidateIsolation:
    """No candidate's resume text may reach another candidate's prompt."""

    @pytest.fixture
    def isolated(
        self, isolation_embedder: FakeEmbedder, isolation_candidates: list[Candidate]
    ) -> RagPipeline:
        instance = RagPipeline(
            embedder=isolation_embedder,
            llm=FakeLLMProvider(),
            config=RagConfig(chunk_size=12, chunk_overlap=2, top_k=5),
        )
        instance.index_candidates(isolation_candidates)
        return instance

    def test_context_holds_only_the_subject_candidate(self, isolated: RagPipeline) -> None:
        requirements = analyze_job_description("Python and SQL and Docker and Kubernetes.")
        for candidate_id in ("cand-a", "cand-b"):
            context = isolated.build_context(candidate_id, requirements)
            assert {item.candidate_id for item in context.evidence} == {candidate_id}

    def test_prompt_never_contains_the_other_resume(self, isolated: RagPipeline) -> None:
        """The strongest form of the check: inspect the literal prompt text."""
        isolated.analyze_candidate("cand-a", "Python and SQL and Docker and Kubernetes.")
        prompt, _system = isolated.llm.calls[-1]

        assert "Alice Alpha" in prompt
        assert "Bob Beta" not in prompt

        # Split first, lowercase after: the section markers are upper-case.
        evidence_section = prompt.split("RETRIEVED RESUME EVIDENCE:")[1].lower()
        assert "python" in evidence_section
        assert "kubernetes" not in evidence_section
        assert "docker" not in evidence_section

    def test_analysis_evidence_is_single_candidate(self, isolated: RagPipeline) -> None:
        for candidate_id in ("cand-a", "cand-b"):
            analysis = isolated.analyze_candidate(
                candidate_id, "Python and SQL and Docker and Kubernetes."
            )
            assert {item.candidate_id for item in analysis.evidence} == {candidate_id}

    def test_analyzing_everyone_keeps_evidence_separate(self, isolated: RagPipeline) -> None:
        _requirements, analyses = isolated.analyze_all(
            "Python and SQL and Docker and Kubernetes."
        )
        for analysis in analyses:
            assert {item.candidate_id for item in analysis.evidence} == {analysis.candidate_id}

    def test_every_prompt_mentions_exactly_one_candidate(self, isolated: RagPipeline) -> None:
        isolated.analyze_all("Python and SQL and Docker and Kubernetes.")

        for prompt, _system in isolated.llm.calls:
            names = [name for name in ("Alice Alpha", "Bob Beta") if name in prompt]
            assert len(names) == 1


class TestHallucinationSafetyEndToEnd:
    """A misbehaving model must not produce an ungrounded analysis."""

    def _pipeline(self, embedder: FakeEmbedder, candidates, response: str) -> RagPipeline:
        instance = RagPipeline(
            embedder=embedder,
            llm=ScriptedLLMProvider([response]),
            config=RagConfig(chunk_size=30, chunk_overlap=8, top_k=3),
        )
        instance.index_candidates(candidates)
        return instance

    def test_absent_skill_is_not_claimed(
        self, fake_embedder: FakeEmbedder, analyst_candidates: list[Candidate]
    ) -> None:
        raw = json.dumps(
            {
                "summary": "Strong cloud engineer.",
                "recommendation": "STRONG_MATCH",
                "matched_skills": ["AWS", "Kubernetes", "Python"],
                "skill_gaps": [],
                "experience_assessment": "Plenty.",
                "limitations": [],
            }
        )
        pipeline = self._pipeline(fake_embedder, analyst_candidates, raw)
        analysis = pipeline.analyze_candidate("c-strong", ANALYST_JOB)

        assert "AWS" not in analysis.matched_skills
        assert "Kubernetes" not in analysis.matched_skills
        assert not analysis.is_grounded

    def test_unknown_experience_remains_unknown(
        self, fake_embedder: FakeEmbedder, analyst_candidates: list[Candidate]
    ) -> None:
        raw = json.dumps(
            {
                "summary": "A designer.",
                "recommendation": "WEAK_MATCH",
                "matched_skills": [],
                "skill_gaps": ["Python"],
                "experience_assessment": "The candidate has 12 years of experience.",
                "limitations": [],
            }
        )
        pipeline = self._pipeline(fake_embedder, analyst_candidates, raw)
        analysis = pipeline.analyze_candidate("c-poor", ANALYST_JOB)

        assert "12 years" not in analysis.experience_assessment
        assert analysis.experience_assessment.startswith("Not stated")

    def test_unsupported_education_is_flagged(
        self, fake_embedder: FakeEmbedder, analyst_candidates: list[Candidate]
    ) -> None:
        raw = json.dumps(
            {
                "summary": "Holds a PhD in Econometrics from a top school.",
                "recommendation": "GOOD_MATCH",
                "matched_skills": [],
                "skill_gaps": [],
                "experience_assessment": "Not stated",
                "limitations": [],
            }
        )
        pipeline = self._pipeline(fake_embedder, analyst_candidates, raw)
        analysis = pipeline.analyze_candidate("c-poor", ANALYST_JOB)

        assert any("PhD" in warning for warning in analysis.warnings)

    def test_invalid_recommendation_becomes_insufficient_information(
        self, fake_embedder: FakeEmbedder, analyst_candidates: list[Candidate]
    ) -> None:
        raw = json.dumps({"summary": "text", "recommendation": "HIRE_IMMEDIATELY"})
        pipeline = self._pipeline(fake_embedder, analyst_candidates, raw)

        analysis = pipeline.analyze_candidate("c-strong", ANALYST_JOB)
        assert analysis.recommendation is Recommendation.INSUFFICIENT_INFORMATION

    def test_unparseable_response_raises(
        self, fake_embedder: FakeEmbedder, analyst_candidates: list[Candidate]
    ) -> None:
        pipeline = self._pipeline(fake_embedder, analyst_candidates, "I cannot help with that.")
        with pytest.raises(AnalysisParseError):
            pipeline.analyze_candidate("c-strong", ANALYST_JOB)

    def test_fenced_json_is_still_accepted(
        self, fake_embedder: FakeEmbedder, analyst_candidates: list[Candidate]
    ) -> None:
        raw = '```json\n{"summary": "ok", "recommendation": "GOOD_MATCH"}\n```'
        pipeline = self._pipeline(fake_embedder, analyst_candidates, raw)

        assert pipeline.analyze_candidate("c-strong", ANALYST_JOB).summary == "ok"


class TestRagCLI:
    """The ``rag`` subcommand, driven by the offline providers."""

    @pytest.fixture(autouse=True)
    def _offline(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("app.main.get_default_embedder", lambda *_a: FakeEmbedder())
        monkeypatch.setattr("app.main.get_llm_provider", lambda **_k: FakeLLMProvider())

    @pytest.fixture
    def job_file(self, tmp_path: Path) -> Path:
        path = tmp_path / "job.txt"
        path.write_text(ANALYST_JOB, encoding="utf-8")
        return path

    def test_reports_requirements_and_candidates(
        self, analyst_resume_dir: Path, job_file: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["rag", "-r", str(analyst_resume_dir), "-j", str(job_file)]) == 0
        output = capsys.readouterr().out

        assert "JOB REQUIREMENTS" in output
        assert "Recommendation:" in output
        assert "Sarah Wilson" in output

    def test_shows_evidence_by_default(
        self, analyst_resume_dir: Path, job_file: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        main(["rag", "-r", str(analyst_resume_dir), "-j", str(job_file)])
        output = capsys.readouterr().out

        assert "Evidence (verbatim resume passages given to the model):" in output
        assert "similarity=" in output

    def test_evidence_can_be_hidden(
        self, analyst_resume_dir: Path, job_file: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        main(["rag", "-r", str(analyst_resume_dir), "-j", str(job_file), "--hide-evidence"])
        assert "Evidence (verbatim" not in capsys.readouterr().out

    def test_single_candidate_can_be_selected(
        self, analyst_resume_dir: Path, job_file: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        main([
            "rag", "-r", str(analyst_resume_dir), "-j", str(job_file),
            "--candidate", "sarah_wilson",
        ])
        output = capsys.readouterr().out

        assert "Sarah Wilson" in output
        assert output.count("Recommendation:") == 1

    def test_top_k_limits_the_report(
        self, analyst_resume_dir: Path, job_file: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        main(["rag", "-r", str(analyst_resume_dir), "-j", str(job_file), "-k", "1"])
        assert capsys.readouterr().out.count("Recommendation:") == 1

    def test_report_states_the_grounding_caveat(
        self, analyst_resume_dir: Path, job_file: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        main(["rag", "-r", str(analyst_resume_dir), "-j", str(job_file)])
        output = capsys.readouterr().out

        assert "reduces hallucination rather than eliminating it" in output
        assert "Read the evidence" in output

    def test_recommendation_is_never_called_a_probability(
        self, analyst_resume_dir: Path, job_file: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        main(["rag", "-r", str(analyst_resume_dir), "-j", str(job_file)])
        assert "probability" not in capsys.readouterr().out.lower()

    def test_unknown_candidate_exits_cleanly(
        self, analyst_resume_dir: Path, job_file: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        exit_code = main([
            "rag", "-r", str(analyst_resume_dir), "-j", str(job_file), "--candidate", "nobody",
        ])
        captured = capsys.readouterr()

        assert exit_code == 1
        assert "Error: " in captured.err
        assert "Traceback" not in captured.err

    def test_empty_job_text_exits_cleanly(
        self, analyst_resume_dir: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        exit_code = main(["rag", "-r", str(analyst_resume_dir), "-t", "   "])
        captured = capsys.readouterr()

        assert exit_code == 1
        assert "Traceback" not in captured.err

    def test_bad_chunk_settings_exit_cleanly(
        self, analyst_resume_dir: Path, job_file: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        exit_code = main([
            "rag", "-r", str(analyst_resume_dir), "-j", str(job_file),
            "--chunk-size", "10", "--chunk-overlap", "10",
        ])
        captured = capsys.readouterr()

        assert exit_code == 1
        assert "never advance" in captured.err
        assert "Traceback" not in captured.err

    def test_earlier_subcommands_still_work(
        self, analyst_resume_dir: Path, job_file: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["match", "-r", str(analyst_resume_dir), "-j", str(job_file)]) == 0
        assert "Rank" in capsys.readouterr().out

        assert main(["analyze", "-r", str(analyst_resume_dir), "-j", str(job_file)]) == 0
        assert "JOB REQUIREMENTS" in capsys.readouterr().out
