"""End-to-end Phase 3 tests: PDF -> text -> skills/experience/education -> profile.

Exercises the Phase 1 parser, the Phase 2 matcher and the Phase 3 extractors
together, through both the library API and the ``analyze`` subcommand.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.candidate_analyzer import (
    analyze_candidates,
    analyze_candidates_for_job,
    analyze_job_description,
)
from app.main import main
from app.matching import load_candidates_from_directory
from app.resume_parser import extract_text_from_pdf
from app.skill_extractor import extract_skills

from .conftest import FakeEmbedder


class TestPhase3Pipeline:
    """PDF -> text -> skills/experience/education -> CandidateProfile."""

    def test_pdf_to_candidate_profile(self, analyst_resume_dir: Path, analyst_job: str) -> None:
        loaded = load_candidates_from_directory(analyst_resume_dir)
        requirements = analyze_job_description(analyst_job)
        profiles = analyze_candidates(loaded.candidates, requirements=requirements)

        strong = {profile.candidate_id: profile for profile in profiles}["sarah_wilson"]

        assert "Python" in strong.skills
        assert strong.years_experience == 4.0
        assert strong.education[0].degree == "MBA"
        assert strong.meets_experience_requirement is True

    def test_strong_partial_and_poor_candidates_are_distinguished(
        self, analyst_resume_dir: Path, analyst_job: str
    ) -> None:
        loaded = load_candidates_from_directory(analyst_resume_dir)
        requirements = analyze_job_description(analyst_job)
        by_id = {
            profile.candidate_id: profile
            for profile in analyze_candidates(loaded.candidates, requirements=requirements)
        }

        strong = by_id["sarah_wilson"].skill_comparison
        partial = by_id["james_patel"].skill_comparison
        poor = by_id["nina_volkov"].skill_comparison

        assert strong.coverage > partial.coverage > poor.coverage
        assert poor.matched == ()

    def test_experience_verdicts_across_the_three_candidates(
        self, analyst_resume_dir: Path, analyst_job: str
    ) -> None:
        loaded = load_candidates_from_directory(analyst_resume_dir)
        requirements = analyze_job_description(analyst_job)
        by_id = {
            profile.candidate_id: profile
            for profile in analyze_candidates(loaded.candidates, requirements=requirements)
        }

        assert by_id["sarah_wilson"].meets_experience_requirement is True
        assert by_id["james_patel"].meets_experience_requirement is False
        assert by_id["nina_volkov"].meets_experience_requirement is None

    def test_full_pipeline_with_semantic_ranking(
        self, analyst_resume_dir: Path, analyst_job: str, fake_embedder: FakeEmbedder
    ) -> None:
        loaded = load_candidates_from_directory(analyst_resume_dir)
        requirements, profiles = analyze_candidates_for_job(
            loaded.candidates, analyst_job, embedder=fake_embedder
        )

        assert requirements.minimum_experience == 3.0
        assert len(profiles) == 3
        assert [profile.rank for profile in profiles] == [1, 2, 3]
        assert all(profile.semantic_match_score is not None for profile in profiles)

    def test_top_k_limits_the_profiles(
        self, analyst_resume_dir: Path, analyst_job: str, fake_embedder: FakeEmbedder
    ) -> None:
        loaded = load_candidates_from_directory(analyst_resume_dir)
        _, profiles = analyze_candidates_for_job(
            loaded.candidates, analyst_job, top_k=2, embedder=fake_embedder
        )
        assert len(profiles) == 2

    def test_analysis_is_deterministic_across_runs(
        self, analyst_resume_dir: Path, analyst_job: str
    ) -> None:
        loaded = load_candidates_from_directory(analyst_resume_dir)
        requirements = analyze_job_description(analyst_job)

        first = analyze_candidates(loaded.candidates, requirements=requirements)
        second = analyze_candidates(loaded.candidates, requirements=requirements)
        assert first == second

    def test_phase1_extraction_feeds_phase3_directly(self, analyst_resume_dir: Path) -> None:
        """Skills come from text the Phase 1 parser produced, not a second read."""
        text = extract_text_from_pdf(analyst_resume_dir / "sarah_wilson.pdf")
        assert "Power BI" in extract_skills(text)


class TestAnalyzeCLI:
    """The ``analyze`` subcommand, patched onto the offline embedder."""

    @pytest.fixture(autouse=True)
    def _use_fake_embedder(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("app.main.get_default_embedder", lambda *_args: FakeEmbedder())

    @pytest.fixture
    def job_file(self, tmp_path: Path, analyst_job: str) -> Path:
        path = tmp_path / "analyst_job.txt"
        path.write_text(analyst_job, encoding="utf-8")
        return path

    def test_reports_job_requirements(
        self, analyst_resume_dir: Path, job_file: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["analyze", "-r", str(analyst_resume_dir), "-j", str(job_file)]) == 0
        output = capsys.readouterr().out

        assert "JOB REQUIREMENTS" in output
        assert "Minimum Experience: 3 years" in output

    def test_reports_matched_and_missing_skills(
        self, analyst_resume_dir: Path, job_file: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        main(["analyze", "-r", str(analyst_resume_dir), "-j", str(job_file)])
        output = capsys.readouterr().out

        assert "Matched Skills" in output
        assert "Missing Skills" in output

    def test_reports_every_experience_verdict(
        self, analyst_resume_dir: Path, job_file: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        main(["analyze", "-r", str(analyst_resume_dir), "-j", str(job_file)])
        output = capsys.readouterr().out

        assert "Requirement Met: Yes" in output
        assert "Requirement Met: No" in output
        assert "Requirement Met: Unknown" in output

    def test_unknown_experience_is_not_shown_as_a_number(
        self, analyst_resume_dir: Path, job_file: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        main(["analyze", "-r", str(analyst_resume_dir), "-j", str(job_file)])
        assert "not stated on resume" in capsys.readouterr().out

    def test_reports_education(
        self, analyst_resume_dir: Path, job_file: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        main(["analyze", "-r", str(analyst_resume_dir), "-j", str(job_file)])
        output = capsys.readouterr().out

        assert "Education:" in output
        assert "MBA - Finance" in output

    def test_score_is_never_called_a_probability(
        self, analyst_resume_dir: Path, job_file: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        main(["analyze", "-r", str(analyst_resume_dir), "-j", str(job_file)])
        output = capsys.readouterr().out

        assert "Semantic Match Score" in output
        assert "not a probability of being hired" in output
        # The only occurrence of the word is inside that disclaimer.
        assert output.replace("not a probability of being hired", "").count("probability") == 0

    def test_extra_skills_are_hidden_by_default(
        self, analyst_resume_dir: Path, job_file: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        main(["analyze", "-r", str(analyst_resume_dir), "-j", str(job_file)])
        assert "Additional Skills" not in capsys.readouterr().out

    def test_extra_skills_shown_on_request(
        self, analyst_resume_dir: Path, job_file: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        main(
            [
                "analyze",
                "-r",
                str(analyst_resume_dir),
                "-j",
                str(job_file),
                "--show-extra-skills",
            ]
        )
        assert "Additional Skills" in capsys.readouterr().out

    def test_top_k_limits_the_report(
        self, analyst_resume_dir: Path, job_file: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        main(["analyze", "-r", str(analyst_resume_dir), "-j", str(job_file), "-k", "1"])
        assert capsys.readouterr().out.count("Semantic Match Score") == 1

    def test_inline_job_text_works(
        self, analyst_resume_dir: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        exit_code = main(
            [
                "analyze",
                "-r",
                str(analyst_resume_dir),
                "-t",
                "Financial analyst with 3+ years of experience in Python and SQL",
            ]
        )
        assert exit_code == 0
        assert "JOB REQUIREMENTS" in capsys.readouterr().out

    def test_empty_job_text_exits_cleanly(
        self, analyst_resume_dir: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        exit_code = main(["analyze", "-r", str(analyst_resume_dir), "-t", "   "])
        captured = capsys.readouterr()

        assert exit_code == 1
        assert "Error: " in captured.err
        assert "Traceback" not in captured.err

    def test_missing_resume_directory_errors_cleanly(
        self, tmp_path: Path, job_file: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        exit_code = main(["analyze", "-r", str(tmp_path / "nope"), "-j", str(job_file)])
        captured = capsys.readouterr()

        assert exit_code == 1
        assert "not found" in captured.err
        assert "Traceback" not in captured.err

    def test_a_job_description_source_is_required(self, analyst_resume_dir: Path) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main(["analyze", "-r", str(analyst_resume_dir)])
        assert exc_info.value.code == 2

    def test_match_subcommand_is_unaffected(
        self, analyst_resume_dir: Path, job_file: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Phase 2 output must not change now that `analyze` exists."""
        assert main(["match", "-r", str(analyst_resume_dir), "-j", str(job_file)]) == 0
        output = capsys.readouterr().out

        assert "Rank" in output
        assert "JOB REQUIREMENTS" not in output

    def test_phase1_bare_path_form_still_works(
        self, valid_pdf: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Three subcommands in, the original Phase 1 invocation still works."""
        assert main([str(valid_pdf)]) == 0
        assert "Jane Doe" in capsys.readouterr().out
