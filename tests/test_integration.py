"""End-to-end tests: PDF resume -> extracted text -> embedding -> FAISS -> ranking.

These exercise the Phase 1 parser and the Phase 2 engine together, through both
the library API and the command-line interface.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.main import main
from app.matching import CandidateMatcher, load_candidates_from_directory, rank_candidates

from .conftest import BACKEND_JOB, FakeEmbedder


class TestPipeline:
    """The full library pipeline, driven by the offline embedder."""

    def test_pdf_directory_to_ranked_results(
        self, resume_dir: Path, fake_embedder: FakeEmbedder
    ) -> None:
        loaded = load_candidates_from_directory(resume_dir)
        results = rank_candidates(loaded.candidates, BACKEND_JOB, embedder=fake_embedder)

        assert len(results) == 3
        assert results[0].candidate_id == "alice_backend"
        assert results[-1].candidate_id == "carol_chef"

    def test_ranking_is_internally_consistent(
        self, resume_dir: Path, fake_embedder: FakeEmbedder
    ) -> None:
        loaded = load_candidates_from_directory(resume_dir)
        results = rank_candidates(loaded.candidates, BACKEND_JOB, embedder=fake_embedder)

        scores = [r.similarity_score for r in results]
        assert scores == sorted(scores, reverse=True)
        assert [r.rank for r in results] == [1, 2, 3]

    def test_source_paths_survive_the_pipeline(
        self, resume_dir: Path, fake_embedder: FakeEmbedder
    ) -> None:
        loaded = load_candidates_from_directory(resume_dir)
        results = rank_candidates(loaded.candidates, BACKEND_JOB, embedder=fake_embedder)

        for result in results:
            assert result.source_path is not None
            assert result.source_path.is_file()

    def test_text_actually_came_from_the_pdfs(self, resume_dir: Path) -> None:
        """Ties the Phase 1 extractor to the Phase 2 candidate records."""
        loaded = load_candidates_from_directory(resume_dir)
        backend = next(c for c in loaded.candidates if c.candidate_id == "alice_backend")
        assert "backend" in backend.resume_text.lower()


@pytest.mark.model
class TestPipelineWithRealModel:
    """The same pipeline using real Sentence Transformers embeddings."""

    def test_relevant_candidate_outranks_irrelevant_one(
        self, resume_dir: Path, real_embedder
    ) -> None:
        loaded = load_candidates_from_directory(resume_dir)
        results = rank_candidates(loaded.candidates, BACKEND_JOB, embedder=real_embedder)

        ranking = [r.candidate_id for r in results]
        assert ranking[0] == "alice_backend"
        assert ranking[-1] == "carol_chef"

    def test_scores_are_in_range(self, resume_dir: Path, real_embedder) -> None:
        loaded = load_candidates_from_directory(resume_dir)
        results = rank_candidates(loaded.candidates, BACKEND_JOB, embedder=real_embedder)
        assert all(-1.0 <= r.similarity_score <= 1.0 for r in results)

    def test_switching_job_description_changes_the_winner(
        self, resume_dir: Path, real_embedder
    ) -> None:
        """Real semantic behaviour: a different role promotes a different resume."""
        loaded = load_candidates_from_directory(resume_dir)
        matcher = CandidateMatcher(embedder=real_embedder)
        matcher.index_candidates(loaded.candidates)

        backend_top = matcher.match(BACKEND_JOB)[0].candidate_id
        chef_top = matcher.match(
            "We are hiring a dessert cook for our restaurant kitchen."
        )[0].candidate_id

        assert backend_top == "alice_backend"
        assert chef_top == "carol_chef"


class TestMatchCLI:
    """The ``match`` subcommand, patched onto the offline embedder."""

    @pytest.fixture(autouse=True)
    def _use_fake_embedder(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Avoid downloading a model in CLI tests."""
        monkeypatch.setattr("app.main.get_default_embedder", lambda *_args: FakeEmbedder())

    @pytest.fixture
    def job_file(self, tmp_path: Path) -> Path:
        path = tmp_path / "job.txt"
        path.write_text(BACKEND_JOB, encoding="utf-8")
        return path

    def test_ranks_from_a_job_description_file(
        self, resume_dir: Path, job_file: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        exit_code = main(["match", "--resumes", str(resume_dir), "-j", str(job_file)])
        output = capsys.readouterr().out

        assert exit_code == 0
        assert "Alice Backend" in output
        assert output.index("Alice Backend") < output.index("Carol Chef")

    def test_ranks_from_inline_job_text(
        self, resume_dir: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        exit_code = main(["match", "--resumes", str(resume_dir), "--job-text", BACKEND_JOB])
        assert exit_code == 0
        assert "Alice Backend" in capsys.readouterr().out

    def test_top_k_limits_the_table(
        self, resume_dir: Path, job_file: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        main(["match", "-r", str(resume_dir), "-j", str(job_file), "-k", "1"])
        output = capsys.readouterr().out

        assert "Alice Backend" in output
        assert "Carol Chef" not in output

    def test_output_explains_the_score(
        self, resume_dir: Path, job_file: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The score must never be presented as a hiring probability."""
        main(["match", "-r", str(resume_dir), "-j", str(job_file)])
        output = capsys.readouterr().out

        assert "cosine similarity" in output
        assert "not a probability of being hired" in output

    def test_unreadable_resume_is_warned_about_but_does_not_fail(
        self, resume_dir: Path, job_file: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        (resume_dir / "broken.pdf").write_bytes(b"not a pdf")

        exit_code = main(["match", "-r", str(resume_dir), "-j", str(job_file)])
        captured = capsys.readouterr()

        assert exit_code == 0
        assert "Warning: skipped broken.pdf" in captured.err
        assert "Alice Backend" in captured.out

    @pytest.mark.parametrize(
        "argv_factory, expected",
        [
            (lambda d, j: ["match", "-r", str(d / "missing"), "-j", str(j)], "not found"),
            (lambda d, j: ["match", "-r", str(d), "-j", "no_such_job.txt"], "not found"),
            (lambda d, j: ["match", "-r", str(d), "-t", "   "], "empty"),
        ],
    )
    def test_user_errors_exit_cleanly(
        self,
        resume_dir: Path,
        job_file: Path,
        capsys: pytest.CaptureFixture[str],
        argv_factory,
        expected: str,
    ) -> None:
        exit_code = main(argv_factory(resume_dir, job_file))
        captured = capsys.readouterr()

        assert exit_code == 1
        assert captured.err.strip().startswith("Error: ") or "Error: " in captured.err
        assert expected in captured.err
        assert "Traceback" not in captured.err

    def test_directory_without_resumes_errors_cleanly(
        self, tmp_path: Path, job_file: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()

        exit_code = main(["match", "-r", str(empty), "-j", str(job_file)])
        captured = capsys.readouterr()

        assert exit_code == 1
        assert "no readable PDF resumes" in captured.err
        assert "Traceback" not in captured.err

    def test_empty_job_text_fails_before_embedding_resumes(
        self, resume_dir: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Embedding a folder is the expensive step; do not pay for a bad query."""
        exit_code = main(["match", "-r", str(resume_dir), "-t", "   "])
        captured = capsys.readouterr()

        assert exit_code == 1
        assert "Embedding" not in captured.err
        assert "job description is empty" in captured.err

    def test_job_description_and_job_text_are_mutually_exclusive(
        self, resume_dir: Path, job_file: Path
    ) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main(["match", "-r", str(resume_dir), "-j", str(job_file), "-t", "text"])
        assert exc_info.value.code == 2

    def test_a_job_description_source_is_required(self, resume_dir: Path) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main(["match", "-r", str(resume_dir)])
        assert exc_info.value.code == 2


class TestPhase1CLICompatibility:
    """The Phase 1 invocation form must survive the addition of subcommands."""

    def test_bare_path_still_extracts(
        self, valid_pdf: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main([str(valid_pdf)]) == 0
        assert "Jane Doe" in capsys.readouterr().out

    def test_explicit_extract_subcommand_works(
        self, valid_pdf: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["extract", str(valid_pdf)]) == 0
        assert "Jane Doe" in capsys.readouterr().out

    def test_both_forms_produce_identical_output(
        self, valid_pdf: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        main([str(valid_pdf)])
        bare = capsys.readouterr().out
        main(["extract", str(valid_pdf)])
        explicit = capsys.readouterr().out

        assert bare == explicit
