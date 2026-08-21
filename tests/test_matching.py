"""Unit tests for app.matching and app.models."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.embeddings import InvalidTextError
from app.matching import (
    CandidateMatcher,
    DuplicateCandidateError,
    EmptyCandidateListError,
    EmptyJobDescriptionError,
    MatchingError,
    NoCandidatesIndexedError,
    load_candidates_from_directory,
    rank_candidates,
)
from app.models import Candidate, InvalidCandidateError, MatchResult

from .conftest import BACKEND_JOB, BACKEND_RESUME, FakeEmbedder


class TestCandidateModel:
    def test_stores_its_fields(self) -> None:
        candidate = Candidate("c-1", BACKEND_RESUME, "Ada Lovelace")
        assert candidate.candidate_id == "c-1"
        assert candidate.candidate_name == "Ada Lovelace"

    def test_name_is_optional(self) -> None:
        assert Candidate("c-1", BACKEND_RESUME).candidate_name is None

    def test_display_name_falls_back_to_the_id(self) -> None:
        assert Candidate("c-1", BACKEND_RESUME).display_name == "c-1"
        assert Candidate("c-1", BACKEND_RESUME, "Ada").display_name == "Ada"

    def test_is_immutable(self) -> None:
        candidate = Candidate("c-1", BACKEND_RESUME)
        with pytest.raises(Exception):
            candidate.candidate_id = "c-2"  # type: ignore[misc]

    @pytest.mark.parametrize("bad_id", ["", "   ", None, 7])
    def test_invalid_id_raises(self, bad_id: object) -> None:
        with pytest.raises(InvalidCandidateError):
            Candidate(bad_id, BACKEND_RESUME)  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad_text", ["", "   ", "\n\t", None, 7])
    def test_invalid_resume_text_raises(self, bad_text: object) -> None:
        with pytest.raises(InvalidCandidateError):
            Candidate("c-1", bad_text)  # type: ignore[arg-type]

    def test_error_message_names_the_candidate(self) -> None:
        with pytest.raises(InvalidCandidateError, match="c-42"):
            Candidate("c-42", "  ")


class TestIndexing:
    def test_candidate_count_reflects_indexed_candidates(
        self, fake_embedder: FakeEmbedder, sample_candidates: list[Candidate]
    ) -> None:
        matcher = CandidateMatcher(embedder=fake_embedder)
        assert matcher.candidate_count == 0
        matcher.index_candidates(sample_candidates)
        assert matcher.candidate_count == len(sample_candidates)

    def test_reindexing_replaces_the_previous_set(
        self, fake_embedder: FakeEmbedder, sample_candidates: list[Candidate]
    ) -> None:
        matcher = CandidateMatcher(embedder=fake_embedder)
        matcher.index_candidates(sample_candidates)
        matcher.index_candidates(sample_candidates[:2])
        assert matcher.candidate_count == 2

    def test_empty_candidate_list_raises(self, fake_embedder: FakeEmbedder) -> None:
        matcher = CandidateMatcher(embedder=fake_embedder)
        with pytest.raises(EmptyCandidateListError):
            matcher.index_candidates([])

    def test_duplicate_candidate_ids_raise(self, fake_embedder: FakeEmbedder) -> None:
        matcher = CandidateMatcher(embedder=fake_embedder)
        duplicates = [Candidate("same", BACKEND_RESUME), Candidate("same", BACKEND_RESUME)]
        with pytest.raises(DuplicateCandidateError, match="same"):
            matcher.index_candidates(duplicates)

    def test_non_candidate_entry_raises(self, fake_embedder: FakeEmbedder) -> None:
        matcher = CandidateMatcher(embedder=fake_embedder)
        with pytest.raises(InvalidCandidateError):
            matcher.index_candidates([{"candidate_id": "c-1"}])  # type: ignore[list-item]

    def test_embeds_every_resume_exactly_once(
        self, sample_candidates: list[Candidate]
    ) -> None:
        """Guards the performance requirement against duplicate embedding work."""

        class CountingEmbedder(FakeEmbedder):
            def __init__(self) -> None:
                super().__init__()
                self.embedded = 0

            def embed_texts(self, texts):  # type: ignore[no-untyped-def]
                self.embedded += len(list(texts))
                return super().embed_texts(texts)

        embedder = CountingEmbedder()
        CandidateMatcher(embedder=embedder).index_candidates(sample_candidates)
        assert embedder.embedded == len(sample_candidates)


class TestMatching:
    @pytest.fixture
    def matcher(
        self, fake_embedder: FakeEmbedder, sample_candidates: list[Candidate]
    ) -> CandidateMatcher:
        instance = CandidateMatcher(embedder=fake_embedder)
        instance.index_candidates(sample_candidates)
        return instance

    def test_returns_one_result_per_candidate(self, matcher: CandidateMatcher) -> None:
        assert len(matcher.match(BACKEND_JOB)) == 4

    def test_relevant_candidate_ranks_first(self, matcher: CandidateMatcher) -> None:
        assert matcher.match(BACKEND_JOB)[0].candidate_id == "c-backend"

    def test_irrelevant_candidate_ranks_last(self, matcher: CandidateMatcher) -> None:
        assert matcher.match(BACKEND_JOB)[-1].candidate_id == "c-chef"

    def test_relevant_scores_above_irrelevant(self, matcher: CandidateMatcher) -> None:
        by_id = {r.candidate_id: r.similarity_score for r in matcher.match(BACKEND_JOB)}
        assert by_id["c-backend"] > by_id["c-chef"]

    def test_scores_descend_with_rank(self, matcher: CandidateMatcher) -> None:
        scores = [r.similarity_score for r in matcher.match(BACKEND_JOB)]
        assert scores == sorted(scores, reverse=True)

    def test_ranks_are_sequential_from_one(self, matcher: CandidateMatcher) -> None:
        assert [r.rank for r in matcher.match(BACKEND_JOB)] == [1, 2, 3, 4]

    def test_top_k_limits_results_but_keeps_ordering(self, matcher: CandidateMatcher) -> None:
        results = matcher.match(BACKEND_JOB, top_k=2)
        assert len(results) == 2
        assert [r.rank for r in results] == [1, 2]
        assert results[0].candidate_id == "c-backend"

    def test_candidate_metadata_is_preserved(self, matcher: CandidateMatcher) -> None:
        top = matcher.match(BACKEND_JOB)[0]
        assert top.candidate_id == "c-backend"
        assert top.candidate_name == "Backend Person"

    def test_results_are_match_result_objects(self, matcher: CandidateMatcher) -> None:
        assert all(isinstance(r, MatchResult) for r in matcher.match(BACKEND_JOB))

    def test_scores_are_plain_floats(self, matcher: CandidateMatcher) -> None:
        """numpy scalars would leak into any JSON serialisation later."""
        assert all(type(r.similarity_score) is float for r in matcher.match(BACKEND_JOB))

    def test_scores_lie_within_the_documented_range(self, matcher: CandidateMatcher) -> None:
        assert all(-1.0 <= r.similarity_score <= 1.0 for r in matcher.match(BACKEND_JOB))

    def test_matching_is_repeatable(self, matcher: CandidateMatcher) -> None:
        first = matcher.match(BACKEND_JOB)
        second = matcher.match(BACKEND_JOB)
        assert [r.candidate_id for r in first] == [r.candidate_id for r in second]

    def test_one_index_serves_several_job_descriptions(
        self, matcher: CandidateMatcher
    ) -> None:
        """Indexing is separate from matching so resumes are embedded once."""
        backend_top = matcher.match(BACKEND_JOB)[0].candidate_id
        chef_top = matcher.match("pastry baking chocolate dessert kitchen")[0].candidate_id
        assert backend_top == "c-backend"
        assert chef_top == "c-chef"


class TestMatchingErrors:
    def test_matching_before_indexing_raises(self, fake_embedder: FakeEmbedder) -> None:
        with pytest.raises(NoCandidatesIndexedError):
            CandidateMatcher(embedder=fake_embedder).match(BACKEND_JOB)

    @pytest.mark.parametrize("bad", ["", "   ", "\n\t"])
    def test_empty_job_description_raises(
        self, fake_embedder: FakeEmbedder, sample_candidates: list[Candidate], bad: str
    ) -> None:
        matcher = CandidateMatcher(embedder=fake_embedder)
        matcher.index_candidates(sample_candidates)
        with pytest.raises(EmptyJobDescriptionError):
            matcher.match(bad)

    def test_non_string_job_description_raises(
        self, fake_embedder: FakeEmbedder, sample_candidates: list[Candidate]
    ) -> None:
        matcher = CandidateMatcher(embedder=fake_embedder)
        matcher.index_candidates(sample_candidates)
        with pytest.raises(EmptyJobDescriptionError):
            matcher.match(None)  # type: ignore[arg-type]

    def test_embedding_failure_propagates(
        self, sample_candidates: list[Candidate]
    ) -> None:
        class BrokenEmbedder(FakeEmbedder):
            def embed_texts(self, texts):  # type: ignore[no-untyped-def]
                raise InvalidTextError("simulated embedding failure")

        with pytest.raises(InvalidTextError, match="simulated"):
            CandidateMatcher(embedder=BrokenEmbedder()).index_candidates(sample_candidates)

    def test_matching_errors_share_a_base_class(self, fake_embedder: FakeEmbedder) -> None:
        with pytest.raises(MatchingError):
            CandidateMatcher(embedder=fake_embedder).index_candidates([])


class TestRankCandidates:
    def test_ranks_in_a_single_call(
        self, fake_embedder: FakeEmbedder, sample_candidates: list[Candidate]
    ) -> None:
        results = rank_candidates(sample_candidates, BACKEND_JOB, embedder=fake_embedder)
        assert results[0].candidate_id == "c-backend"
        assert results[-1].candidate_id == "c-chef"

    def test_honours_top_k(
        self, fake_embedder: FakeEmbedder, sample_candidates: list[Candidate]
    ) -> None:
        results = rank_candidates(
            sample_candidates, BACKEND_JOB, top_k=1, embedder=fake_embedder
        )
        assert len(results) == 1

    def test_empty_candidates_raise(self, fake_embedder: FakeEmbedder) -> None:
        with pytest.raises(EmptyCandidateListError):
            rank_candidates([], BACKEND_JOB, embedder=fake_embedder)

    def test_empty_job_description_raises(
        self, fake_embedder: FakeEmbedder, sample_candidates: list[Candidate]
    ) -> None:
        with pytest.raises(EmptyJobDescriptionError):
            rank_candidates(sample_candidates, "   ", embedder=fake_embedder)

    def test_job_description_is_validated_before_embedding_resumes(
        self, sample_candidates: list[Candidate]
    ) -> None:
        """Fail fast: do not pay for resume embeddings on an invalid query."""

        class TrackingEmbedder(FakeEmbedder):
            def __init__(self) -> None:
                super().__init__()
                self.calls = 0

            def embed_texts(self, texts):  # type: ignore[no-untyped-def]
                self.calls += 1
                return super().embed_texts(texts)

        embedder = TrackingEmbedder()
        with pytest.raises(EmptyJobDescriptionError):
            rank_candidates(sample_candidates, "", embedder=embedder)
        assert embedder.calls == 0


class TestLoadCandidatesFromDirectory:
    def test_loads_every_pdf(self, resume_dir: Path) -> None:
        loaded = load_candidates_from_directory(resume_dir)
        assert len(loaded.candidates) == 3
        assert loaded.failures == []

    def test_ignores_non_pdf_files(self, resume_dir: Path) -> None:
        ids = {c.candidate_id for c in load_candidates_from_directory(resume_dir).candidates}
        assert "notes" not in ids

    def test_candidate_id_comes_from_the_file_stem(self, resume_dir: Path) -> None:
        ids = {c.candidate_id for c in load_candidates_from_directory(resume_dir).candidates}
        assert ids == {"alice_backend", "bob_ml_resume", "carol_chef"}

    def test_derives_a_readable_name(self, resume_dir: Path) -> None:
        names = {c.candidate_name for c in load_candidates_from_directory(resume_dir).candidates}
        assert "Alice Backend" in names
        assert "Bob Ml" in names  # the "resume" suffix is dropped

    def test_records_the_source_path(self, resume_dir: Path) -> None:
        for candidate in load_candidates_from_directory(resume_dir).candidates:
            assert candidate.source_path is not None
            assert candidate.source_path.suffix == ".pdf"

    def test_resume_text_is_populated(self, resume_dir: Path) -> None:
        assert all(
            c.resume_text.strip() for c in load_candidates_from_directory(resume_dir).candidates
        )

    def test_results_are_sorted_by_filename(self, resume_dir: Path) -> None:
        ids = [c.candidate_id for c in load_candidates_from_directory(resume_dir).candidates]
        assert ids == sorted(ids)

    def test_unreadable_pdf_is_reported_not_raised(self, resume_dir: Path) -> None:
        """One bad file must not abort a whole batch."""
        (resume_dir / "broken.pdf").write_bytes(b"not a pdf at all")

        loaded = load_candidates_from_directory(resume_dir)
        assert len(loaded.candidates) == 3
        assert len(loaded.failures) == 1
        assert loaded.failures[0].path.name == "broken.pdf"
        assert loaded.failures[0].reason

    def test_empty_directory_returns_no_candidates(self, tmp_path: Path) -> None:
        loaded = load_candidates_from_directory(tmp_path)
        assert loaded.candidates == []
        assert loaded.failures == []

    def test_missing_directory_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_candidates_from_directory(tmp_path / "nope")

    def test_accepts_a_string_path(self, resume_dir: Path) -> None:
        assert len(load_candidates_from_directory(str(resume_dir)).candidates) == 3
