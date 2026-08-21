"""Unit tests for app.retriever.

Candidate isolation gets the most attention here. Mixing one person's resume
into another's analysis is the worst failure this system could produce, so it is
tested from several angles rather than once.
"""

from __future__ import annotations

import pytest

from app.chunker import chunk_candidate, chunk_candidates
from app.models import Candidate
from app.retriever import (
    ChunkRetriever,
    RetrievalError,
    RetrievedEvidence,
    UnknownCandidateError,
)

from .conftest import FakeEmbedder


@pytest.fixture
def retriever(isolation_candidates, isolation_embedder: FakeEmbedder) -> ChunkRetriever:
    """A retriever holding two candidates with disjoint vocabulary."""
    instance = ChunkRetriever(embedder=isolation_embedder)
    instance.index_chunks(chunk_candidates(isolation_candidates, chunk_size=15, chunk_overlap=3))
    return instance


class TestIndexing:
    def test_reports_indexed_candidates(self, retriever: ChunkRetriever) -> None:
        assert set(retriever.candidate_ids) == {"cand-a", "cand-b"}

    def test_length_is_the_total_chunk_count(self, retriever: ChunkRetriever) -> None:
        assert len(retriever) == retriever.chunk_count("cand-a") + retriever.chunk_count("cand-b")

    def test_starts_empty(self, isolation_embedder: FakeEmbedder) -> None:
        instance = ChunkRetriever(embedder=isolation_embedder)
        assert instance.is_empty
        assert len(instance) == 0
        assert instance.candidate_ids == ()

    def test_indexing_replaces_previous_content(
        self, retriever: ChunkRetriever, isolation_candidates
    ) -> None:
        retriever.index_chunks(chunk_candidate(isolation_candidates[0], 15, 3))
        assert retriever.candidate_ids == ("cand-a",)

    def test_indexing_nothing_clears_the_retriever(self, retriever: ChunkRetriever) -> None:
        retriever.index_chunks([])
        assert retriever.is_empty

    def test_chunk_count_for_unknown_candidate_is_zero(self, retriever: ChunkRetriever) -> None:
        assert retriever.chunk_count("nobody") == 0

    def test_non_chunk_entry_raises(self, isolation_embedder: FakeEmbedder) -> None:
        instance = ChunkRetriever(embedder=isolation_embedder)
        with pytest.raises(TypeError):
            instance.index_chunks(["not a chunk"])  # type: ignore[list-item]

    def test_embeds_every_chunk_exactly_once(self, isolation_candidates) -> None:
        """Guards against re-embedding, which would be pure waste."""

        class CountingEmbedder(FakeEmbedder):
            def __init__(self) -> None:
                super().__init__(("python", "sql", "kubernetes", "docker"))
                self.embedded = 0

            def embed_texts(self, texts):  # type: ignore[no-untyped-def]
                self.embedded += len(list(texts))
                return super().embed_texts(texts)

        embedder = CountingEmbedder()
        chunks = chunk_candidates(isolation_candidates, chunk_size=15, chunk_overlap=3)
        ChunkRetriever(embedder=embedder).index_chunks(chunks)

        assert embedder.embedded == len(chunks)


class TestRetrieval:
    def test_returns_evidence_objects(self, retriever: ChunkRetriever) -> None:
        hits = retriever.retrieve("python sql", top_k=2, candidate_id="cand-a")
        assert all(isinstance(hit, RetrievedEvidence) for hit in hits)

    def test_relevant_chunk_ranks_above_irrelevant_one(
        self, isolation_embedder: FakeEmbedder
    ) -> None:
        """The chunk containing the queried terms must come first."""
        candidate = Candidate(
            "cand-x",
            "Header line about nothing in particular. "
            + "filler " * 30
            + " python python sql database api",
            "X",
        )
        instance = ChunkRetriever(embedder=isolation_embedder)
        instance.index_chunks(chunk_candidate(candidate, chunk_size=12, chunk_overlap=0))

        top = instance.retrieve("python sql database", top_k=1, candidate_id="cand-x")[0]
        assert "python" in top.text.lower()

    def test_results_are_ordered_by_descending_score(self, retriever: ChunkRetriever) -> None:
        scores = [
            hit.retrieval_score
            for hit in retriever.retrieve("python sql", top_k=5, candidate_id="cand-a")
        ]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_limits_the_results(self, retriever: ChunkRetriever) -> None:
        assert len(retriever.retrieve("python", top_k=1, candidate_id="cand-a")) == 1

    def test_top_k_larger_than_the_index_is_clamped(self, retriever: ChunkRetriever) -> None:
        hits = retriever.retrieve("python", top_k=999, candidate_id="cand-a")
        assert len(hits) == retriever.chunk_count("cand-a")

    def test_metadata_is_preserved(self, retriever: ChunkRetriever) -> None:
        hit = retriever.retrieve("python", top_k=1, candidate_id="cand-a")[0]
        assert hit.candidate_id == "cand-a"
        assert hit.chunk_id.startswith("cand-a#")
        assert hit.chunk_index >= 0
        assert hit.text

    def test_source_path_is_preserved(self, isolation_embedder: FakeEmbedder, tmp_path) -> None:
        path = tmp_path / "cv.pdf"
        candidate = Candidate("cand-p", "python sql database api work", "P", path)
        instance = ChunkRetriever(embedder=isolation_embedder)
        instance.index_chunks(chunk_candidate(candidate))

        assert instance.retrieve("python", top_k=1, candidate_id="cand-p")[0].source == path

    def test_scores_are_plain_floats_in_range(self, retriever: ChunkRetriever) -> None:
        for hit in retriever.retrieve("python", top_k=3, candidate_id="cand-a"):
            assert type(hit.retrieval_score) is float
            assert -1.0 <= hit.retrieval_score <= 1.0

    def test_retrieval_is_deterministic(self, retriever: ChunkRetriever) -> None:
        first = retriever.retrieve("python sql", top_k=3, candidate_id="cand-a")
        second = retriever.retrieve("python sql", top_k=3, candidate_id="cand-a")
        assert [hit.chunk_id for hit in first] == [hit.chunk_id for hit in second]

    def test_unscoped_retrieval_may_span_candidates(self, retriever: ChunkRetriever) -> None:
        """Documented behaviour: omitting candidate_id searches everyone."""
        hits = retriever.retrieve("python kubernetes", top_k=10)
        assert len({hit.candidate_id for hit in hits}) >= 1


class TestEmptyAndInvalid:
    def test_retrieving_from_an_empty_retriever_returns_nothing(
        self, isolation_embedder: FakeEmbedder
    ) -> None:
        instance = ChunkRetriever(embedder=isolation_embedder)
        assert instance.retrieve("python", top_k=3) == ()

    @pytest.mark.parametrize("query", ["", "   ", "\n"])
    def test_blank_query_returns_nothing(self, retriever: ChunkRetriever, query: str) -> None:
        assert retriever.retrieve(query, top_k=3, candidate_id="cand-a") == ()

    def test_unknown_candidate_raises(self, retriever: ChunkRetriever) -> None:
        with pytest.raises(UnknownCandidateError, match="nobody"):
            retriever.retrieve("python", top_k=3, candidate_id="nobody")

    def test_unknown_candidate_error_is_a_retrieval_error(
        self, retriever: ChunkRetriever
    ) -> None:
        with pytest.raises(RetrievalError):
            retriever.retrieve("python", top_k=3, candidate_id="nobody")

    @pytest.mark.parametrize("bad", [0, -1, 2.5, "3", None])
    def test_invalid_top_k_raises(self, retriever: ChunkRetriever, bad: object) -> None:
        with pytest.raises(ValueError):
            retriever.retrieve("python", top_k=bad, candidate_id="cand-a")  # type: ignore[arg-type]


class TestCandidateIsolation:
    """Candidate A's evidence must never surface for candidate B."""

    def test_scoped_retrieval_returns_only_that_candidate(
        self, retriever: ChunkRetriever
    ) -> None:
        for candidate_id in ("cand-a", "cand-b"):
            hits = retriever.retrieve("python sql kubernetes docker", top_k=10, candidate_id=candidate_id)
            assert hits
            assert {hit.candidate_id for hit in hits} == {candidate_id}

    def test_query_matching_the_other_candidate_returns_no_foreign_text(
        self, retriever: ChunkRetriever
    ) -> None:
        """A Python query scoped to the Kubernetes candidate stays with them."""
        hits = retriever.retrieve("python sql database api", top_k=5, candidate_id="cand-b")

        assert all(hit.candidate_id == "cand-b" for hit in hits)
        assert all("python" not in hit.text.lower() for hit in hits)

    def test_chunk_ids_are_never_foreign(self, retriever: ChunkRetriever) -> None:
        hits = retriever.retrieve("kubernetes docker", top_k=10, candidate_id="cand-a")
        assert all(hit.chunk_id.startswith("cand-a#") for hit in hits)

    def test_scoped_retrieval_still_returns_results_when_the_other_scores_higher(
        self, retriever: ChunkRetriever
    ) -> None:
        """A post-search filter would return an empty list here; per-candidate
        indexes return that candidate's best passages instead."""
        hits = retriever.retrieve("kubernetes docker", top_k=3, candidate_id="cand-a")
        assert hits

    def test_multi_query_retrieval_is_isolated(self, retriever: ChunkRetriever) -> None:
        hits = retriever.retrieve_for_queries(
            ["python", "sql", "kubernetes", "docker"],
            candidate_id="cand-b",
            top_k_per_query=2,
            max_results=10,
        )
        assert hits
        assert {hit.candidate_id for hit in hits} == {"cand-b"}


class TestRetrieveForQueries:
    def test_merges_results_from_several_queries(self, retriever: ChunkRetriever) -> None:
        single = retriever.retrieve("python", top_k=1, candidate_id="cand-a")
        merged = retriever.retrieve_for_queries(
            ["python", "sql", "database", "api"],
            candidate_id="cand-a",
            top_k_per_query=1,
            max_results=10,
        )
        assert len(merged) >= len(single)

    def test_deduplicates_chunks_matched_by_several_queries(
        self, retriever: ChunkRetriever
    ) -> None:
        hits = retriever.retrieve_for_queries(
            ["python", "python", "python"],
            candidate_id="cand-a",
            top_k_per_query=2,
            max_results=10,
        )
        ids = [hit.chunk_id for hit in hits]
        assert len(ids) == len(set(ids))

    def test_max_results_caps_the_merged_set(self, retriever: ChunkRetriever) -> None:
        hits = retriever.retrieve_for_queries(
            ["python", "sql", "database", "api"],
            candidate_id="cand-a",
            top_k_per_query=3,
            max_results=2,
        )
        assert len(hits) <= 2

    def test_results_are_ordered_by_score(self, retriever: ChunkRetriever) -> None:
        hits = retriever.retrieve_for_queries(
            ["python", "sql"], candidate_id="cand-a", top_k_per_query=3, max_results=10
        )
        scores = [hit.retrieval_score for hit in hits]
        assert scores == sorted(scores, reverse=True)

    def test_unknown_candidate_raises(self, retriever: ChunkRetriever) -> None:
        with pytest.raises(UnknownCandidateError):
            retriever.retrieve_for_queries(["python"], candidate_id="nobody")

    def test_no_queries_yields_nothing(self, retriever: ChunkRetriever) -> None:
        assert retriever.retrieve_for_queries([], candidate_id="cand-a") == ()
