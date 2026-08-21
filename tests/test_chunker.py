"""Unit tests for app.chunker."""

from __future__ import annotations

import pytest

from app.chunker import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    ChunkingError,
    ResumeChunk,
    chunk_candidate,
    chunk_candidates,
    chunk_text,
)
from app.models import Candidate

WORDS = [f"w{index}" for index in range(250)]
TEXT = " ".join(WORDS)


def texts(chunks) -> list[str]:
    """Extract just the chunk text from ``chunk_text`` output."""
    return [chunk for chunk, _start, _end in chunks]


class TestShortText:
    def test_text_shorter_than_one_chunk_yields_one_chunk(self) -> None:
        chunks = chunk_text("a short resume", chunk_size=50, chunk_overlap=10)
        assert len(chunks) == 1
        assert chunks[0][0] == "a short resume"

    def test_text_exactly_one_chunk_long_yields_one_chunk(self) -> None:
        text = " ".join(WORDS[:20])
        assert len(chunk_text(text, chunk_size=20, chunk_overlap=5)) == 1

    def test_single_word(self) -> None:
        assert texts(chunk_text("Python", chunk_size=10, chunk_overlap=2)) == ["Python"]


class TestLongText:
    def test_long_text_produces_several_chunks(self) -> None:
        chunks = chunk_text(TEXT, chunk_size=50, chunk_overlap=10)
        assert len(chunks) > 1

    def test_every_chunk_respects_the_size_limit(self) -> None:
        for chunk, _start, _end in chunk_text(TEXT, chunk_size=50, chunk_overlap=10):
            assert len(chunk.split()) <= 50

    def test_all_words_are_covered(self) -> None:
        """Nothing may be dropped between windows."""
        covered: set[str] = set()
        for chunk, _start, _end in chunk_text(TEXT, chunk_size=50, chunk_overlap=10):
            covered.update(chunk.split())
        assert covered == set(WORDS)

    def test_first_chunk_starts_at_the_beginning(self) -> None:
        assert chunk_text(TEXT, chunk_size=50, chunk_overlap=10)[0][0].startswith("w0 ")

    def test_last_chunk_reaches_the_end(self) -> None:
        assert chunk_text(TEXT, chunk_size=50, chunk_overlap=10)[-1][0].endswith("w249")

    def test_chunk_count_matches_the_step_arithmetic(self) -> None:
        chunks = chunk_text(" ".join(WORDS[:100]), chunk_size=40, chunk_overlap=10)
        # step 30 over 100 words: windows at 0, 30, 60 (reaches 100) -> 3
        assert len(chunks) == 3


class TestOverlap:
    def test_consecutive_chunks_share_the_overlap(self) -> None:
        chunks = texts(chunk_text(TEXT, chunk_size=50, chunk_overlap=10))
        first_tail = chunks[0].split()[-10:]
        second_head = chunks[1].split()[:10]
        assert first_tail == second_head

    def test_zero_overlap_produces_disjoint_chunks(self) -> None:
        chunks = texts(chunk_text(" ".join(WORDS[:100]), chunk_size=25, chunk_overlap=0))
        seen: list[str] = []
        for chunk in chunks:
            seen.extend(chunk.split())
        assert len(seen) == len(set(seen)) == 100

    def test_larger_overlap_produces_more_chunks(self) -> None:
        few = chunk_text(TEXT, chunk_size=50, chunk_overlap=5)
        many = chunk_text(TEXT, chunk_size=50, chunk_overlap=40)
        assert len(many) > len(few)


class TestWordBoundaries:
    def test_words_are_never_split(self) -> None:
        original = set(TEXT.split())
        for chunk, _start, _end in chunk_text(TEXT, chunk_size=17, chunk_overlap=3):
            assert set(chunk.split()).issubset(original)

    def test_original_formatting_is_preserved_inside_a_chunk(self) -> None:
        text = "Sarah Wilson\nSenior Analyst\n\nSKILLS\nPython, SQL"
        assert chunk_text(text, chunk_size=100, chunk_overlap=10)[0][0] == text

    def test_offsets_point_back_at_the_source_text(self) -> None:
        for chunk, start, end in chunk_text(TEXT, chunk_size=30, chunk_overlap=5):
            assert TEXT[start:end] == chunk

    def test_leading_and_trailing_whitespace_is_not_included(self) -> None:
        chunk, _start, _end = chunk_text("   padded resume text   ", 50, 10)[0]
        assert chunk == "padded resume text"


class TestEmptyInput:
    @pytest.mark.parametrize("empty", ["", "   ", "\n\t\n", None, 42])
    def test_empty_or_invalid_input_yields_no_chunks(self, empty: object) -> None:
        assert chunk_text(empty) == ()  # type: ignore[arg-type]


class TestConfigurationValidation:
    @pytest.mark.parametrize("bad", [0, -1, 2.5, "50", None, True])
    def test_invalid_chunk_size_raises(self, bad: object) -> None:
        with pytest.raises(ChunkingError):
            chunk_text(TEXT, chunk_size=bad)  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad", [-1, 2.5, "10", None, True])
    def test_invalid_overlap_raises(self, bad: object) -> None:
        with pytest.raises(ChunkingError):
            chunk_text(TEXT, chunk_size=50, chunk_overlap=bad)  # type: ignore[arg-type]

    @pytest.mark.parametrize("overlap", [50, 51, 100])
    def test_overlap_not_smaller_than_size_raises(self, overlap: int) -> None:
        """Otherwise the window would never advance and chunking would not end."""
        with pytest.raises(ChunkingError, match="never advance"):
            chunk_text(TEXT, chunk_size=50, chunk_overlap=overlap)

    def test_defaults_are_a_valid_combination(self) -> None:
        assert DEFAULT_CHUNK_OVERLAP < DEFAULT_CHUNK_SIZE
        assert chunk_text(TEXT)


class TestDeterminism:
    def test_repeated_calls_are_identical(self) -> None:
        assert chunk_text(TEXT, 40, 10) == chunk_text(TEXT, 40, 10)

    def test_chunks_are_in_reading_order(self) -> None:
        starts = [start for _chunk, start, _end in chunk_text(TEXT, 40, 10)]
        assert starts == sorted(starts)


class TestChunkCandidate:
    @pytest.fixture
    def candidate(self) -> Candidate:
        return Candidate("cand-1", TEXT, "Test Person")

    def test_returns_resume_chunks(self, candidate: Candidate) -> None:
        chunks = chunk_candidate(candidate, chunk_size=50, chunk_overlap=10)
        assert all(isinstance(chunk, ResumeChunk) for chunk in chunks)

    def test_every_chunk_carries_the_candidate_id(self, candidate: Candidate) -> None:
        for chunk in chunk_candidate(candidate, chunk_size=50, chunk_overlap=10):
            assert chunk.candidate_id == "cand-1"

    def test_chunk_ids_are_unique_and_indexed(self, candidate: Candidate) -> None:
        chunks = chunk_candidate(candidate, chunk_size=50, chunk_overlap=10)
        assert [chunk.chunk_id for chunk in chunks] == [
            f"cand-1#{index}" for index in range(len(chunks))
        ]

    def test_indexes_are_sequential(self, candidate: Candidate) -> None:
        chunks = chunk_candidate(candidate, chunk_size=50, chunk_overlap=10)
        assert [chunk.index for chunk in chunks] == list(range(len(chunks)))

    def test_source_path_is_carried_through(self, tmp_path) -> None:
        path = tmp_path / "cv.pdf"
        candidate = Candidate("cand-1", TEXT, "Test Person", path)
        assert all(chunk.source == path for chunk in chunk_candidate(candidate))

    def test_word_count_property(self, candidate: Candidate) -> None:
        chunk = chunk_candidate(candidate, chunk_size=50, chunk_overlap=10)[0]
        assert chunk.word_count == 50

    def test_non_candidate_raises(self) -> None:
        with pytest.raises(TypeError):
            chunk_candidate({"candidate_id": "c"})  # type: ignore[arg-type]

    def test_chunks_are_immutable(self, candidate: Candidate) -> None:
        chunk = chunk_candidate(candidate)[0]
        with pytest.raises(Exception):
            chunk.text = "changed"  # type: ignore[misc]


class TestChunkCandidates:
    def test_chunks_several_candidates(self, isolation_candidates) -> None:
        chunks = chunk_candidates(isolation_candidates, chunk_size=20, chunk_overlap=5)
        owners = {chunk.candidate_id for chunk in chunks}
        assert owners == {"cand-a", "cand-b"}

    def test_candidate_order_is_preserved(self, isolation_candidates) -> None:
        chunks = chunk_candidates(isolation_candidates, chunk_size=20, chunk_overlap=5)
        first_b = next(i for i, c in enumerate(chunks) if c.candidate_id == "cand-b")
        assert all(chunk.candidate_id == "cand-a" for chunk in chunks[:first_b])

    def test_chunk_ids_stay_unique_across_candidates(self, isolation_candidates) -> None:
        chunks = chunk_candidates(isolation_candidates, chunk_size=20, chunk_overlap=5)
        ids = [chunk.chunk_id for chunk in chunks]
        assert len(ids) == len(set(ids))

    def test_empty_candidate_list_yields_nothing(self) -> None:
        assert chunk_candidates([]) == ()

    def test_long_resume_produces_many_chunks(self, long_resume_candidate) -> None:
        chunks = chunk_candidate(long_resume_candidate)
        assert len(chunks) > 5
