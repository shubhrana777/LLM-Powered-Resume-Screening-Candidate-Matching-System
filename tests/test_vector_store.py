"""Unit tests for app.vector_store."""

from __future__ import annotations

import numpy as np
import pytest

from app.vector_store import (
    DimensionMismatchError,
    EmptyIndexError,
    SearchResult,
    VectorStore,
    VectorStoreError,
)

DIMENSION = 4


def unit(*values: float) -> np.ndarray:
    """Build a 1-D float32 unit vector from ``values``."""
    vector = np.array(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


# Three orthogonal-ish directions plus one near-duplicate of the first.
VEC_A = unit(1, 0, 0, 0)
VEC_B = unit(0, 1, 0, 0)
VEC_C = unit(0, 0, 1, 0)
VEC_A_NEAR = unit(0.95, 0.05, 0, 0)


@pytest.fixture
def populated_store() -> VectorStore[str]:
    """A store holding three labelled vectors."""
    store: VectorStore[str] = VectorStore(DIMENSION)
    store.add(np.vstack([VEC_A, VEC_B, VEC_C]), ["alpha", "beta", "gamma"])
    return store


class TestConstruction:
    def test_reports_its_dimension(self) -> None:
        assert VectorStore(DIMENSION).dimension == DIMENSION

    def test_starts_empty(self) -> None:
        store: VectorStore[str] = VectorStore(DIMENSION)
        assert store.is_empty
        assert len(store) == 0

    @pytest.mark.parametrize("bad", [0, -1, 2.5, "4", None, True])
    def test_invalid_dimension_raises(self, bad: object) -> None:
        with pytest.raises(ValueError):
            VectorStore(bad)  # type: ignore[arg-type]


class TestAdd:
    def test_length_reflects_added_vectors(self, populated_store: VectorStore[str]) -> None:
        assert len(populated_store) == 3
        assert not populated_store.is_empty

    def test_accepts_a_single_one_dimensional_vector(self) -> None:
        store: VectorStore[str] = VectorStore(DIMENSION)
        store.add(VEC_A, ["alpha"])
        assert len(store) == 1

    def test_add_is_cumulative(self, populated_store: VectorStore[str]) -> None:
        populated_store.add(VEC_A_NEAR, ["delta"])
        assert len(populated_store) == 4

    def test_accepts_a_plain_nested_list(self) -> None:
        store: VectorStore[str] = VectorStore(DIMENSION)
        store.add([[1.0, 0.0, 0.0, 0.0]], ["alpha"])
        assert len(store) == 1

    def test_wrong_dimension_raises(self) -> None:
        store: VectorStore[str] = VectorStore(DIMENSION)
        with pytest.raises(DimensionMismatchError, match="dimension 3"):
            store.add(np.zeros((1, 3), dtype=np.float32), ["alpha"])

    def test_metadata_count_mismatch_raises(self) -> None:
        store: VectorStore[str] = VectorStore(DIMENSION)
        with pytest.raises(ValueError, match="one-to-one"):
            store.add(np.vstack([VEC_A, VEC_B]), ["only one"])

    def test_adding_nothing_raises(self) -> None:
        store: VectorStore[str] = VectorStore(DIMENSION)
        with pytest.raises(ValueError):
            store.add(np.zeros((0, DIMENSION), dtype=np.float32), [])

    def test_three_dimensional_array_raises(self) -> None:
        store: VectorStore[str] = VectorStore(DIMENSION)
        with pytest.raises(DimensionMismatchError):
            store.add(np.zeros((1, 1, DIMENSION), dtype=np.float32), ["alpha"])

    def test_string_metadata_raises(self) -> None:
        """A bare string would silently be treated as a sequence of characters."""
        store: VectorStore[str] = VectorStore(DIMENSION)
        with pytest.raises(ValueError):
            store.add(VEC_A, "alpha")  # type: ignore[arg-type]

    def test_non_numeric_vectors_raise(self) -> None:
        store: VectorStore[str] = VectorStore(DIMENSION)
        with pytest.raises(DimensionMismatchError):
            store.add([["a", "b", "c", "d"]], ["alpha"])


class TestSearch:
    def test_returns_the_closest_vector_first(self, populated_store: VectorStore[str]) -> None:
        results = populated_store.search(VEC_A_NEAR)
        assert results[0].metadata == "alpha"

    def test_results_are_ordered_by_descending_score(
        self, populated_store: VectorStore[str]
    ) -> None:
        scores = [result.score for result in populated_store.search(VEC_A_NEAR)]
        assert scores == sorted(scores, reverse=True)

    def test_returns_every_vector_by_default(self, populated_store: VectorStore[str]) -> None:
        assert len(populated_store.search(VEC_A)) == 3

    def test_top_k_limits_the_result_count(self, populated_store: VectorStore[str]) -> None:
        assert len(populated_store.search(VEC_A, top_k=2)) == 2

    def test_top_k_larger_than_the_store_is_clamped(
        self, populated_store: VectorStore[str]
    ) -> None:
        """FAISS pads with position -1; those must not become bogus results."""
        results = populated_store.search(VEC_A, top_k=99)
        assert len(results) == 3
        assert all(result.position >= 0 for result in results)

    def test_metadata_is_associated_with_the_right_vector(
        self, populated_store: VectorStore[str]
    ) -> None:
        for query, expected in [(VEC_A, "alpha"), (VEC_B, "beta"), (VEC_C, "gamma")]:
            assert populated_store.search(query, top_k=1)[0].metadata == expected

    def test_identical_vector_scores_close_to_one(
        self, populated_store: VectorStore[str]
    ) -> None:
        assert populated_store.search(VEC_A, top_k=1)[0].score == pytest.approx(1.0, abs=1e-5)

    def test_orthogonal_vector_scores_close_to_zero(
        self, populated_store: VectorStore[str]
    ) -> None:
        result = next(r for r in populated_store.search(VEC_A) if r.metadata == "beta")
        assert result.score == pytest.approx(0.0, abs=1e-5)

    def test_position_matches_insertion_order(self, populated_store: VectorStore[str]) -> None:
        assert populated_store.search(VEC_C, top_k=1)[0].position == 2

    def test_returns_search_result_objects(self, populated_store: VectorStore[str]) -> None:
        assert all(isinstance(r, SearchResult) for r in populated_store.search(VEC_A))

    def test_arbitrary_metadata_objects_are_preserved(self) -> None:
        payload = {"candidate_id": "c-1", "tags": ["python"]}
        store: VectorStore[dict] = VectorStore(DIMENSION)
        store.add(VEC_A, [payload])
        assert store.search(VEC_A, top_k=1)[0].metadata is payload


class TestErrorHandling:
    def test_searching_an_empty_store_raises(self) -> None:
        store: VectorStore[str] = VectorStore(DIMENSION)
        with pytest.raises(EmptyIndexError, match="empty vector store"):
            store.search(VEC_A)

    def test_query_with_wrong_dimension_raises(self, populated_store: VectorStore[str]) -> None:
        with pytest.raises(DimensionMismatchError):
            populated_store.search(np.zeros(DIMENSION + 1, dtype=np.float32))

    def test_multiple_query_vectors_raise(self, populated_store: VectorStore[str]) -> None:
        with pytest.raises(DimensionMismatchError, match="single query vector"):
            populated_store.search(np.vstack([VEC_A, VEC_B]))

    @pytest.mark.parametrize("bad", [0, -1, 2.5, "3"])
    def test_invalid_top_k_raises(self, populated_store: VectorStore[str], bad: object) -> None:
        with pytest.raises(ValueError):
            populated_store.search(VEC_A, top_k=bad)  # type: ignore[arg-type]

    def test_all_store_errors_share_a_base_class(self) -> None:
        store: VectorStore[str] = VectorStore(DIMENSION)
        with pytest.raises(VectorStoreError):
            store.search(VEC_A)
