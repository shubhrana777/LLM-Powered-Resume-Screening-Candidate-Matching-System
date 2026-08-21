"""Unit tests for app.embeddings.

Validation and contract tests run against the offline FakeEmbedder; the tests
that need real transformer weights are marked ``model`` and skip when the model
cannot be loaded. See tests/conftest.py for the rationale.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.embeddings import (
    DEFAULT_MODEL_NAME,
    VECTOR_DTYPE,
    EmbeddingError,
    InvalidTextError,
    ModelLoadError,
    SentenceTransformerEmbedder,
    TextEmbedder,
    get_default_embedder,
)

from .conftest import BACKEND_JOB, BACKEND_RESUME, CHEF_RESUME, FakeEmbedder


class TestTextValidation:
    """Invalid text must raise rather than yield a meaningless vector."""

    @pytest.mark.parametrize("bad", ["", "   ", "\n\n", "\t "])
    def test_empty_or_whitespace_text_raises(self, fake_embedder: FakeEmbedder, bad: str) -> None:
        with pytest.raises(InvalidTextError):
            fake_embedder.embed_text(bad)

    @pytest.mark.parametrize("bad", [None, 42, 3.5, ["a list"], {"a": "dict"}])
    def test_non_string_text_raises(self, fake_embedder: FakeEmbedder, bad: object) -> None:
        with pytest.raises(InvalidTextError):
            fake_embedder.embed_text(bad)  # type: ignore[arg-type]

    def test_empty_batch_raises(self, fake_embedder: FakeEmbedder) -> None:
        with pytest.raises(InvalidTextError):
            fake_embedder.embed_texts([])

    def test_batch_containing_empty_string_raises(self, fake_embedder: FakeEmbedder) -> None:
        with pytest.raises(InvalidTextError):
            fake_embedder.embed_texts([BACKEND_RESUME, "   "])

    def test_error_message_identifies_the_offending_position(
        self, fake_embedder: FakeEmbedder
    ) -> None:
        with pytest.raises(InvalidTextError, match=r"texts\[1\]"):
            fake_embedder.embed_texts([BACKEND_RESUME, ""])

    def test_single_string_instead_of_sequence_raises(self, fake_embedder: FakeEmbedder) -> None:
        """A bare string is iterable; treating it as a batch would embed characters."""
        with pytest.raises(InvalidTextError):
            fake_embedder.embed_texts("a single string")  # type: ignore[arg-type]

    def test_non_iterable_batch_raises(self, fake_embedder: FakeEmbedder) -> None:
        with pytest.raises(InvalidTextError):
            fake_embedder.embed_texts(42)  # type: ignore[arg-type]

    def test_all_errors_share_a_base_class(self, fake_embedder: FakeEmbedder) -> None:
        with pytest.raises(EmbeddingError):
            fake_embedder.embed_text("")


class TestEmbeddingShape:
    """Contract every TextEmbedder implementation must satisfy."""

    def test_single_embedding_is_one_dimensional(self, fake_embedder: FakeEmbedder) -> None:
        vector = fake_embedder.embed_text(BACKEND_RESUME)
        assert vector.ndim == 1
        assert vector.shape == (fake_embedder.dimension,)

    def test_batch_embedding_is_two_dimensional(self, fake_embedder: FakeEmbedder) -> None:
        vectors = fake_embedder.embed_texts([BACKEND_RESUME, CHEF_RESUME])
        assert vectors.shape == (2, fake_embedder.dimension)

    def test_dtype_is_float32_for_faiss(self, fake_embedder: FakeEmbedder) -> None:
        assert fake_embedder.embed_texts([BACKEND_RESUME]).dtype == VECTOR_DTYPE

    def test_vectors_are_contiguous_for_faiss(self, fake_embedder: FakeEmbedder) -> None:
        assert fake_embedder.embed_texts([BACKEND_RESUME, CHEF_RESUME]).flags["C_CONTIGUOUS"]

    def test_vectors_are_unit_normalized(self, fake_embedder: FakeEmbedder) -> None:
        vectors = fake_embedder.embed_texts([BACKEND_RESUME, CHEF_RESUME])
        norms = np.linalg.norm(vectors, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)

    def test_embedding_is_deterministic(self, fake_embedder: FakeEmbedder) -> None:
        first = fake_embedder.embed_text(BACKEND_RESUME)
        second = fake_embedder.embed_text(BACKEND_RESUME)
        assert np.array_equal(first, second)

    def test_batch_matches_individual_embeddings(self, fake_embedder: FakeEmbedder) -> None:
        batch = fake_embedder.embed_texts([BACKEND_RESUME, CHEF_RESUME])
        assert np.allclose(batch[0], fake_embedder.embed_text(BACKEND_RESUME))
        assert np.allclose(batch[1], fake_embedder.embed_text(CHEF_RESUME))

    def test_leading_and_trailing_whitespace_is_ignored(self, fake_embedder: FakeEmbedder) -> None:
        assert np.allclose(
            fake_embedder.embed_text(BACKEND_RESUME),
            fake_embedder.embed_text(f"  \n {BACKEND_RESUME}  \n "),
        )


class TestSimilarityBehaviour:
    """Cosine similarity via dot product of the normalized vectors."""

    @staticmethod
    def _cosine(embedder: FakeEmbedder, left: str, right: str) -> float:
        return float(np.dot(embedder.embed_text(left), embedder.embed_text(right)))

    def test_related_text_scores_above_unrelated_text(self, fake_embedder: FakeEmbedder) -> None:
        related = self._cosine(fake_embedder, BACKEND_JOB, BACKEND_RESUME)
        unrelated = self._cosine(fake_embedder, BACKEND_JOB, CHEF_RESUME)
        assert related > unrelated

    def test_identical_text_scores_close_to_one(self, fake_embedder: FakeEmbedder) -> None:
        assert self._cosine(fake_embedder, BACKEND_RESUME, BACKEND_RESUME) == pytest.approx(
            1.0, abs=1e-5
        )


class TestEmbedderConstruction:
    def test_construction_does_not_load_the_model(self) -> None:
        """Constructing must stay cheap; the model loads lazily on first use."""
        embedder = SentenceTransformerEmbedder()
        assert embedder._model is None
        assert embedder.model_name == DEFAULT_MODEL_NAME

    def test_fake_embedder_satisfies_the_protocol(self, fake_embedder: FakeEmbedder) -> None:
        assert isinstance(fake_embedder, TextEmbedder)

    def test_real_embedder_satisfies_the_protocol(self) -> None:
        assert isinstance(SentenceTransformerEmbedder(), TextEmbedder)

    def test_unknown_model_name_raises_model_load_error(self) -> None:
        embedder = SentenceTransformerEmbedder("this-org/definitely-not-a-real-model-xyz")
        with pytest.raises(ModelLoadError):
            embedder.embed_text("some text")

    def test_default_embedder_is_reused_across_calls(self) -> None:
        """The cache is what prevents reloading the model for every document."""
        assert get_default_embedder() is get_default_embedder()

    def test_different_model_names_get_different_instances(self) -> None:
        assert get_default_embedder(DEFAULT_MODEL_NAME) is not get_default_embedder("other/model")


@pytest.mark.model
class TestRealModel:
    """Exercised against the actual Sentence Transformers weights."""

    def test_dimension_is_384(self, real_embedder: SentenceTransformerEmbedder) -> None:
        assert real_embedder.dimension == 384

    def test_produces_normalized_float32_vector(
        self, real_embedder: SentenceTransformerEmbedder
    ) -> None:
        vector = real_embedder.embed_text(BACKEND_RESUME)
        assert vector.shape == (384,)
        assert vector.dtype == VECTOR_DTYPE
        assert float(np.linalg.norm(vector)) == pytest.approx(1.0, abs=1e-4)

    def test_batch_shape(self, real_embedder: SentenceTransformerEmbedder) -> None:
        assert real_embedder.embed_texts([BACKEND_RESUME, CHEF_RESUME]).shape == (2, 384)

    def test_semantically_related_text_scores_higher(
        self, real_embedder: SentenceTransformerEmbedder
    ) -> None:
        """The real payoff: no vocabulary overlap needed, unlike keyword matching."""
        job = real_embedder.embed_text(BACKEND_JOB)
        related = float(np.dot(job, real_embedder.embed_text(BACKEND_RESUME)))
        unrelated = float(np.dot(job, real_embedder.embed_text(CHEF_RESUME)))
        assert related > unrelated

    def test_rejects_empty_text_before_touching_the_model(
        self, real_embedder: SentenceTransformerEmbedder
    ) -> None:
        with pytest.raises(InvalidTextError):
            real_embedder.embed_text("   ")
