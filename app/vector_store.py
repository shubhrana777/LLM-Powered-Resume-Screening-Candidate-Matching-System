"""A small in-memory FAISS vector store with attached metadata.

Phase 2 scope: hold resume embeddings, search them by similarity, and give back
the metadata that was stored alongside each vector.

Index choice
------------
``faiss.IndexFlatIP`` performs an exact inner-product search over every stored
vector. Combined with the L2-normalized vectors produced by
:mod:`app.embeddings`, the inner product *is* the cosine similarity, so scores
come straight out of FAISS with no rescaling.

Exact search is the right default at this scale: a recruiter workflow deals with
hundreds or thousands of resumes, where a flat index is fast and, unlike the
approximate indexes (IVF, HNSW), never misses a relevant candidate and needs no
training step.

The index lives in memory only. Persistence is deliberately left out until a
later phase needs it.
"""

from __future__ import annotations

from typing import Generic, Sequence, TypeVar

import faiss
import numpy as np

from app.embeddings import VECTOR_DTYPE

__all__ = [
    "VectorStoreError",
    "DimensionMismatchError",
    "EmptyIndexError",
    "SearchResult",
    "VectorStore",
]

MetadataT = TypeVar("MetadataT")


class VectorStoreError(Exception):
    """Base class for every error raised by this module."""


class DimensionMismatchError(VectorStoreError):
    """A vector does not have the dimension this store was built for."""


class EmptyIndexError(VectorStoreError):
    """A search was attempted against a store that holds no vectors."""


class SearchResult(Generic[MetadataT]):
    """One hit returned by :meth:`VectorStore.search`.

    Attributes:
        position: Insertion position of the vector inside the store.
        score: Cosine similarity to the query vector, in ``[-1.0, 1.0]``.
        metadata: The object stored alongside the vector when it was added.
    """

    __slots__ = ("position", "score", "metadata")

    def __init__(self, position: int, score: float, metadata: MetadataT) -> None:
        self.position = position
        self.score = score
        self.metadata = metadata

    def __repr__(self) -> str:
        return (
            f"SearchResult(position={self.position}, "
            f"score={self.score:.4f}, metadata={self.metadata!r})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SearchResult):
            return NotImplemented
        return (
            self.position == other.position
            and self.score == other.score
            and self.metadata == other.metadata
        )


class VectorStore(Generic[MetadataT]):
    """An in-memory FAISS index that keeps a metadata object per vector.

    Args:
        dimension: Length of every vector this store will accept. Must be positive.

    Raises:
        ValueError: If ``dimension`` is not a positive integer.
    """

    def __init__(self, dimension: int) -> None:
        if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension <= 0:
            raise ValueError(f"dimension must be a positive integer, got {dimension!r}")

        self._dimension = dimension
        self._index = faiss.IndexFlatIP(dimension)
        self._metadata: list[MetadataT] = []

    @property
    def dimension(self) -> int:
        """The vector length this store accepts."""
        return self._dimension

    @property
    def is_empty(self) -> bool:
        """Whether the store currently holds no vectors."""
        return len(self._metadata) == 0

    def __len__(self) -> int:
        """Number of vectors currently stored."""
        return len(self._metadata)

    def _prepare(self, vectors: np.ndarray, *, field: str) -> np.ndarray:
        """Coerce ``vectors`` to a contiguous 2-D float32 array of the right width.

        Args:
            vectors: A 1-D vector or a 2-D array of vectors.
            field: Name used in error messages.

        Returns:
            A C-contiguous ``(n, dimension)`` float32 array ready for FAISS.

        Raises:
            DimensionMismatchError: If the array is not numeric, is not 1-D or
                2-D, or its width is not :attr:`dimension`.
        """
        try:
            array = np.asarray(vectors, dtype=VECTOR_DTYPE)
        except (TypeError, ValueError) as exc:
            raise DimensionMismatchError(f"{field} must be a numeric array ({exc})") from exc

        if array.ndim == 1:
            array = array.reshape(1, -1)

        if array.ndim != 2:
            raise DimensionMismatchError(
                f"{field} must be 1-D or 2-D, got a {array.ndim}-D array"
            )

        if array.shape[1] != self._dimension:
            raise DimensionMismatchError(
                f"{field} has dimension {array.shape[1]}, "
                f"but this store was built for dimension {self._dimension}"
            )

        return np.ascontiguousarray(array, dtype=VECTOR_DTYPE)

    def add(self, vectors: np.ndarray, metadata: Sequence[MetadataT]) -> None:
        """Add vectors and their matching metadata to the store.

        Args:
            vectors: A ``(n, dimension)`` array, or a single 1-D vector.
            metadata: Exactly ``n`` objects, aligned positionally with ``vectors``.

        Raises:
            DimensionMismatchError: If the vectors have the wrong shape or width.
            ValueError: If the number of metadata entries does not match the
                number of vectors, or if both are empty.
        """
        if isinstance(metadata, (str, bytes)):
            raise ValueError("metadata must be a sequence of objects, not a single string")

        entries = list(metadata)
        array = self._prepare(vectors, field="vectors")

        if array.shape[0] != len(entries):
            raise ValueError(
                f"got {array.shape[0]} vectors but {len(entries)} metadata entries; "
                "they must correspond one-to-one"
            )

        if not entries:
            raise ValueError("nothing to add: vectors and metadata are both empty")

        self._index.add(array)
        self._metadata.extend(entries)

    def search(self, query: np.ndarray, top_k: int | None = None) -> list[SearchResult[MetadataT]]:
        """Find the stored vectors most similar to ``query``.

        Args:
            query: A single query vector of length :attr:`dimension`.
            top_k: Maximum number of results. Defaults to every stored vector.
                Values larger than the store are clamped to its size.

        Returns:
            Up to ``top_k`` results ordered by descending cosine similarity.

        Raises:
            EmptyIndexError: If the store holds no vectors.
            DimensionMismatchError: If ``query`` has the wrong dimension or is
                not a single vector.
            ValueError: If ``top_k`` is not a positive integer.
        """
        if self.is_empty:
            raise EmptyIndexError(
                "cannot search an empty vector store; add candidate vectors first"
            )

        array = self._prepare(query, field="query")
        if array.shape[0] != 1:
            raise DimensionMismatchError(
                f"search expects a single query vector, got {array.shape[0]}"
            )

        if top_k is None:
            top_k = len(self._metadata)
        elif not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got {top_k!r}")

        limit = min(top_k, len(self._metadata))
        scores, positions = self._index.search(array, limit)

        # FAISS pads with position -1 when fewer than `limit` vectors match.
        return [
            SearchResult(int(position), float(score), self._metadata[int(position)])
            for score, position in zip(scores[0], positions[0])
            if position != -1
        ]
