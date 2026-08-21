"""Text embedding component built on Sentence Transformers.

Phase 2 scope: turn resume and job-description text into dense vectors that can
be compared with cosine similarity.

Design notes
------------
* Embeddings are **L2-normalized** on generation. That makes an inner product
  between any two vectors equal to their cosine similarity, which lets the
  vector store use a plain ``faiss.IndexFlatIP`` and read the raw FAISS score
  as a cosine similarity with no post-hoc rescaling.
* All vectors are ``float32`` and C-contiguous, because that is what FAISS
  requires.
* The model is expensive to load (hundreds of MB, several seconds), so
  :func:`get_default_embedder` memoizes one instance per model name and the
  underlying ``SentenceTransformer`` is loaded lazily on first use.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Protocol, Sequence, runtime_checkable

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - import only needed for type checking
    from sentence_transformers import SentenceTransformer

__all__ = [
    "DEFAULT_MODEL_NAME",
    "EmbeddingError",
    "InvalidTextError",
    "ModelLoadError",
    "TextEmbedder",
    "SentenceTransformerEmbedder",
    "get_default_embedder",
]

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Vectors handed to FAISS must be float32; this is the single source of truth.
VECTOR_DTYPE = np.float32


class EmbeddingError(Exception):
    """Base class for every error raised by this module."""


class InvalidTextError(EmbeddingError):
    """Input text is missing, not a string, or contains no usable characters."""


class ModelLoadError(EmbeddingError):
    """The Sentence Transformer model could not be loaded."""


@runtime_checkable
class TextEmbedder(Protocol):
    """Minimal interface the rest of the application depends on.

    Depending on this protocol rather than on :class:`SentenceTransformerEmbedder`
    keeps the vector store and matching engine testable without downloading a
    model. Any object exposing ``dimension``, ``embed_text`` and ``embed_texts``
    can be substituted.
    """

    @property
    def dimension(self) -> int:
        """Length of the vectors this embedder produces."""

    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single string as a 1-D float32 vector."""

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Embed many strings as a 2-D float32 array of shape (n, dimension)."""


def _validate_text(text: str, *, field: str = "text") -> str:
    """Return ``text`` stripped, or raise if it carries no usable content.

    Args:
        text: Candidate string to validate.
        field: Name used in the error message, for a clearer diagnostic.

    Returns:
        The stripped string.

    Raises:
        InvalidTextError: If ``text`` is not a string, is empty, or is
            whitespace only. Embedding such input would produce a vector that
            looks valid but means nothing, so it is rejected instead.
    """
    if not isinstance(text, str):
        raise InvalidTextError(f"{field} must be a string, got {type(text).__name__}")

    stripped = text.strip()
    if not stripped:
        raise InvalidTextError(f"{field} is empty or whitespace only")

    return stripped


def _validate_texts(texts: Sequence[str]) -> list[str]:
    """Validate a batch of strings, reporting the position of any bad entry.

    Args:
        texts: A sequence of strings to validate.

    Returns:
        The stripped strings, in the original order.

    Raises:
        InvalidTextError: If ``texts`` is not a sequence, is empty, or contains
            an entry that fails :func:`_validate_text`.
    """
    if isinstance(texts, (str, bytes)):
        raise InvalidTextError("texts must be a sequence of strings, not a single string")

    try:
        items = list(texts)
    except TypeError as exc:
        raise InvalidTextError(f"texts must be an iterable of strings ({exc})") from exc

    if not items:
        raise InvalidTextError("texts is empty; nothing to embed")

    return [
        _validate_text(item, field=f"texts[{position}]")
        for position, item in enumerate(items)
    ]


class SentenceTransformerEmbedder:
    """Embed text with a Sentence Transformers model.

    The model is loaded lazily on first use and then reused for the lifetime of
    the instance, so constructing this class is cheap.

    Args:
        model_name: Hugging Face model id. Defaults to
            ``sentence-transformers/all-MiniLM-L6-v2``, a 384-dimension model
            small and fast enough for local development.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        self._model_name = model_name
        self._model: SentenceTransformer | None = None

    @property
    def model_name(self) -> str:
        """The Hugging Face model id backing this embedder."""
        return self._model_name

    @property
    def dimension(self) -> int:
        """Length of the vectors this model produces (384 for MiniLM-L6-v2)."""
        return int(self._load().get_sentence_embedding_dimension())

    def _load(self) -> SentenceTransformer:
        """Load and cache the underlying model, downloading it on first run."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self._model_name)
            except Exception as exc:  # network, disk, or an unknown model id
                raise ModelLoadError(
                    f"Could not load embedding model {self._model_name!r}. "
                    "The first run downloads it from Hugging Face, so check your "
                    f"network connection and the model name. ({exc})"
                ) from exc
        return self._model

    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single piece of text.

        Args:
            text: Non-empty text, e.g. a resume or a job description.

        Returns:
            A 1-D L2-normalized float32 vector of length :attr:`dimension`.

        Raises:
            InvalidTextError: If ``text`` is empty, whitespace only, or not a string.
            ModelLoadError: If the model cannot be loaded.
        """
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Embed several pieces of text in one batched call.

        Batching is meaningfully faster than looping over :meth:`embed_text`,
        so prefer this method when embedding a set of resumes.

        Args:
            texts: A non-empty sequence of non-empty strings.

        Returns:
            A 2-D L2-normalized float32 array of shape ``(len(texts), dimension)``.

        Raises:
            InvalidTextError: If the sequence is empty or any entry is unusable.
            ModelLoadError: If the model cannot be loaded.
        """
        cleaned = _validate_texts(texts)
        model = self._load()

        vectors = model.encode(
            cleaned,
            normalize_embeddings=True,  # makes inner product == cosine similarity
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.ascontiguousarray(vectors, dtype=VECTOR_DTYPE)


@lru_cache(maxsize=4)
def get_default_embedder(
    model_name: str = DEFAULT_MODEL_NAME,
) -> SentenceTransformerEmbedder:
    """Return a shared embedder for ``model_name``.

    Memoized so repeated calls across the CLI and the matching engine reuse one
    model instead of paying the load cost again. Call
    ``get_default_embedder.cache_clear()`` to drop the cached instances.

    Args:
        model_name: Hugging Face model id.

    Returns:
        A process-wide :class:`SentenceTransformerEmbedder` for that model.
    """
    return SentenceTransformerEmbedder(model_name)
