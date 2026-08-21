"""Retrieval over resume chunks.

Reuses the Phase 2 embedding and FAISS infrastructure: this module adds chunk
bookkeeping and candidate isolation, not a second vector implementation.

Candidate isolation
-------------------
Mixing one candidate's resume into another's analysis would make the LLM state
things about a person that their resume never said -- the most damaging failure
this system could have. Isolation is therefore **structural, not a filter**:
each candidate gets its own :class:`~app.vector_store.VectorStore`, so a search
scoped to a candidate physically cannot reach another candidate's vectors.

A post-search filter over one shared index was rejected deliberately. It leaks
in a subtle way: with ``top_k=5`` on a shared index, all five nearest vectors can
belong to another candidate, leaving the requested candidate with nothing, and
any bug in the filter silently produces cross-contamination rather than an
error.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from app.chunker import ResumeChunk
from app.embeddings import TextEmbedder, get_default_embedder
from app.vector_store import VectorStore

__all__ = [
    "DEFAULT_TOP_K",
    "RetrievalError",
    "UnknownCandidateError",
    "RetrievedEvidence",
    "ChunkRetriever",
]

DEFAULT_TOP_K = 4


class RetrievalError(Exception):
    """Base class for every error raised by this module."""


class UnknownCandidateError(RetrievalError):
    """Retrieval was scoped to a candidate that was never indexed."""


@dataclass(frozen=True, slots=True)
class RetrievedEvidence:
    """One resume passage retrieved for a query.

    Attributes:
        candidate_id: Candidate the passage belongs to.
        chunk_id: Identifier of the source chunk.
        text: The passage itself, verbatim from the resume.
        retrieval_score: Cosine similarity to the query, in ``[-1.0, 1.0]``.
            A **similarity score**, not a probability and not a confidence that
            the passage answers the query.
        chunk_index: Position of the chunk within the resume.
        source: Path the resume came from, when known.
    """

    candidate_id: str
    chunk_id: str
    text: str
    retrieval_score: float
    chunk_index: int
    source: Path | None = None


class ChunkRetriever:
    """Indexes resume chunks and retrieves the passages relevant to a query.

    Args:
        embedder: Component used to embed chunks and queries. Defaults to the
            shared Phase 2 embedder. Inject a fake for offline testing.
    """

    def __init__(self, embedder: TextEmbedder | None = None) -> None:
        self._embedder = embedder if embedder is not None else get_default_embedder()
        self._stores: dict[str, VectorStore[ResumeChunk]] = {}
        self._chunk_count = 0

    @property
    def embedder(self) -> TextEmbedder:
        """The embedder backing this retriever."""
        return self._embedder

    @property
    def candidate_ids(self) -> tuple[str, ...]:
        """Candidates currently indexed, in first-indexed order."""
        return tuple(self._stores)

    @property
    def is_empty(self) -> bool:
        """Whether anything has been indexed yet."""
        return self._chunk_count == 0

    def __len__(self) -> int:
        """Total number of indexed chunks across all candidates."""
        return self._chunk_count

    def chunk_count(self, candidate_id: str) -> int:
        """Number of chunks indexed for one candidate, or 0 if unknown."""
        store = self._stores.get(candidate_id)
        return 0 if store is None else len(store)

    def index_chunks(self, chunks: Sequence[ResumeChunk]) -> None:
        """Embed and index resume chunks, replacing anything indexed before.

        Every chunk is embedded in a single batched call, then partitioned into
        one vector store per candidate.

        Args:
            chunks: Chunks to index. An empty sequence clears the retriever.

        Raises:
            TypeError: If an entry is not a :class:`~app.chunker.ResumeChunk`.
            app.embeddings.EmbeddingError: If embedding fails.
        """
        entries = list(chunks)
        self._stores = {}
        self._chunk_count = 0

        if not entries:
            return

        for position, chunk in enumerate(entries):
            if not isinstance(chunk, ResumeChunk):
                raise TypeError(
                    f"chunks[{position}] must be a ResumeChunk, got {type(chunk).__name__}"
                )

        vectors = self._embedder.embed_texts([chunk.text for chunk in entries])

        grouped: dict[str, list[int]] = {}
        for position, chunk in enumerate(entries):
            grouped.setdefault(chunk.candidate_id, []).append(position)

        for candidate_id, positions in grouped.items():
            store: VectorStore[ResumeChunk] = VectorStore(self._embedder.dimension)
            store.add(vectors[positions], [entries[position] for position in positions])
            self._stores[candidate_id] = store

        self._chunk_count = len(entries)

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        candidate_id: str | None = None,
    ) -> tuple[RetrievedEvidence, ...]:
        """Retrieve the chunks most similar to ``query``.

        Args:
            query: Free text, typically a job description or a single requirement.
            top_k: Maximum passages to return. Values above the number of
                indexed chunks are clamped, not padded.
            candidate_id: Restrict retrieval to one candidate. **Pass this
                whenever building per-candidate context**; leaving it ``None``
                searches every candidate and returns a mixed result set.

        Returns:
            Evidence ordered by descending similarity. Empty when nothing is
            indexed, or when the query has no usable content.

        Raises:
            UnknownCandidateError: If ``candidate_id`` was never indexed.
            ValueError: If ``top_k`` is not a positive integer.
        """
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got {top_k!r}")

        if candidate_id is not None and candidate_id not in self._stores:
            raise UnknownCandidateError(
                f"candidate {candidate_id!r} has no indexed chunks; "
                f"indexed candidates: {sorted(self._stores) or 'none'}"
            )

        if not isinstance(query, str) or not query.strip() or self.is_empty:
            return ()

        vector = self._embedder.embed_text(query)

        if candidate_id is not None:
            return self._search_store(self._stores[candidate_id], vector, top_k)

        merged: list[RetrievedEvidence] = []
        for store in self._stores.values():
            merged.extend(self._search_store(store, vector, top_k))

        merged.sort(key=lambda item: (-item.retrieval_score, item.chunk_id))
        return tuple(merged[:top_k])

    @staticmethod
    def _search_store(
        store: VectorStore[ResumeChunk],
        vector,
        top_k: int,
    ) -> tuple[RetrievedEvidence, ...]:
        """Search one candidate's store and convert hits into evidence."""
        return tuple(
            RetrievedEvidence(
                candidate_id=hit.metadata.candidate_id,
                chunk_id=hit.metadata.chunk_id,
                text=hit.metadata.text,
                retrieval_score=hit.score,
                chunk_index=hit.metadata.index,
                source=hit.metadata.source,
            )
            for hit in store.search(vector, top_k=top_k)
        )

    def retrieve_for_queries(
        self,
        queries: Iterable[str],
        candidate_id: str,
        top_k_per_query: int = 2,
        max_results: int = DEFAULT_TOP_K,
    ) -> tuple[RetrievedEvidence, ...]:
        """Retrieve evidence for several queries and merge the results.

        Running one query per requirement surfaces passages a single
        whole-job-description query would miss, since a short mention of one
        skill is easily outweighed by overall topical similarity.

        Args:
            queries: Query strings, e.g. the job description plus each required skill.
            candidate_id: Candidate to scope retrieval to. Required -- this
                method is only for building per-candidate context.
            top_k_per_query: Passages to take from each individual query.
            max_results: Cap on the merged result set.

        Returns:
            Deduplicated evidence ordered by descending similarity. A chunk
            matched by several queries keeps its highest score.

        Raises:
            UnknownCandidateError: If ``candidate_id`` was never indexed.
        """
        best: dict[str, RetrievedEvidence] = {}

        for query in queries:
            for item in self.retrieve(query, top_k=top_k_per_query, candidate_id=candidate_id):
                existing = best.get(item.chunk_id)
                if existing is None or item.retrieval_score > existing.retrieval_score:
                    best[item.chunk_id] = item

        ordered = sorted(best.values(), key=lambda item: (-item.retrieval_score, item.chunk_id))
        return tuple(ordered[:max_results])
