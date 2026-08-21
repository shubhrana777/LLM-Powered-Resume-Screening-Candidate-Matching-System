"""Deterministic resume chunking for retrieval.

Why chunk at all
----------------
Two independent reasons:

1. **The embedding model truncates.** ``all-MiniLM-L6-v2`` (Phase 2) has
   ``max_seq_length = 256`` word pieces, roughly 200 words. A whole resume is
   typically 400-800 tokens, so embedding it as one vector silently discards the
   back half -- often the skills and education sections. Chunking keeps every
   part of the resume reachable.
2. **Retrieval needs granularity.** A single whole-resume vector can only answer
   "is this resume broadly similar to the job?". Chunks let the system pull the
   *specific* passages that speak to a requirement, which is what the LLM needs
   as grounded evidence.

Strategy
--------
Fixed-size sliding window over whitespace-separated words, with overlap.
Deliberately simple: no sentence models, no semantic clustering. It is
deterministic, cheap, and easy to reason about when a chunk looks wrong.

Chunk boundaries always fall between words, and each chunk is emitted as the
original substring spanning its first and last word, so punctuation, casing and
line breaks inside a chunk are preserved exactly as the parser produced them.

Overlap exists so a fact that straddles a boundary still appears whole in one of
the two neighbouring chunks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from app.models import Candidate

__all__ = [
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_CHUNK_OVERLAP",
    "ChunkingError",
    "ResumeChunk",
    "chunk_text",
    "chunk_candidate",
    "chunk_candidates",
]

# Measured in words. 80 words is roughly 110-150 MiniLM word pieces -- comfortably
# inside the 256 limit even when words tokenize into several pieces each -- while
# staying small enough that a retrieved chunk is a specific passage rather than
# most of the resume. Retrieval granularity, not just the token cap, sets this.
DEFAULT_CHUNK_SIZE = 80
DEFAULT_CHUNK_OVERLAP = 20

_WORD = re.compile(r"\S+")


class ChunkingError(ValueError):
    """Chunking was asked for with an impossible configuration."""


@dataclass(frozen=True, slots=True)
class ResumeChunk:
    """One retrievable passage of a resume.

    Attributes:
        candidate_id: Owner of this chunk. Every downstream component keys on
            this to keep candidates isolated.
        chunk_id: Identifier unique across all candidates, ``"<candidate_id>#<index>"``.
        text: The chunk text, exactly as it appeared in the resume.
        index: 0-based position of this chunk within the resume.
        source: Path the resume came from, when known.
        start: Character offset of the chunk in the resume text.
        end: End character offset of the chunk in the resume text.

    Note:
        There is no page number. The Phase 1 parser joins pages into a single
        string and does not report page boundaries; inventing one here would
        mean either rewriting the parser or guessing.
    """

    candidate_id: str
    chunk_id: str
    text: str
    index: int
    source: Path | None = None
    start: int = 0
    end: int = 0

    @property
    def word_count(self) -> int:
        """Number of whitespace-separated words in this chunk."""
        return len(self.text.split())


def _validate_settings(chunk_size: int, overlap: int) -> None:
    """Check the chunk size and overlap are usable together.

    Args:
        chunk_size: Maximum words per chunk.
        overlap: Words repeated between neighbouring chunks.

    Raises:
        ChunkingError: If either value is invalid, or overlap is not smaller
            than chunk_size (which would make the window never advance).
    """
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size <= 0:
        raise ChunkingError(f"chunk_size must be a positive integer, got {chunk_size!r}")

    if not isinstance(overlap, int) or isinstance(overlap, bool) or overlap < 0:
        raise ChunkingError(f"chunk_overlap must be a non-negative integer, got {overlap!r}")

    if overlap >= chunk_size:
        raise ChunkingError(
            f"chunk_overlap ({overlap}) must be smaller than chunk_size ({chunk_size}); "
            "otherwise the window would never advance"
        )


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> tuple[tuple[str, int, int], ...]:
    """Split text into overlapping word-bounded windows.

    Args:
        text: Cleaned resume text. Non-string or blank input yields no chunks.
        chunk_size: Maximum words per chunk.
        chunk_overlap: Words shared between consecutive chunks.

    Returns:
        A tuple of ``(chunk_text, start_offset, end_offset)`` triples in reading
        order. A resume shorter than ``chunk_size`` produces exactly one chunk.

    Raises:
        ChunkingError: If the size/overlap combination is invalid.
    """
    _validate_settings(chunk_size, chunk_overlap)

    if not isinstance(text, str) or not text.strip():
        return ()

    spans = [(match.start(), match.end()) for match in _WORD.finditer(text)]
    if not spans:
        return ()

    step = chunk_size - chunk_overlap
    chunks: list[tuple[str, int, int]] = []

    position = 0
    while position < len(spans):
        window = spans[position : position + chunk_size]
        start, end = window[0][0], window[-1][1]
        chunks.append((text[start:end], start, end))

        # The final window is the one that reached the end of the text.
        if position + chunk_size >= len(spans):
            break
        position += step

    return tuple(chunks)


def chunk_candidate(
    candidate: Candidate,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> tuple[ResumeChunk, ...]:
    """Split one candidate's resume into :class:`ResumeChunk` records.

    Args:
        candidate: The candidate whose resume text should be chunked.
        chunk_size: Maximum words per chunk.
        chunk_overlap: Words shared between consecutive chunks.

    Returns:
        Chunks in reading order, each tagged with the candidate id.

    Raises:
        TypeError: If ``candidate`` is not a :class:`~app.models.Candidate`.
        ChunkingError: If the size/overlap combination is invalid.
    """
    if not isinstance(candidate, Candidate):
        raise TypeError(f"expected a Candidate, got {type(candidate).__name__}")

    pieces = chunk_text(candidate.resume_text, chunk_size, chunk_overlap)

    return tuple(
        ResumeChunk(
            candidate_id=candidate.candidate_id,
            chunk_id=f"{candidate.candidate_id}#{index}",
            text=piece,
            index=index,
            source=candidate.source_path,
            start=start,
            end=end,
        )
        for index, (piece, start, end) in enumerate(pieces)
    )


def chunk_candidates(
    candidates: list[Candidate] | tuple[Candidate, ...],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> tuple[ResumeChunk, ...]:
    """Chunk several candidates, preserving candidate and chunk order.

    Args:
        candidates: Candidate records to chunk.
        chunk_size: Maximum words per chunk.
        chunk_overlap: Words shared between consecutive chunks.

    Returns:
        All chunks, grouped by candidate in input order.
    """
    return tuple(
        chunk
        for candidate in candidates
        for chunk in chunk_candidate(candidate, chunk_size, chunk_overlap)
    )
