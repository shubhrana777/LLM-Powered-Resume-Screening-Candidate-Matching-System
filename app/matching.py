"""Semantic matching engine: rank candidates against a job description.

Pipeline
--------
1. Take candidate records (id, name, resume text).
2. Embed every resume in one batched call.
3. Store the vectors in a FAISS index, keeping each :class:`~app.models.Candidate`
   as the metadata for its vector.
4. Embed the job description.
5. Search the index and return candidates ordered by similarity.

What the score means
--------------------
``MatchResult.similarity_score`` is the **cosine similarity** between the
job-description embedding and the resume embedding.

* Both vectors are L2-normalized by :mod:`app.embeddings`, so the inner product
  FAISS computes is exactly the cosine of the angle between them. **No
  rescaling, normalization, or transformation is applied to the FAISS output** --
  the number returned here is the raw index score.
* The mathematical range is ``[-1.0, 1.0]``. In practice all-MiniLM-L6-v2 maps
  English prose into a narrow cone, so real resume/job pairs land well inside
  it. On the bundled sample data a strongly matching resume scores around
  ``0.70-0.80``, a loosely related one around ``0.35-0.50``, and an unrelated
  profile around ``0.15``. Negative scores are possible in theory but rare
  between two pieces of ordinary English text.
* It measures how similar two texts are *in the embedding space*. It is **not**
  a probability of being hired, not a percentage of requirements met, and not a
  calibrated measure of candidate quality. Only the relative ordering of scores
  within a single run is meaningful; absolute values should not be compared
  across different job descriptions or presented to a recruiter as a percentage.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple, Sequence

from app.embeddings import TextEmbedder, get_default_embedder
from app.models import Candidate, InvalidCandidateError, MatchResult
from app.resume_parser import ResumeParserError, extract_text_from_pdf
from app.vector_store import VectorStore

__all__ = [
    "MatchingError",
    "EmptyCandidateListError",
    "EmptyJobDescriptionError",
    "DuplicateCandidateError",
    "validate_job_description",
    "NoCandidatesIndexedError",
    "ResumeLoadFailure",
    "LoadedResumes",
    "CandidateMatcher",
    "rank_candidates",
    "load_candidates_from_directory",
]


class MatchingError(Exception):
    """Base class for every error raised by this module."""


class EmptyCandidateListError(MatchingError):
    """No candidates were supplied to match against."""


class EmptyJobDescriptionError(MatchingError):
    """The job description is missing or contains no usable text."""


class DuplicateCandidateError(MatchingError):
    """Two candidate records share the same ``candidate_id``."""


class NoCandidatesIndexedError(MatchingError):
    """:meth:`CandidateMatcher.match` was called before any candidates were indexed."""


class ResumeLoadFailure(NamedTuple):
    """A resume file that could not be read, and why."""

    path: Path
    reason: str


class LoadedResumes(NamedTuple):
    """The outcome of scanning a directory of resumes.

    Attributes:
        candidates: Candidates successfully parsed, sorted by file name.
        failures: Files that could not be parsed, each with a reason.
    """

    candidates: list[Candidate]
    failures: list[ResumeLoadFailure]


def _validate_candidates(candidates: Sequence[Candidate]) -> list[Candidate]:
    """Check that ``candidates`` is a usable, non-empty set of unique records.

    Args:
        candidates: The candidate records to validate.

    Returns:
        The candidates as a list.

    Raises:
        EmptyCandidateListError: If the sequence is empty.
        InvalidCandidateError: If an entry is not a :class:`Candidate`.
        DuplicateCandidateError: If two entries share a ``candidate_id``.
    """
    entries = list(candidates)
    if not entries:
        raise EmptyCandidateListError("no candidates supplied; nothing to rank")

    seen: set[str] = set()
    for position, candidate in enumerate(entries):
        if not isinstance(candidate, Candidate):
            raise InvalidCandidateError(
                f"candidates[{position}] must be a Candidate, "
                f"got {type(candidate).__name__}"
            )
        if candidate.candidate_id in seen:
            raise DuplicateCandidateError(
                f"duplicate candidate_id {candidate.candidate_id!r}; ids must be unique"
            )
        seen.add(candidate.candidate_id)

    return entries


def validate_job_description(job_description: str) -> str:
    """Return the stripped job description, or raise if it is unusable.

    Args:
        job_description: The job description text.

    Returns:
        The stripped text.

    Raises:
        EmptyJobDescriptionError: If it is not a string, or has no content.
    """
    if not isinstance(job_description, str):
        raise EmptyJobDescriptionError(
            f"job description must be a string, got {type(job_description).__name__}"
        )

    stripped = job_description.strip()
    if not stripped:
        raise EmptyJobDescriptionError("job description is empty or whitespace only")

    return stripped


class CandidateMatcher:
    """Rank candidates against job descriptions using semantic similarity.

    Indexing is separated from matching so that one set of resumes can be scored
    against several job descriptions without re-embedding the resumes.

    Args:
        embedder: Component used to turn text into vectors. Defaults to the
            shared :func:`app.embeddings.get_default_embedder` instance. Inject
            a different implementation of
            :class:`~app.embeddings.TextEmbedder` for testing.

    Example:
        >>> matcher = CandidateMatcher()                      # doctest: +SKIP
        >>> matcher.index_candidates(candidates)              # doctest: +SKIP
        >>> for result in matcher.match(job_description):     # doctest: +SKIP
        ...     print(result.rank, result.display_name, result.similarity_score)
    """

    def __init__(self, embedder: TextEmbedder | None = None) -> None:
        self._embedder = embedder if embedder is not None else get_default_embedder()
        self._store: VectorStore[Candidate] | None = None

    @property
    def embedder(self) -> TextEmbedder:
        """The embedder backing this matcher."""
        return self._embedder

    @property
    def candidate_count(self) -> int:
        """How many candidates are currently indexed."""
        return 0 if self._store is None else len(self._store)

    def index_candidates(self, candidates: Sequence[Candidate]) -> None:
        """Embed the given candidates and build a fresh FAISS index.

        Replaces any previously indexed candidates. Every resume is embedded in
        a single batched call, so each resume is embedded exactly once.

        Args:
            candidates: Non-empty sequence of unique :class:`Candidate` records.

        Raises:
            EmptyCandidateListError: If no candidates are supplied.
            DuplicateCandidateError: If two candidates share an id.
            InvalidCandidateError: If an entry is not a :class:`Candidate`.
            app.embeddings.EmbeddingError: If embedding fails.
        """
        entries = _validate_candidates(candidates)

        vectors = self._embedder.embed_texts([c.resume_text for c in entries])

        store: VectorStore[Candidate] = VectorStore(self._embedder.dimension)
        store.add(vectors, entries)
        self._store = store

    def match(self, job_description: str, top_k: int | None = None) -> list[MatchResult]:
        """Rank the indexed candidates against a job description.

        Args:
            job_description: The job description text.
            top_k: Maximum number of candidates to return. Defaults to all.

        Returns:
            :class:`~app.models.MatchResult` records ordered best-first, with
            ``rank`` starting at 1. See the module docstring for exactly what
            ``similarity_score`` means.

        Raises:
            NoCandidatesIndexedError: If :meth:`index_candidates` has not run.
            EmptyJobDescriptionError: If the job description has no content.
            app.embeddings.EmbeddingError: If embedding fails.
            app.vector_store.DimensionMismatchError: If the query vector width
                does not match the index.
        """
        if self._store is None:
            raise NoCandidatesIndexedError(
                "no candidates indexed; call index_candidates() before match()"
            )

        text = validate_job_description(job_description)
        query = self._embedder.embed_text(text)
        hits = self._store.search(query, top_k=top_k)

        return [
            MatchResult(
                candidate_id=hit.metadata.candidate_id,
                candidate_name=hit.metadata.candidate_name,
                similarity_score=hit.score,
                rank=rank,
                source_path=hit.metadata.source_path,
            )
            for rank, hit in enumerate(hits, start=1)
        ]


def rank_candidates(
    candidates: Sequence[Candidate],
    job_description: str,
    top_k: int | None = None,
    embedder: TextEmbedder | None = None,
) -> list[MatchResult]:
    """Index candidates and rank them against a job description in one call.

    Convenience wrapper around :class:`CandidateMatcher` for the common
    one-shot case. To score the same resumes against several job descriptions,
    use :class:`CandidateMatcher` directly and index once.

    Args:
        candidates: Non-empty sequence of unique candidate records.
        job_description: The job description text.
        top_k: Maximum number of results. Defaults to all candidates.
        embedder: Optional embedder override.

    Returns:
        Ranked results, best match first.

    Raises:
        EmptyCandidateListError: If no candidates are supplied.
        EmptyJobDescriptionError: If the job description has no content.
    """
    # Validate the job description before paying for resume embeddings.
    text = validate_job_description(job_description)

    matcher = CandidateMatcher(embedder=embedder)
    matcher.index_candidates(candidates)
    return matcher.match(text, top_k=top_k)


def _candidate_name_from_filename(path: Path) -> str:
    """Derive a display name from a resume file name.

    ``jane_doe_resume.pdf`` becomes ``Jane Doe``. This is a deliberately simple
    convention: extracting a real name from resume *content* is entity
    extraction, which belongs to a later phase.

    Args:
        path: Path to the resume file.

    Returns:
        A title-cased name, falling back to the raw stem if nothing is left.
    """
    words = [
        word
        for word in path.stem.replace("-", " ").replace("_", " ").split()
        if word.lower() not in {"resume", "cv"}
    ]
    return " ".join(words).title() if words else path.stem


def load_candidates_from_directory(directory: str | Path) -> LoadedResumes:
    """Build candidate records from every PDF in a directory.

    Reuses the Phase 1 parser for extraction. Files that cannot be parsed are
    collected as failures rather than aborting the whole run, so one unreadable
    resume does not block a batch.

    Args:
        directory: Directory to scan. Not searched recursively.

    Returns:
        A :class:`LoadedResumes` with the parsed candidates and any failures.

    Raises:
        FileNotFoundError: If ``directory`` does not exist or is not a directory.
    """
    folder = Path(directory).expanduser().resolve()

    if not folder.is_dir():
        raise FileNotFoundError(f"resume directory not found: {folder}")

    candidates: list[Candidate] = []
    failures: list[ResumeLoadFailure] = []

    for pdf_path in sorted(folder.glob("*.pdf")):
        try:
            text = extract_text_from_pdf(pdf_path)
            candidates.append(
                Candidate(
                    candidate_id=pdf_path.stem,
                    resume_text=text,
                    candidate_name=_candidate_name_from_filename(pdf_path),
                    source_path=pdf_path,
                )
            )
        except (ResumeParserError, InvalidCandidateError) as exc:
            failures.append(ResumeLoadFailure(pdf_path, str(exc)))

    return LoadedResumes(candidates, failures)
