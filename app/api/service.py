"""Application service: cached access to the existing screening components.

This layer exists for one reason -- **reuse without repeated cost**. The Phase
1-4 components are already complete and are called here exactly as the CLI
calls them; what the API adds is that a resume pool is parsed once, embedded
once and chunked once, instead of on every request.

Nothing here parses a PDF, computes a similarity, extracts a skill, retrieves a
passage or talks to a model. Each of those is delegated:

=====================================  ==========================================
Load and parse resumes                 :func:`app.matching.load_candidates_from_directory`
Rank against a job description         :class:`app.matching.CandidateMatcher`
Analyse a candidate                    :class:`app.rag_pipeline.RagPipeline`
=====================================  ==========================================

Cache invalidation
------------------
The pool is keyed by a signature over the directory listing -- each PDF's name,
size and modification time. Adding, removing or editing a resume changes the
signature, and the next request rebuilds the index. This keeps the API correct
when the directory changes underneath it without needing a restart, a watcher
or a cache-busting endpoint.

Concurrency
-----------
Route functions are synchronous, so FastAPI runs them in a worker thread and
several can be in flight at once. Building the index is guarded by a re-entrant
lock so two simultaneous first requests cannot both pay to embed the pool.
"""

from __future__ import annotations

import logging
import re
import shutil
import threading
from dataclasses import dataclass
from pathlib import Path

from app.api.config import Settings
from app.api.errors import BadRequestError, NotFoundError
from app.embeddings import TextEmbedder, get_default_embedder
from app.llm import LLMProvider, get_llm_provider
from app.matching import (
    CandidateMatcher,
    ResumeLoadFailure,
    load_candidates_from_directory,
)
from app.models import Candidate, CandidateAnalysis, MatchResult
from app.rag_pipeline import RagPipeline
from app.resume_parser import PDF_SUFFIX

__all__ = [
    "CandidatePool",
    "ScreeningService",
    "scrub_path",
    "slugify_candidate_id",
]

logger = logging.getLogger(__name__)

DirectorySignature = tuple[tuple[str, int, int], ...]

# Anything outside this set is replaced when a submitted file name is turned
# into a candidate id, so a client-supplied name can never shape a path.
_UNSAFE_IN_SLUG = re.compile(r"[^a-z0-9]+")
MAX_SLUG_LENGTH = 64
FALLBACK_SLUG = "resume"


def slugify_candidate_id(filename: str) -> str:
    """Reduce a submitted file name to a safe candidate id.

    ``"Sarah Wilson (CV).pdf"`` becomes ``"sarah_wilson_cv"``. The result
    contains only ``[a-z0-9_]``, so it cannot express a path, a parent
    directory, a drive letter or a device name.

    Args:
        filename: The client-supplied file name.

    Returns:
        A non-empty slug, at most :data:`MAX_SLUG_LENGTH` characters.
    """
    stem = Path(filename).name
    if stem.lower().endswith(PDF_SUFFIX):
        stem = stem[: -len(PDF_SUFFIX)]

    slug = _UNSAFE_IN_SLUG.sub("_", stem.casefold()).strip("_")
    return (slug[:MAX_SLUG_LENGTH].strip("_") or FALLBACK_SLUG)


def scrub_path(message: str, path: Path) -> str:
    """Remove a filesystem path from a message, leaving the file name.

    Parser errors embed the absolute path they were given. That is useful in a
    log and wrong in an HTTP response, so it is replaced before the message
    reaches a client.

    Args:
        message: The message to clean.
        path: The path that may appear in it.

    Returns:
        The message with the full path replaced by its file name.
    """
    for text in (str(path), str(path.parent)):
        message = message.replace(text, path.name if text == str(path) else "<resume-dir>")
    return message


@dataclass(frozen=True, slots=True)
class CandidatePool:
    """The parsed contents of the resume directory at one point in time.

    Attributes:
        directory: The directory that was scanned.
        candidates: Candidates parsed successfully.
        failures: Files that could not be parsed, with the reason.
        signature: Directory fingerprint this pool was built from.
    """

    directory: Path
    candidates: tuple[Candidate, ...]
    failures: tuple[ResumeLoadFailure, ...]
    signature: DirectorySignature


class ScreeningService:
    """Caches the resume pool and the indexes built from it.

    Args:
        settings: API configuration.
        embedder: Embedding component. Defaults to the shared, memoized Phase 2
            embedder, which loads the transformer model once per process.
            Injectable so tests never download model weights.
        llm: Language-model provider. Defaults to
            :func:`app.llm.get_llm_provider`, which returns the offline
            deterministic provider unless one is configured.
    """

    def __init__(
        self,
        settings: Settings,
        embedder: TextEmbedder | None = None,
        llm: LLMProvider | None = None,
    ) -> None:
        self._settings = settings
        self._embedder = embedder
        self._llm = llm
        self._lock = threading.RLock()
        self._pool: CandidatePool | None = None
        self._matcher: CandidateMatcher | None = None
        self._pipeline: RagPipeline | None = None

    @property
    def settings(self) -> Settings:
        """The configuration this service was built with."""
        return self._settings

    @property
    def resume_directory(self) -> Path:
        """The directory candidates are loaded from."""
        return self._settings.resume_dir

    def _embedder_instance(self) -> TextEmbedder:
        """The embedder to use, resolving the shared default on first need."""
        if self._embedder is None:
            self._embedder = get_default_embedder(self._settings.embedding_model)
        return self._embedder

    def _llm_instance(self) -> LLMProvider:
        """The provider to use, built from configuration on first need."""
        if self._llm is None:
            self._llm = get_llm_provider()
        return self._llm

    @staticmethod
    def _signature(directory: Path) -> DirectorySignature:
        """Fingerprint a directory's PDFs by name, size and modification time."""
        entries = []
        for pdf_path in sorted(directory.glob("*.pdf")):
            try:
                stat = pdf_path.stat()
            except OSError:  # removed between listing and stat
                continue
            entries.append((pdf_path.name, stat.st_size, stat.st_mtime_ns))
        return tuple(entries)

    def load_pool(self) -> CandidatePool:
        """Return the candidate pool, rebuilding it when the directory changed.

        Returns:
            The current :class:`CandidatePool`. It may hold no candidates: an
            empty directory is a valid state, not an error.

        Raises:
            NotFoundError: If the configured directory does not exist.
        """
        directory = self._settings.resume_dir
        if not directory.is_dir():
            raise NotFoundError(
                "The configured resume directory does not exist on the server.",
                code="resume_directory_not_found",
            )

        signature = self._signature(directory)

        with self._lock:
            if self._pool is not None and self._pool.signature == signature:
                return self._pool

            loaded = load_candidates_from_directory(directory)
            for failure in loaded.failures:
                logger.warning(
                    "Skipped unreadable resume %s: %s",
                    failure.path.name,
                    scrub_path(failure.reason, failure.path),
                )

            self._pool = CandidatePool(
                directory=directory,
                candidates=tuple(loaded.candidates),
                failures=tuple(loaded.failures),
                signature=signature,
            )
            # The indexes were built from the previous contents.
            self._matcher = None
            self._pipeline = None

            logger.info(
                "Loaded %d resume(s) from %s (%d unreadable)",
                len(self._pool.candidates),
                directory.name,
                len(self._pool.failures),
            )
            return self._pool

    def require_pool(self) -> CandidatePool:
        """Return the pool, insisting that it holds at least one candidate.

        Returns:
            A non-empty :class:`CandidatePool`.

        Raises:
            NotFoundError: If the directory is missing, or holds no readable PDF.
        """
        pool = self.load_pool()
        if not pool.candidates:
            raise NotFoundError(
                "No readable PDF resumes are available on the server.",
                code="no_resumes",
            )
        return pool

    def matcher(self) -> CandidateMatcher:
        """Return a matcher with the current pool indexed.

        The index is built once per pool version, so repeated match requests
        re-embed only the job description.
        """
        pool = self.require_pool()
        with self._lock:
            if self._matcher is None:
                logger.info("Embedding %d resume(s) for matching", len(pool.candidates))
                matcher = CandidateMatcher(embedder=self._embedder_instance())
                matcher.index_candidates(list(pool.candidates))
                self._matcher = matcher
            return self._matcher

    def pipeline(self) -> RagPipeline:
        """Return a RAG pipeline with the current pool chunked and indexed."""
        pool = self.require_pool()
        with self._lock:
            if self._pipeline is None:
                pipeline = RagPipeline(
                    embedder=self._embedder_instance(),
                    llm=self._llm_instance(),
                )
                chunk_count = pipeline.index_candidates(list(pool.candidates))
                logger.info(
                    "Indexed %d chunk(s) from %d resume(s) for retrieval",
                    chunk_count,
                    len(pool.candidates),
                )
                self._pipeline = pipeline
            return self._pipeline

    def resolve_candidate(self, reference: str) -> Candidate:
        """Find a pooled candidate by id, display name or file name.

        The reference is matched against candidates already loaded from the
        server's own directory. It is never treated as a path, so it cannot
        reach a file the pool does not contain.

        Args:
            reference: Candidate id, display name, or resume file name.

        Returns:
            The matching :class:`~app.models.Candidate`.

        Raises:
            NotFoundError: If nothing in the pool matches.
        """
        pool = self.require_pool()
        key = reference.strip().casefold()

        for candidate in pool.candidates:
            if candidate.candidate_id.casefold() == key:
                return candidate

        for candidate in pool.candidates:
            if (candidate.candidate_name or "").casefold() == key:
                return candidate

        for candidate in pool.candidates:
            if candidate.source_path is not None and candidate.source_path.name.casefold() == key:
                return candidate

        known = ", ".join(candidate.candidate_id for candidate in pool.candidates[:10])
        suffix = ", ..." if len(pool.candidates) > 10 else ""
        raise NotFoundError(
            f"No candidate matching {reference!r}. Known candidates: {known}{suffix}",
            code="candidate_not_found",
        )

    def match(self, job_description: str, top_k: int) -> tuple[list[MatchResult], int]:
        """Rank the pool against a job description.

        Args:
            job_description: The job description text.
            top_k: Maximum number of results.

        Returns:
            A ``(results, considered)`` pair, where ``considered`` is how many
            resumes were in the index.

        Raises:
            NotFoundError: If no resumes are available.
        """
        matcher = self.matcher()
        results = matcher.match(job_description, top_k=top_k)
        return results, matcher.candidate_count

    def analyze(self, reference: str, job_description: str) -> CandidateAnalysis:
        """Run the Phase 3 + Phase 4 analysis for one pooled candidate.

        The candidate is resolved against the pool first, so the profile and the
        evidence always come from the resume on disk. A client cannot supply
        skills, experience or evidence, and therefore cannot route around the
        grounding checks in :mod:`app.analysis_parser`.

        Args:
            reference: Candidate id, display name, or resume file name.
            job_description: The job description text.

        Returns:
            The validated :class:`~app.models.CandidateAnalysis`.

        Raises:
            NotFoundError: If the candidate is not in the pool.
            app.llm.LLMError: If the provider call fails.
            app.analysis_parser.AnalysisParseError: If the response is unusable.
        """
        candidate = self.resolve_candidate(reference)
        pipeline = self.pipeline()
        return pipeline.analyze_candidate(candidate.candidate_id, job_description)

    def store_resume(self, source: Path, original_filename: str) -> Candidate:
        """Copy a validated resume into the pool directory and return its candidate.

        This is what turns an upload into something that can be ranked. The
        caller has already validated that ``source`` is a readable PDF; this
        method only decides where it lands and refreshes the pool.

        The destination name is **generated**, never taken from the client: the
        submitted name is reduced to a slug of ``[a-z0-9_]``, which also becomes
        the candidate id. Re-uploading a file whose name slugs the same replaces
        the earlier one, because in this system the slug *is* the candidate's
        identity.

        Args:
            source: Path to the already-validated PDF.
            original_filename: Client-supplied name, used only to derive a slug.

        Returns:
            The stored :class:`~app.models.Candidate`, read back from the pool.

        Raises:
            NotFoundError: If the pool directory does not exist, or the stored
                file cannot be read back.
            BadRequestError: If the destination would fall outside the pool
                directory.
        """
        directory = self._settings.resume_dir
        if not directory.is_dir():
            raise NotFoundError(
                "The configured resume directory does not exist on the server.",
                code="resume_directory_not_found",
            )

        candidate_id = slugify_candidate_id(original_filename)
        destination = directory / f"{candidate_id}{PDF_SUFFIX}"

        # Defence in depth: the slug cannot contain a separator, but the check
        # costs nothing and would catch a future change to the slug rules.
        if destination.resolve().parent != directory.resolve():
            raise BadRequestError(
                "The resume could not be stored under that name.",
                code="invalid_filename",
            )

        with self._lock:
            shutil.copyfile(source, destination)
            self._pool = None
            self._matcher = None
            self._pipeline = None

        logger.info("Stored resume as %s", destination.name)

        for candidate in self.load_pool().candidates:
            if candidate.candidate_id == candidate_id:
                return candidate

        raise NotFoundError(
            "The resume was stored but could not be read back.",
            code="storage_failed",
        )
