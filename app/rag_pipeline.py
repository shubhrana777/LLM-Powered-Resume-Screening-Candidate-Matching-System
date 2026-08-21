"""End-to-end retrieval-augmented candidate analysis.

Orchestration only. Every step delegates to an existing component:

======================================  ==================================================
Step                                    Owner
======================================  ==================================================
Load PDFs, extract text                 :mod:`app.resume_parser` (Phase 1)
Chunk resumes                           :mod:`app.chunker`
Embed chunks, index, retrieve           :mod:`app.embeddings`, :mod:`app.retriever` (Phase 2)
Analyse the job description             :mod:`app.candidate_analyzer` (Phase 3)
Build the candidate profile             :mod:`app.candidate_analyzer` (Phase 3)
Assemble grounded context               :mod:`app.rag_context`
Prompt and call the model               :mod:`app.prompts`, :mod:`app.llm`
Parse, validate, ground the response    :mod:`app.analysis_parser`
======================================  ==================================================

Why retrieval instead of sending the whole resume
-------------------------------------------------
Passing the full text would work for a one-page resume, but it defeats the
point: there would be no record of *which* passages support a conclusion, so a
reviewer could not check a claim without rereading the resume. Retrieval makes
the supporting passages an explicit, inspectable part of the output. It also
keeps the prompt bounded as resumes grow.

Retrieval runs several queries per candidate -- the job description plus each
required skill -- because a short mention of one requirement is easily outranked
by overall topical similarity when a single query is used.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from app.candidate_analyzer import analyze_job_description, build_candidate_profile
from app.chunker import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, chunk_candidates
from app.embeddings import TextEmbedder
from app.llm import LLMProvider, get_llm_provider
from app.matching import CandidateMatcher, EmptyCandidateListError
from app.models import Candidate, CandidateAnalysis, JobRequirements
from app.prompts import SYSTEM_PROMPT, build_analysis_prompt
from app.analysis_parser import parse_candidate_analysis
from app.rag_context import RagContext, build_rag_context
from app.retriever import DEFAULT_TOP_K, ChunkRetriever
from app.skill_extractor import DEFAULT_EXTRACTOR, SkillExtractor

__all__ = [
    "RagConfig",
    "RagPipeline",
    "UnknownCandidateError",
]

from app.retriever import UnknownCandidateError  # re-exported for callers


@dataclass(frozen=True, slots=True)
class RagConfig:
    """Tuning knobs for the pipeline.

    Attributes:
        chunk_size: Words per resume chunk.
        chunk_overlap: Words shared between neighbouring chunks.
        top_k: Passages of evidence to put in front of the model per candidate.
        top_k_per_query: Passages taken from each individual retrieval query.
        max_skill_queries: Cap on per-skill retrieval queries, so a job
            description listing many skills does not cause a burst of embeddings.
        per_skill_queries: Whether to run one retrieval query per required
            skill in addition to the job-description query.
    """

    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    top_k: int = DEFAULT_TOP_K
    top_k_per_query: int = 2
    max_skill_queries: int = 8
    per_skill_queries: bool = True


class RagPipeline:
    """Analyses candidates against a job description using retrieved evidence.

    Indexing is separate from analysis so one set of resumes can be analysed
    against several job descriptions without re-chunking or re-embedding.

    Args:
        embedder: Embedding component. Defaults to the shared Phase 2 embedder.
        llm: Language-model provider. Defaults to :func:`app.llm.get_llm_provider`,
            which falls back to the offline deterministic provider when nothing
            is configured.
        extractor: Skill extractor supplying the taxonomy.
        config: Chunking and retrieval settings.
    """

    def __init__(
        self,
        embedder: TextEmbedder | None = None,
        llm: LLMProvider | None = None,
        extractor: SkillExtractor = DEFAULT_EXTRACTOR,
        config: RagConfig | None = None,
    ) -> None:
        self._config = config or RagConfig()
        self._extractor = extractor
        self._llm = llm if llm is not None else get_llm_provider()
        self._retriever = ChunkRetriever(embedder=embedder)
        self._candidates: dict[str, Candidate] = {}

    @property
    def llm(self) -> LLMProvider:
        """The provider this pipeline calls."""
        return self._llm

    @property
    def retriever(self) -> ChunkRetriever:
        """The retriever holding the indexed chunks."""
        return self._retriever

    @property
    def config(self) -> RagConfig:
        """The active configuration."""
        return self._config

    @property
    def candidate_ids(self) -> tuple[str, ...]:
        """Candidates currently indexed, in input order."""
        return tuple(self._candidates)

    def index_candidates(self, candidates: Sequence[Candidate]) -> int:
        """Chunk and index candidate resumes, replacing anything indexed before.

        Args:
            candidates: Candidate records carrying parsed resume text.

        Returns:
            The number of chunks indexed.

        Raises:
            EmptyCandidateListError: If no candidates are supplied.
        """
        entries = list(candidates)
        if not entries:
            raise EmptyCandidateListError("no candidates supplied; nothing to index")

        chunks = chunk_candidates(
            entries,
            chunk_size=self._config.chunk_size,
            chunk_overlap=self._config.chunk_overlap,
        )
        self._retriever.index_chunks(chunks)
        self._candidates = {candidate.candidate_id: candidate for candidate in entries}

        return len(chunks)

    def _retrieval_queries(self, requirements: JobRequirements) -> list[str]:
        """Build the query set used to gather evidence for one candidate."""
        queries = [requirements.raw_text]

        if self._config.per_skill_queries:
            for skill in requirements.required_skills[: self._config.max_skill_queries]:
                queries.append(f"experience with {skill}")

        return queries

    def build_context(
        self,
        candidate_id: str,
        requirements: JobRequirements,
    ) -> RagContext:
        """Retrieve evidence and assemble the LLM context for one candidate.

        Args:
            candidate_id: Candidate to build context for.
            requirements: Structured job requirements.

        Returns:
            The assembled context, carrying the evidence it was built from.

        Raises:
            UnknownCandidateError: If the candidate was never indexed.
        """
        candidate = self._candidates.get(candidate_id)
        if candidate is None:
            raise UnknownCandidateError(
                f"candidate {candidate_id!r} is not indexed; "
                f"indexed candidates: {sorted(self._candidates) or 'none'}"
            )

        evidence = self._retriever.retrieve_for_queries(
            self._retrieval_queries(requirements),
            candidate_id=candidate_id,
            top_k_per_query=self._config.top_k_per_query,
            max_results=self._config.top_k,
        )

        profile = build_candidate_profile(
            candidate, requirements=requirements, extractor=self._extractor
        )

        return build_rag_context(profile, requirements, evidence)

    def analyze_candidate(
        self,
        candidate_id: str,
        job_description: str | JobRequirements,
    ) -> CandidateAnalysis:
        """Run the full pipeline for one candidate.

        Args:
            candidate_id: Candidate to analyse.
            job_description: Job-description text, or already-parsed requirements.

        Returns:
            The grounded :class:`~app.models.CandidateAnalysis`.

        Raises:
            UnknownCandidateError: If the candidate was never indexed.
            app.llm.LLMError: If the provider call fails.
            app.analysis_parser.AnalysisParseError: If the response is unreadable.
        """
        requirements = (
            job_description
            if isinstance(job_description, JobRequirements)
            else analyze_job_description(job_description, self._extractor)
        )

        context = self.build_context(candidate_id, requirements)
        prompt = build_analysis_prompt(context.text)
        raw = self._llm.generate(prompt, system=SYSTEM_PROMPT)

        return parse_candidate_analysis(
            raw,
            profile=context.profile,
            evidence=context.evidence,
            model_name=self._llm.name,
        )

    def analyze_all(
        self,
        job_description: str,
        top_k_candidates: int | None = None,
    ) -> tuple[JobRequirements, list[CandidateAnalysis]]:
        """Rank the indexed candidates, then analyse them in rank order.

        Semantic ranking comes from the Phase 2 matcher, so the ordering here is
        the same one ``match`` and ``analyze`` produce.

        Args:
            job_description: The job-description text.
            top_k_candidates: Analyse only the top N ranked candidates. Each
                analysis costs one model call, so this is the cost control.

        Returns:
            A ``(requirements, analyses)`` pair, analyses in rank order.

        Raises:
            EmptyCandidateListError: If nothing is indexed.
        """
        if not self._candidates:
            raise EmptyCandidateListError("no candidates indexed; call index_candidates() first")

        requirements = analyze_job_description(job_description, self._extractor)

        matcher = CandidateMatcher(embedder=self._retriever.embedder)
        matcher.index_candidates(list(self._candidates.values()))
        ranked = matcher.match(requirements.raw_text, top_k=top_k_candidates)

        return requirements, [
            self.analyze_candidate(result.candidate_id, requirements) for result in ranked
        ]
