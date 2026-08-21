"""Command-line entry point.

Two subcommands:

``extract`` (Phase 1)
    python -m app.main extract data/resumes/sample_resume.pdf

``match`` (Phase 2)
    python -m app.main match --job-description data/job_descriptions/backend_engineer.txt

``analyze`` (Phase 3)
    python -m app.main analyze --job-description data/job_descriptions/backend_engineer.txt

``rag`` (Phase 4)
    python -m app.main rag --job-description data/job_descriptions/backend_engineer.txt

For backward compatibility the Phase 1 form without a subcommand still works and
is treated as ``extract``::

    python -m app.main data/resumes/sample_resume.pdf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app.analysis_parser import AnalysisParseError
from app.candidate_analyzer import analyze_candidates_for_job, analyze_job_description
from app.chunker import ChunkingError
from app.embeddings import DEFAULT_MODEL_NAME, EmbeddingError, get_default_embedder
from app.llm import LLMError, get_llm_provider
from app.matching import (
    CandidateMatcher,
    MatchingError,
    load_candidates_from_directory,
    validate_job_description,
)
from app.models import (
    CandidateAnalysis,
    CandidateProfile,
    InvalidCandidateError,
    JobRequirements,
    MatchResult,
)
from app.rag_context import ContextIsolationError
from app.rag_pipeline import RagConfig, RagPipeline
from app.resume_parser import ResumeParserError, extract_text_from_pdf
from app.retriever import RetrievalError
from app.skill_taxonomy import TaxonomyError
from app.vector_store import VectorStoreError

EXIT_OK = 0
EXIT_ERROR = 1

DEFAULT_RESUME_DIR = Path("data/resumes")
# An instance, not the class: RagConfig uses slots, so class attribute access
# yields the slot descriptor rather than the default value.
RAG_DEFAULTS = RagConfig()
SUBCOMMANDS = ("extract", "match", "analyze", "rag")

# Errors that represent a normal user mistake: report the message, never a traceback.
USER_ERRORS = (
    ResumeParserError,
    EmbeddingError,
    VectorStoreError,
    MatchingError,
    InvalidCandidateError,
    TaxonomyError,
    LLMError,
    AnalysisParseError,
    ChunkingError,
    RetrievalError,
    ContextIsolationError,
    FileNotFoundError,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m app.main",
        description="Extract text from PDF resumes and rank candidates against a job description.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract = subparsers.add_parser(
        "extract",
        help="Extract clean text from a single PDF resume (Phase 1).",
        description="Extract clean text from a single PDF resume.",
    )
    extract.add_argument(
        "resume",
        type=Path,
        help="Path to a PDF resume, e.g. data/resumes/sample_resume.pdf",
    )
    extract.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Write the extracted text to this file instead of standard output.",
    )

    match = subparsers.add_parser(
        "match",
        help="Rank resumes against a job description by semantic similarity (Phase 2).",
        description=(
            "Rank PDF resumes against a job description using sentence-embedding "
            "cosine similarity."
        ),
    )
    _add_job_arguments(match)

    analyze = subparsers.add_parser(
        "analyze",
        help="Rank resumes and report skills, experience and education (Phase 3).",
        description=(
            "Rank PDF resumes against a job description, then report matched and "
            "missing skills, stated experience, and education for each candidate."
        ),
    )
    _add_job_arguments(analyze)
    analyze.add_argument(
        "--show-extra-skills",
        action="store_true",
        help="Also list candidate skills the job description did not ask for.",
    )

    rag = subparsers.add_parser(
        "rag",
        help="LLM analysis grounded in retrieved resume evidence (Phase 4).",
        description=(
            "Chunk and index resumes, retrieve the passages relevant to the job "
            "description, and ask an LLM to analyse each candidate using only that "
            "evidence. Uses the offline deterministic provider unless LLM_PROVIDER "
            "says otherwise, so it runs without an API key."
        ),
    )
    _add_job_arguments(rag)
    rag.add_argument(
        "--llm-provider",
        type=str,
        default=None,
        help="Override LLM_PROVIDER for this run (fake or anthropic).",
    )
    rag.add_argument(
        "--llm-model",
        type=str,
        default=None,
        help="Override LLM_MODEL for this run.",
    )
    rag.add_argument(
        "--candidate",
        type=str,
        default=None,
        help="Analyse only this candidate id (the resume file stem).",
    )
    rag.add_argument(
        "--chunk-size",
        type=int,
        default=RAG_DEFAULTS.chunk_size,
        help=f"Words per resume chunk (default: {RAG_DEFAULTS.chunk_size}).",
    )
    rag.add_argument(
        "--chunk-overlap",
        type=int,
        default=RAG_DEFAULTS.chunk_overlap,
        help=f"Words shared between chunks (default: {RAG_DEFAULTS.chunk_overlap}).",
    )
    rag.add_argument(
        "--evidence-k",
        type=int,
        default=RAG_DEFAULTS.top_k,
        help=f"Resume passages given to the model (default: {RAG_DEFAULTS.top_k}).",
    )
    rag.add_argument(
        "--hide-evidence",
        action="store_true",
        help="Omit the supporting passages from the report.",
    )

    return parser


def _add_job_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the resume-folder and job-description arguments shared by subcommands.

    Args:
        parser: The ``match`` or ``analyze`` subparser to populate.
    """
    parser.add_argument(
        "-r",
        "--resumes",
        type=Path,
        default=DEFAULT_RESUME_DIR,
        help=f"Directory of PDF resumes to rank (default: {DEFAULT_RESUME_DIR}).",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "-j",
        "--job-description",
        type=Path,
        default=None,
        help="Path to a text file containing the job description.",
    )
    source.add_argument(
        "-t",
        "--job-text",
        type=str,
        default=None,
        help="Job description supplied directly on the command line.",
    )
    parser.add_argument(
        "-k",
        "--top-k",
        type=int,
        default=None,
        help="Show only the top K candidates (default: all).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help=f"Sentence Transformers model to embed with (default: {DEFAULT_MODEL_NAME}).",
    )


def _normalize_argv(argv: list[str]) -> list[str]:
    """Insert the implicit ``extract`` subcommand for the Phase 1 CLI form.

    ``python -m app.main resume.pdf`` predates subcommands and must keep working,
    so a leading token that is neither a subcommand nor an option is treated as
    the argument to ``extract``.

    Args:
        argv: Raw argument list.

    Returns:
        The argument list, with ``extract`` prepended when appropriate.
    """
    if argv and argv[0] not in SUBCOMMANDS and not argv[0].startswith("-"):
        return ["extract", *argv]
    return argv


def _read_job_description(args: argparse.Namespace) -> str:
    """Read the job description from a file or from the inline argument.

    Args:
        args: Parsed ``match`` arguments.

    Returns:
        The job description text.

    Raises:
        FileNotFoundError: If the job-description file does not exist.
    """
    if args.job_text is not None:
        return args.job_text

    path: Path = args.job_description
    if not path.is_file():
        raise FileNotFoundError(f"job description file not found: {path}")

    return path.read_text(encoding="utf-8")


def format_results(results: list[MatchResult]) -> str:
    """Render ranked results as a readable fixed-width table.

    Args:
        results: Ranked match results, best first.

    Returns:
        The table as a string, including a header and a scoring caveat.
    """
    if not results:
        return "No candidates matched."

    name_width = max(len("Candidate"), *(len(r.display_name) for r in results))

    lines = [
        f"{'Rank':>4}  {'Candidate':<{name_width}}  {'Similarity':>10}",
        f"{'-' * 4}  {'-' * name_width}  {'-' * 10}",
    ]
    lines.extend(
        f"{r.rank:>4}  {r.display_name:<{name_width}}  {r.similarity_score:>10.4f}"
        for r in results
    )
    lines.append("")
    lines.append(
        "Similarity is the cosine similarity between the job-description and resume "
        "embeddings, in [-1, 1]."
    )
    lines.append(
        "It reflects semantic closeness of the text only. It is not a probability of "
        "being hired and"
    )
    lines.append(
        "not a measure of candidate quality; only the relative ordering within this "
        "run is meaningful."
    )
    return "\n".join(lines)


def _format_experience(profile: CandidateProfile) -> list[str]:
    """Render the experience block for one candidate profile."""
    candidate = (
        "not stated on resume"
        if profile.years_experience is None
        else f"{profile.years_experience:g} years"
    )
    required = (
        "not stated in job description"
        if profile.required_experience is None
        else f"{profile.required_experience:g} years"
    )

    verdict = {True: "Yes", False: "No", None: "Unknown"}[profile.meets_experience_requirement]

    lines = [
        "  Experience:",
        f"    Candidate: {candidate}",
        f"    Required:  {required}",
        f"    Requirement Met: {verdict}",
    ]
    if profile.meets_experience_requirement is None:
        lines.append("      (unknown is reported as unknown, not as a pass or a fail)")
    return lines


def format_profile(profile: CandidateProfile, show_extra_skills: bool = False) -> str:
    """Render one candidate profile as a readable block.

    Args:
        profile: The profile to render.
        show_extra_skills: Whether to list skills the job did not ask for.

    Returns:
        The formatted block, without a trailing newline.
    """
    rank = "" if profile.rank is None else f"{profile.rank}. "
    lines = [f"{rank}{profile.display_name}"]

    if profile.semantic_match_score is not None:
        lines.append(f"  Semantic Match Score: {profile.semantic_match_score:.4f}")

    def skill_block(title: str, skills: tuple[str, ...]) -> list[str]:
        if not skills:
            return [f"  {title}: none"]
        return [f"  {title}:", *(f"    - {skill}" for skill in skills)]

    lines.extend(skill_block("Matched Skills", profile.matched_skills))
    lines.extend(skill_block("Missing Skills", profile.missing_skills))

    if show_extra_skills:
        lines.extend(skill_block("Additional Skills", profile.additional_skills))

    coverage = profile.skill_comparison.coverage
    if coverage is not None:
        matched = len(profile.matched_skills)
        total = profile.skill_comparison.required_count
        lines.append(f"  Skill Coverage: {matched}/{total} required skills ({coverage:.0%})")

    lines.extend(_format_experience(profile))

    if profile.education:
        lines.append("  Education:")
        lines.extend(f"    {entry}" for entry in profile.education)
    else:
        lines.append("  Education: not found")

    return "\n".join(lines)


def format_analysis(
    requirements: JobRequirements,
    profiles: list[CandidateProfile],
    show_extra_skills: bool = False,
) -> str:
    """Render the full Phase 3 analysis report.

    Args:
        requirements: The structured job requirements.
        profiles: Candidate profiles, best match first.
        show_extra_skills: Whether to list skills the job did not ask for.

    Returns:
        The formatted report.
    """
    required = ", ".join(requirements.required_skills) or "none recognised"
    minimum = (
        "not stated"
        if requirements.minimum_experience is None
        else f"{requirements.minimum_experience:g} years"
    )

    lines = [
        "JOB REQUIREMENTS",
        f"  Required Skills: {required}",
        f"  Minimum Experience: {minimum}",
        "",
        "CANDIDATES",
    ]

    if not profiles:
        lines.append("  No candidates analysed.")
    else:
        for profile in profiles:
            lines.append("")
            lines.append(format_profile(profile, show_extra_skills=show_extra_skills))

    lines.extend(
        [
            "",
            "Skills, experience and education are extracted deterministically by exact",
            "matching against a fixed vocabulary, so anything phrased unusually or absent",
            "from the taxonomy is simply not reported. The semantic match score is a cosine",
            "similarity, not a probability of being hired.",
        ]
    )
    return "\n".join(lines)


def format_candidate_analysis(
    analysis: CandidateAnalysis,
    show_evidence: bool = True,
) -> str:
    """Render one LLM candidate analysis as a readable block.

    Args:
        analysis: The grounded analysis to render.
        show_evidence: Whether to include the supporting resume passages.

    Returns:
        The formatted block, without a trailing newline.
    """
    lines = [
        f"Candidate: {analysis.display_name}",
        f"Recommendation: {analysis.recommendation.value}",
        "",
        "Summary:",
        f"  {analysis.summary}",
        "",
    ]

    def skill_block(title: str, skills: tuple[str, ...]) -> list[str]:
        if not skills:
            return [f"{title}: none"]
        return [f"{title}:", *(f"  - {skill}" for skill in skills)]

    lines.extend(skill_block("Matched Skills", analysis.matched_skills))
    lines.append("")
    lines.extend(skill_block("Skill Gaps", analysis.skill_gaps))
    lines.extend(["", "Experience:", f"  {analysis.experience_assessment}"])

    if show_evidence:
        lines.extend(["", "Evidence (verbatim resume passages given to the model):"])
        if not analysis.evidence:
            lines.append("  none retrieved")
        for item in analysis.evidence:
            excerpt = " ".join(getattr(item, "text", str(item)).split())
            if len(excerpt) > 220:
                excerpt = excerpt[:220].rstrip() + " ..."
            chunk_id = getattr(item, "chunk_id", "?")
            score = getattr(item, "retrieval_score", None)
            score_text = "" if score is None else f", similarity={score:.4f}"
            lines.append(f"  - [{chunk_id}{score_text}]")
            lines.append(f"    {excerpt}")

    if analysis.limitations:
        lines.append("")
        lines.append("Limitations:")
        lines.extend(f"  - {item}" for item in analysis.limitations)

    if analysis.warnings:
        lines.append("")
        lines.append("Grounding warnings (unsupported claims corrected during validation):")
        lines.extend(f"  ! {item}" for item in analysis.warnings)

    return "\n".join(lines)


def format_rag_report(
    requirements: JobRequirements,
    analyses: list[CandidateAnalysis],
    model_name: str,
    show_evidence: bool = True,
) -> str:
    """Render the full Phase 4 report.

    Args:
        requirements: Structured job requirements.
        analyses: Candidate analyses, best match first.
        model_name: Provider and model that produced the analyses.
        show_evidence: Whether to include supporting passages.

    Returns:
        The formatted report.
    """
    required = ", ".join(requirements.required_skills) or "none recognised"
    minimum = (
        "not stated"
        if requirements.minimum_experience is None
        else f"{requirements.minimum_experience:g} years"
    )

    lines = [
        "JOB REQUIREMENTS",
        f"  Required Skills: {required}",
        f"  Minimum Experience: {minimum}",
        "",
        f"LLM: {model_name}",
        "",
        "=" * 72,
    ]

    if not analyses:
        lines.append("No candidates analysed.")
    else:
        for analysis in analyses:
            lines.append("")
            lines.append(format_candidate_analysis(analysis, show_evidence=show_evidence))
            lines.append("")
            lines.append("-" * 72)

    lines.extend(
        [
            "",
            "Each analysis is generated from the job description, the deterministic",
            "candidate profile, and the retrieved passages shown above -- nothing else.",
            "Claims are checked against the profile after generation and unsupported ones",
            "are removed, but this reduces hallucination rather than eliminating it.",
            "Read the evidence before acting on any conclusion.",
        ]
    )
    return "\n".join(lines)


def run_extract(args: argparse.Namespace) -> int:
    """Run the ``extract`` subcommand.

    Args:
        args: Parsed arguments.

    Returns:
        :data:`EXIT_OK` on success.
    """
    text = extract_text_from_pdf(args.resume)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"Wrote {len(text)} characters to {args.output}")
    else:
        print(text)

    return EXIT_OK


def run_match(args: argparse.Namespace) -> int:
    """Run the ``match`` subcommand.

    Args:
        args: Parsed arguments.

    Returns:
        :data:`EXIT_OK` on success.

    Raises:
        FileNotFoundError: If the resume directory or job-description file is missing.
        MatchingError: If there is nothing to rank.
    """
    job_description, candidates = _prepare_job_and_resumes(args)

    matcher = CandidateMatcher(embedder=get_default_embedder(args.model))
    matcher.index_candidates(candidates)
    results = matcher.match(job_description, top_k=args.top_k)

    print(format_results(results))
    return EXIT_OK


def run_analyze(args: argparse.Namespace) -> int:
    """Run the ``analyze`` subcommand.

    Args:
        args: Parsed arguments.

    Returns:
        :data:`EXIT_OK` on success.

    Raises:
        FileNotFoundError: If the resume directory or job-description file is missing.
        MatchingError: If there is nothing to rank.
    """
    job_description, candidates = _prepare_job_and_resumes(args)

    requirements, profiles = analyze_candidates_for_job(
        candidates,
        job_description,
        top_k=args.top_k,
        embedder=get_default_embedder(args.model),
    )

    print(format_analysis(requirements, profiles, show_extra_skills=args.show_extra_skills))
    return EXIT_OK


def run_rag(args: argparse.Namespace) -> int:
    """Run the ``rag`` subcommand.

    Args:
        args: Parsed arguments.

    Returns:
        :data:`EXIT_OK` on success.

    Raises:
        FileNotFoundError: If the resume directory or job-description file is missing.
        app.llm.LLMError: If the provider is misconfigured or the call fails.
    """
    job_description, candidates = _prepare_job_and_resumes(args)

    provider = get_llm_provider(provider=args.llm_provider, model=args.llm_model)
    config = RagConfig(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        top_k=args.evidence_k,
    )
    pipeline = RagPipeline(
        embedder=get_default_embedder(args.model),
        llm=provider,
        config=config,
    )

    chunk_count = pipeline.index_candidates(candidates)
    print(
        f"Indexed {chunk_count} chunk(s) from {len(candidates)} resume(s); "
        f"generating with {provider.name} ...",
        file=sys.stderr,
    )

    if args.candidate:
        requirements = analyze_job_description(job_description)
        analyses = [pipeline.analyze_candidate(args.candidate, requirements)]
    else:
        requirements, analyses = pipeline.analyze_all(
            job_description, top_k_candidates=args.top_k
        )

    print(
        format_rag_report(
            requirements,
            analyses,
            model_name=provider.name,
            show_evidence=not args.hide_evidence,
        )
    )
    return EXIT_OK


def _prepare_job_and_resumes(args: argparse.Namespace) -> tuple[str, list]:
    """Read the job description and load the resume folder, with user-facing warnings.

    Shared by ``match`` and ``analyze``.

    Args:
        args: Parsed arguments carrying ``resumes``, the job-description source,
            and ``model``.

    Returns:
        A ``(job_description, candidates)`` pair.

    Raises:
        FileNotFoundError: If the resume directory or job-description file is
            missing, or the directory holds no readable PDF.
        app.matching.EmptyJobDescriptionError: If the job description is empty.
    """
    # Validate first: embedding a folder of resumes is the expensive step, and
    # there is no point paying for it when the query is unusable.
    job_description = validate_job_description(_read_job_description(args))

    loaded = load_candidates_from_directory(args.resumes)
    for failure in loaded.failures:
        print(f"Warning: skipped {failure.path.name}: {failure.reason}", file=sys.stderr)

    if not loaded.candidates:
        raise FileNotFoundError(
            f"no readable PDF resumes found in {Path(args.resumes).resolve()}"
        )

    print(
        f"Embedding {len(loaded.candidates)} resume(s) with {args.model} ...",
        file=sys.stderr,
    )
    return job_description, loaded.candidates


def main(argv: list[str] | None = None) -> int:
    """Run the CLI.

    Args:
        argv: Argument list to parse. Defaults to ``sys.argv[1:]``.

    Returns:
        ``0`` on success, ``1`` on a handled user error.
    """
    raw = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(_normalize_argv(raw))

    handlers = {
        "extract": run_extract,
        "match": run_match,
        "analyze": run_analyze,
        "rag": run_rag,
    }

    try:
        return handlers[args.command](args)
    except USER_ERRORS as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
