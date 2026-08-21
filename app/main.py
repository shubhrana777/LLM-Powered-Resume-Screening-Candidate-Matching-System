"""Command-line entry point.

Two subcommands:

``extract`` (Phase 1)
    python -m app.main extract data/resumes/sample_resume.pdf

``match`` (Phase 2)
    python -m app.main match --job-description data/job_descriptions/backend_engineer.txt

``analyze`` (Phase 3)
    python -m app.main analyze --job-description data/job_descriptions/backend_engineer.txt

For backward compatibility the Phase 1 form without a subcommand still works and
is treated as ``extract``::

    python -m app.main data/resumes/sample_resume.pdf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app.candidate_analyzer import analyze_candidates_for_job
from app.embeddings import DEFAULT_MODEL_NAME, EmbeddingError, get_default_embedder
from app.matching import (
    CandidateMatcher,
    MatchingError,
    load_candidates_from_directory,
    validate_job_description,
)
from app.models import CandidateProfile, InvalidCandidateError, JobRequirements, MatchResult
from app.resume_parser import ResumeParserError, extract_text_from_pdf
from app.skill_taxonomy import TaxonomyError
from app.vector_store import VectorStoreError

EXIT_OK = 0
EXIT_ERROR = 1

DEFAULT_RESUME_DIR = Path("data/resumes")
SUBCOMMANDS = ("extract", "match", "analyze")

# Errors that represent a normal user mistake: report the message, never a traceback.
USER_ERRORS = (
    ResumeParserError,
    EmbeddingError,
    VectorStoreError,
    MatchingError,
    InvalidCandidateError,
    TaxonomyError,
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

    handlers = {"extract": run_extract, "match": run_match, "analyze": run_analyze}

    try:
        return handlers[args.command](args)
    except USER_ERRORS as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
