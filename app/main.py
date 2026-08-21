"""Command-line entry point.

Two subcommands:

``extract`` (Phase 1)
    python -m app.main extract data/resumes/sample_resume.pdf

``match`` (Phase 2)
    python -m app.main match --job-description data/job_descriptions/backend_engineer.txt

For backward compatibility the Phase 1 form without a subcommand still works and
is treated as ``extract``::

    python -m app.main data/resumes/sample_resume.pdf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app.embeddings import DEFAULT_MODEL_NAME, EmbeddingError, get_default_embedder
from app.matching import (
    CandidateMatcher,
    MatchingError,
    load_candidates_from_directory,
    validate_job_description,
)
from app.models import InvalidCandidateError, MatchResult
from app.resume_parser import ResumeParserError, extract_text_from_pdf
from app.vector_store import VectorStoreError

EXIT_OK = 0
EXIT_ERROR = 1

DEFAULT_RESUME_DIR = Path("data/resumes")
SUBCOMMANDS = ("extract", "match")

# Errors that represent a normal user mistake: report the message, never a traceback.
USER_ERRORS = (
    ResumeParserError,
    EmbeddingError,
    VectorStoreError,
    MatchingError,
    InvalidCandidateError,
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
    match.add_argument(
        "-r",
        "--resumes",
        type=Path,
        default=DEFAULT_RESUME_DIR,
        help=f"Directory of PDF resumes to rank (default: {DEFAULT_RESUME_DIR}).",
    )
    source = match.add_mutually_exclusive_group(required=True)
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
    match.add_argument(
        "-k",
        "--top-k",
        type=int,
        default=None,
        help="Show only the top K candidates (default: all).",
    )
    match.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help=f"Sentence Transformers model to embed with (default: {DEFAULT_MODEL_NAME}).",
    )

    return parser


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

    matcher = CandidateMatcher(embedder=get_default_embedder(args.model))
    matcher.index_candidates(loaded.candidates)
    results = matcher.match(job_description, top_k=args.top_k)

    print(format_results(results))
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    """Run the CLI.

    Args:
        argv: Argument list to parse. Defaults to ``sys.argv[1:]``.

    Returns:
        ``0`` on success, ``1`` on a handled user error.
    """
    raw = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(_normalize_argv(raw))

    handlers = {"extract": run_extract, "match": run_match}

    try:
        return handlers[args.command](args)
    except USER_ERRORS as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
