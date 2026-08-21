"""Command-line entry point for the Phase 1 resume text extractor.

Usage:
    python -m app.main data/resumes/sample_resume.pdf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app.resume_parser import ResumeParserError, extract_text_from_pdf

EXIT_OK = 0
EXIT_ERROR = 1


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m app.main",
        description="Extract clean text from a PDF resume.",
    )
    parser.add_argument(
        "resume",
        type=Path,
        help="Path to a PDF resume, e.g. data/resumes/sample_resume.pdf",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Write the extracted text to this file instead of standard output.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI.

    Args:
        argv: Argument list to parse. Defaults to ``sys.argv[1:]``.

    Returns:
        ``0`` on success, ``1`` when the resume could not be parsed.
    """
    args = build_parser().parse_args(argv)

    try:
        text = extract_text_from_pdf(args.resume)
    except ResumeParserError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return EXIT_ERROR

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"Wrote {len(text)} characters to {args.output}")
    else:
        print(text)

    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
