"""API configuration, read from the environment.

Deliberately small: a frozen dataclass and one function that reads environment
variables. There is no settings framework, no config file format and no schema
registry, because the API needs five values and the project already uses plain
environment variables everywhere else.

============================  =================================================
``RESUME_DIR``                Directory scanned for candidate resumes
``EMBEDDING_MODEL``           Sentence Transformers model id
``API_CORS_ORIGINS``          Comma-separated allowed origins, or ``*``
``API_MAX_UPLOAD_BYTES``      Upload size ceiling for ``POST /upload-resume``
``API_LOG_LEVEL``             Log level for the ``app`` logger
============================  =================================================

``LLM_PROVIDER``, ``LLM_MODEL``, ``LLM_API_KEY`` and ``LLM_MAX_TOKENS`` are read
by :mod:`app.llm` exactly as they were in Phase 4; the API does not re-read or
re-validate them, and never returns a key or reports whether one is set.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from app.embeddings import DEFAULT_MODEL_NAME

__all__ = [
    "DEFAULT_RESUME_DIR",
    "DEFAULT_MAX_UPLOAD_BYTES",
    "DEFAULT_CORS_ORIGINS",
    "Settings",
    "load_settings",
]

DEFAULT_RESUME_DIR = Path("data/resumes")

# 5 MB. A text-based resume PDF is tens of kilobytes; anything approaching this
# is either scanned (unreadable here anyway) or not a resume.
DEFAULT_MAX_UPLOAD_BYTES = 5 * 1024 * 1024

# The Streamlit dashboard of Phase 6 runs on 8501 by default. Nothing else is
# allowed unless it is named explicitly.
DEFAULT_CORS_ORIGINS = ("http://localhost:8501", "http://127.0.0.1:8501")

DEFAULT_LOG_LEVEL = "INFO"


@dataclass(frozen=True, slots=True)
class Settings:
    """Everything the API layer needs to know about its environment.

    Attributes:
        resume_dir: Directory the candidate pool is loaded from. Server-side
            configuration only -- no endpoint accepts a path from a client.
        embedding_model: Sentence Transformers model id passed to the shared
            Phase 2 embedder.
        llm_provider: Configured provider *name*, reported by ``/health``. The
            provider itself is built by :func:`app.llm.get_llm_provider`; this
            string is for display and is never a credential.
        max_upload_bytes: Largest accepted upload.
        cors_origins: Allowed browser origins. ``("*",)`` means any origin and
            is development-only.
        log_level: Level applied to the ``app`` logger.
    """

    resume_dir: Path = DEFAULT_RESUME_DIR
    embedding_model: str = DEFAULT_MODEL_NAME
    llm_provider: str = "fake"
    max_upload_bytes: int = DEFAULT_MAX_UPLOAD_BYTES
    cors_origins: tuple[str, ...] = DEFAULT_CORS_ORIGINS
    log_level: str = DEFAULT_LOG_LEVEL

    @property
    def allows_any_origin(self) -> bool:
        """Whether CORS is wide open, which is only appropriate in development."""
        return "*" in self.cors_origins


def _int_from_env(name: str, default: int) -> int:
    """Read a positive integer from the environment, falling back on nonsense."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _origins_from_env(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    """Read a comma-separated origin list from the environment."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    origins = tuple(part.strip() for part in raw.split(",") if part.strip())
    return origins or default


def load_settings() -> Settings:
    """Build :class:`Settings` from the current environment.

    Returns:
        The settings. Missing or unparsable values fall back to the documented
        defaults rather than raising, so a typo in an optional variable cannot
        stop the server from starting.
    """
    resume_dir = os.environ.get("RESUME_DIR", "").strip()
    model = os.environ.get("EMBEDDING_MODEL", "").strip()
    provider = os.environ.get("LLM_PROVIDER", "").strip().lower()
    level = os.environ.get("API_LOG_LEVEL", "").strip().upper()

    return Settings(
        resume_dir=Path(resume_dir) if resume_dir else DEFAULT_RESUME_DIR,
        embedding_model=model or DEFAULT_MODEL_NAME,
        llm_provider=provider or "fake",
        max_upload_bytes=_int_from_env("API_MAX_UPLOAD_BYTES", DEFAULT_MAX_UPLOAD_BYTES),
        cors_origins=_origins_from_env("API_CORS_ORIGINS", DEFAULT_CORS_ORIGINS),
        log_level=level or DEFAULT_LOG_LEVEL,
    )
