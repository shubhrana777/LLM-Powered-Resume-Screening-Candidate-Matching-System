"""FastAPI application entry point.

Run it with::

    uvicorn app.api.main:app --reload

Interactive documentation is then at ``/docs`` (Swagger UI), ``/redoc``, and the
raw schema at ``/openapi.json``.

The application is built by :func:`create_app` rather than assembled at module
level, so a test can construct an isolated instance without inheriting whatever
the environment happens to hold. The module-level ``app`` exists because that is
what ``uvicorn`` imports.

This module wires; it does not implement. Routing lives in :mod:`app.api.routes`,
error translation in :mod:`app.api.errors`, and every screening behaviour in the
Phase 1-4 packages.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.api.config import Settings
from app.api.dependencies import get_settings
from app.api.errors import install_error_handlers
from app.api.routes import router

__all__ = ["API_TITLE", "API_DESCRIPTION", "create_app", "app"]

logger = logging.getLogger("app.api")

API_TITLE = "Resume Screening & Candidate Matching API"

API_DESCRIPTION = """
REST interface to an LLM-powered resume screening pipeline.

Upload and parse PDF resumes, rank a pool of candidates against a job
description by semantic similarity, and produce a candidate analysis grounded in
passages retrieved from that candidate's own resume.

### What the numbers mean

* `similarity_score` is **cosine similarity** between two embeddings, in
  `[-1.0, 1.0]`. It is a semantic similarity score -- not a probability of being
  hired, not a percentage of requirements met, and meaningful only as an ordering
  within one ranking.
* `recommendation` is a **coarse ordinal label** from a fixed vocabulary, not a
  score and not a hiring decision.

### Grounding

An analysis is assembled from the candidate's deterministic profile plus
passages retrieved from their resume, and the model's response is validated
against that profile before it is returned. Claims the resume does not support
are removed and listed in `warnings`; `is_grounded` is `false` whenever anything
had to be corrected. This reduces hallucination -- it does not eliminate it. The
`evidence` returned with every analysis is there to be read.

### Scope

There is no authentication: this is a portfolio project intended to be run
locally. Do not expose it to an untrusted network, and do not upload real
candidate data to a deployment you do not control.
"""

TAGS_METADATA = [
    {
        "name": "system",
        "description": "Liveness and service information.",
    },
    {
        "name": "resumes",
        "description": "Upload a resume for text extraction, and list the candidates available on the server.",
    },
    {
        "name": "matching",
        "description": "Rank candidates against a job description by semantic similarity.",
    },
    {
        "name": "analysis",
        "description": "Retrieval-augmented candidate analysis, validated against the resume.",
    },
]


def configure_logging(settings: Settings) -> None:
    """Give the ``app`` logger somewhere to write, without touching global config.

    ``logging.basicConfig`` would reconfigure the root logger for whatever else
    is running in the process, so instead a handler is attached to this
    project's own logger, and only when it has none.

    Args:
        settings: Configuration carrying the desired level.
    """
    app_logger = logging.getLogger("app")
    app_logger.setLevel(settings.log_level)

    if not app_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s")
        )
        app_logger.addHandler(handler)


def configure_cors(application: FastAPI, settings: Settings) -> None:
    """Allow the configured browser origins, and warn if that is all of them.

    Args:
        application: The application to configure.
        settings: Configuration carrying the origin list.
    """
    if settings.allows_any_origin:
        logger.warning(
            "CORS is open to any origin (API_CORS_ORIGINS='*'). "
            "This is a development-only setting; name the origins explicitly "
            "before running this anywhere else."
        )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the application.

    Args:
        settings: Configuration to use. Defaults to the process-wide settings
            read from the environment.

    Returns:
        A configured :class:`~fastapi.FastAPI` instance.
    """
    active = settings if settings is not None else get_settings()

    configure_logging(active)

    application = FastAPI(
        title=API_TITLE,
        description=API_DESCRIPTION,
        version=__version__,
        openapi_tags=TAGS_METADATA,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    configure_cors(application, active)
    install_error_handlers(application)
    application.include_router(router)

    logger.info(
        "API ready: resumes=%s, embedding model=%s, LLM provider=%s",
        active.resume_dir,
        active.embedding_model,
        active.llm_provider,
    )

    return application


app = create_app()
