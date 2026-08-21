"""Dashboard configuration, read from the environment.

The same approach as :mod:`app.api.config`: a frozen dataclass and one function.
The dashboard needs four values, so there is no settings framework here either.

===============================  ==============================================
``API_BASE_URL``                 Where the FastAPI backend is listening
``API_TIMEOUT_SECONDS``          Timeout for ordinary requests
``API_ANALYSIS_TIMEOUT_SECONDS`` Timeout for analysis, which calls a model
``UI_DEFAULT_TOP_K``             How many candidates the ranking asks for
===============================  ==============================================

No filesystem path is configured here on purpose. The dashboard never reads the
resume directory: it uploads through the API and reads candidates back from it,
so it works unchanged when the backend runs on another host.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

__all__ = [
    "DEFAULT_API_BASE_URL",
    "DEFAULT_TIMEOUT_SECONDS",
    "DEFAULT_ANALYSIS_TIMEOUT_SECONDS",
    "DEFAULT_TOP_K",
    "UISettings",
    "load_ui_settings",
]

DEFAULT_API_BASE_URL = "http://127.0.0.1:8000"

# Matching embeds a job description against an already-built index: fast.
DEFAULT_TIMEOUT_SECONDS = 30.0

# Analysis makes one model call. Against a real provider that is seconds, and
# the very first request also loads the embedding model, so this is generous.
DEFAULT_ANALYSIS_TIMEOUT_SECONDS = 180.0

DEFAULT_TOP_K = 10
MAX_TOP_K = 100


@dataclass(frozen=True, slots=True)
class UISettings:
    """Everything the dashboard needs to reach its backend.

    Attributes:
        api_base_url: Backend root, without a trailing slash.
        timeout_seconds: Timeout for health, listing, upload and matching.
        analysis_timeout_seconds: Timeout for candidate analysis.
        default_top_k: Default number of candidates to rank.
    """

    api_base_url: str = DEFAULT_API_BASE_URL
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    analysis_timeout_seconds: float = DEFAULT_ANALYSIS_TIMEOUT_SECONDS
    default_top_k: int = DEFAULT_TOP_K


def _float_from_env(name: str, default: float) -> float:
    """Read a positive float from the environment, falling back on nonsense."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _int_from_env(name: str, default: int, maximum: int) -> int:
    """Read a positive, capped integer from the environment."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    if value < 1:
        return default
    return min(value, maximum)


def load_ui_settings() -> UISettings:
    """Build :class:`UISettings` from the current environment.

    Returns:
        The settings. Unparsable values fall back to the documented defaults
        rather than raising, so a typo cannot stop the dashboard from starting
        with no way to see why.
    """
    base_url = os.environ.get("API_BASE_URL", "").strip() or DEFAULT_API_BASE_URL

    return UISettings(
        api_base_url=base_url.rstrip("/"),
        timeout_seconds=_float_from_env("API_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS),
        analysis_timeout_seconds=_float_from_env(
            "API_ANALYSIS_TIMEOUT_SECONDS", DEFAULT_ANALYSIS_TIMEOUT_SECONDS
        ),
        default_top_k=_int_from_env("UI_DEFAULT_TOP_K", DEFAULT_TOP_K, MAX_TOP_K),
    )
