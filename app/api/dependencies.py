"""Shared FastAPI dependencies.

Both providers are memoized, which is the whole point: the service holds the
loaded resume pool, the FAISS indexes and the Sentence Transformers model, and
building one per request would reload the model every time. One instance is
created on first use and reused for the life of the process.

Tests replace either provider through ``app.dependency_overrides`` -- that is
how the suite runs against a temporary resume directory with an offline
embedder and never downloads model weights::

    app.dependency_overrides[get_service] = lambda: my_service
"""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from app.api.config import Settings, load_settings
from app.api.service import ScreeningService

__all__ = [
    "get_settings",
    "get_service",
    "reset_caches",
    "SettingsDep",
    "ServiceDep",
]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings, read from the environment once."""
    return load_settings()


@lru_cache(maxsize=1)
def get_service() -> ScreeningService:
    """Return the process-wide screening service.

    Constructing it is cheap -- the embedder and the provider are resolved
    lazily on first use, so importing the app does not load a model.
    """
    return ScreeningService(get_settings())


def reset_caches() -> None:
    """Drop both cached instances, so the next call re-reads the environment."""
    get_settings.cache_clear()
    get_service.cache_clear()


SettingsDep = Annotated[Settings, Depends(get_settings)]
ServiceDep = Annotated[ScreeningService, Depends(get_service)]
