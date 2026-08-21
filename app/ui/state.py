"""Session state: what the dashboard remembers between reruns.

Streamlit re-executes the whole script on every interaction, so anything that
should survive a click lives in ``st.session_state``. Rather than scattering
string keys through the pages, every read and write goes through this module.

What is kept, and why:

``job_description``
    The recruiter typed it; losing it on a rerun would be unforgivable.
``ranking``
    One ``/match-candidates`` response. Kept so switching pages does not
    re-rank, and cleared when the job description changes -- a ranking against
    a different job would be quietly wrong.
``analyses``
    Analyses by candidate id. Each one costs a model call, so they are cached
    for the current job description and dropped when it changes.
``selected_candidate``
    Which candidate the detail page shows.

Nothing here is a cache of backend data that could go stale silently: the
candidate pool is re-read from the API on each render, because uploads change it.
"""

from __future__ import annotations

from typing import Any, MutableMapping

__all__ = [
    "PAGES",
    "DEFAULT_PAGE",
    "init_state",
    "get_job_description",
    "set_job_description",
    "has_job_description",
    "get_ranking",
    "set_ranking",
    "clear_results",
    "get_analysis",
    "store_analysis",
    "all_analyses",
    "analysed_count",
    "select_candidate",
    "get_selected_candidate",
    "goto",
    "apply_pending_navigation",
    "current_page",
    "completed_steps",
]

PAGES: tuple[str, ...] = ("Overview", "Screening", "Ranking", "Candidate")
DEFAULT_PAGE = "Overview"

# Where a requested navigation waits until it can be applied. See :func:`goto`.
PENDING_PAGE_KEY = "_pending_page"

_DEFAULTS: dict[str, Any] = {
    "page": DEFAULT_PAGE,
    "job_description": "",
    "ranking": None,
    "analyses": {},
    "selected_candidate": None,
    "upload_results": [],
    PENDING_PAGE_KEY: None,
}


def init_state(state: MutableMapping[str, Any]) -> None:
    """Ensure every key this dashboard reads exists.

    Args:
        state: The session state mapping.
    """
    for key, default in _DEFAULTS.items():
        if key not in state:
            state[key] = default.copy() if isinstance(default, (dict, list)) else default


# --- Job description ------------------------------------------------------


def get_job_description(state: MutableMapping[str, Any]) -> str:
    """Return the job description currently being screened against."""
    return str(state.get("job_description") or "")


def set_job_description(state: MutableMapping[str, Any], text: str) -> bool:
    """Store a job description, discarding results if it actually changed.

    A ranking and its analyses are only meaningful against the job description
    they were produced for, so changing the text invalidates both rather than
    leaving stale numbers on screen under a new heading.

    Args:
        state: The session state mapping.
        text: The new job description.

    Returns:
        ``True`` if the text changed and results were cleared.
    """
    cleaned = text.strip()
    if cleaned == get_job_description(state):
        return False

    state["job_description"] = cleaned
    clear_results(state)
    return True


def has_job_description(state: MutableMapping[str, Any]) -> bool:
    """Whether a usable job description has been entered."""
    return bool(get_job_description(state))


# --- Ranking and analyses -------------------------------------------------


def get_ranking(state: MutableMapping[str, Any]) -> dict[str, Any] | None:
    """Return the stored ranking response, if there is one."""
    ranking = state.get("ranking")
    return ranking if isinstance(ranking, dict) else None


def set_ranking(state: MutableMapping[str, Any], ranking: dict[str, Any]) -> None:
    """Store a ranking response."""
    state["ranking"] = ranking


def clear_results(state: MutableMapping[str, Any]) -> None:
    """Drop the ranking, every analysis and the current selection."""
    state["ranking"] = None
    state["analyses"] = {}
    state["selected_candidate"] = None


def get_analysis(state: MutableMapping[str, Any], candidate_id: str) -> dict[str, Any] | None:
    """Return the stored analysis for one candidate, if there is one."""
    analyses = state.get("analyses") or {}
    analysis = analyses.get(candidate_id)
    return analysis if isinstance(analysis, dict) else None


def store_analysis(
    state: MutableMapping[str, Any], candidate_id: str, analysis: dict[str, Any]
) -> None:
    """Cache one analysis for the current job description."""
    analyses = state.setdefault("analyses", {})
    analyses[candidate_id] = analysis


def all_analyses(state: MutableMapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Return every cached analysis, keyed by candidate id."""
    analyses = state.get("analyses")
    return analyses if isinstance(analyses, dict) else {}


def analysed_count(state: MutableMapping[str, Any]) -> int:
    """How many candidates have been analysed for the current job."""
    return len(all_analyses(state))


# --- Selection and navigation --------------------------------------------


def select_candidate(state: MutableMapping[str, Any], candidate_id: str | None) -> None:
    """Set which candidate the detail page shows."""
    state["selected_candidate"] = candidate_id


def get_selected_candidate(state: MutableMapping[str, Any]) -> str | None:
    """Return the selected candidate id, if any."""
    selected = state.get("selected_candidate")
    return selected if isinstance(selected, str) and selected else None


def goto(state: MutableMapping[str, Any], page: str) -> None:
    """Request a move to a page.

    The move is recorded as *pending* rather than written straight to ``page``.
    ``page`` is the sidebar radio's own widget key, and Streamlit refuses to let
    a script modify a widget's value after that widget has been created in the
    same run -- which is exactly when a button handler wants to navigate.
    :func:`apply_pending_navigation` performs the move on the next run, before
    the radio is built.

    Args:
        state: The session state mapping.
        page: One of :data:`PAGES`. An unknown name is ignored, so a typo
            cannot leave the dashboard on a blank screen.
    """
    if page in PAGES:
        state[PENDING_PAGE_KEY] = page


def apply_pending_navigation(state: MutableMapping[str, Any]) -> None:
    """Commit a pending move. Call before the navigation widget is created.

    Args:
        state: The session state mapping.
    """
    pending = state.get(PENDING_PAGE_KEY)
    if pending in PAGES:
        state["page"] = pending
    state[PENDING_PAGE_KEY] = None


def current_page(state: MutableMapping[str, Any]) -> str:
    """Return the current page, honouring any pending move."""
    pending = state.get(PENDING_PAGE_KEY)
    if pending in PAGES:
        return pending
    page = state.get("page")
    return page if page in PAGES else DEFAULT_PAGE


def completed_steps(state: MutableMapping[str, Any]) -> tuple[bool, bool, bool]:
    """Report progress through the three screening steps.

    The steps are the recruiter's workflow, not the system's internals:
    describe the role, rank the pool, analyse the shortlist. Loading resumes is
    a precondition of ranking rather than a step of its own -- a pool can be
    populated long before anyone screens against it.

    Args:
        state: The session state mapping.

    Returns:
        ``(role_described, candidates_ranked, candidates_analyzed)``.
    """
    return (
        has_job_description(state),
        get_ranking(state) is not None,
        analysed_count(state) > 0,
    )
