"""Streamlit entry point.

Run it with::

    streamlit run app/ui/dashboard.py

This module is the shell: page configuration, theme, sidebar, the one API call
that every page needs, and dispatch. The screens themselves live in
:mod:`app.ui.pages`.

The candidate pool is fetched here, once per render, rather than cached: an
upload changes it, and a stale count on a dashboard is worse than a fetch.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Streamlit puts the script's own directory on sys.path, so `app/ui` sits ahead
# of the project root and `import app.…` would resolve to whatever module lives
# beside this file. The project root is prepended so the real `app` package wins.
# (This file is named dashboard.py rather than app.py for the same reason: a
# script named app.py registers a top-level module `app` that shadows the
# package outright, which no sys.path ordering can undo.)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_HERE = str(Path(__file__).resolve().parent)
if _HERE in sys.path:
    sys.path.remove(_HERE)
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.ui import state as ui_state  # noqa: E402
from app.ui.api_client import APIClientError, APIUnavailableError, ScreeningAPIClient  # noqa: E402
from app.ui.components import (  # noqa: E402
    api_unavailable_state,
    brand,
    error_state,
    inject_theme,
    sidebar_eyebrow,
)
from app.ui.config import load_ui_settings  # noqa: E402
from app.ui.pages import render_page  # noqa: E402

APP_TITLE = "Resume Screening AI"
APP_TAGLINE = "Recruiter dashboard"
BRAND_MARK = "RS"


@st.cache_resource(show_spinner=False)
def get_client() -> ScreeningAPIClient:
    """Return the process-wide API client.

    Cached so one connection pool is reused across reruns instead of a new one
    being opened on every interaction.
    """
    return ScreeningAPIClient(load_ui_settings())


def render_sidebar(client: ScreeningAPIClient, state) -> str:
    """Draw the sidebar and return the selected page.

    Args:
        client: The API client, used for the status indicator.
        state: Session state.

    Returns:
        The page the recruiter chose. It is also written straight to
        ``state["page"]`` by the widget, so callers rarely need the return value.
    """
    with st.sidebar:
        brand(APP_TITLE, APP_TAGLINE, BRAND_MARK)

        sidebar_eyebrow("Workspace")
        # key="page" makes the radio *be* the stored page rather than a copy of
        # it. Without that, a programmatic move -- "Rank candidates" jumping to
        # the ranking -- is silently undone on the next run by the widget's own
        # sticky value.
        page = st.radio(
            "Navigation",
            options=ui_state.PAGES,
            key="page",
            label_visibility="collapsed",
        )

        sidebar_eyebrow("Session")
        if ui_state.has_job_description(state):
            st.caption("Role described")
        else:
            st.caption("No role described yet")

        ranking = ui_state.get_ranking(state)
        if ranking:
            st.caption(f"{len(ranking.get('results') or [])} ranked")
        analysed = ui_state.analysed_count(state)
        if analysed:
            st.caption(f"{analysed} analyzed")

        st.markdown("")
        if st.button(
            "Start over",
            help="Clear the job description, ranking and analyses.",
        ):
            ui_state.set_job_description(state, "")
            ui_state.clear_results(state)
            state["upload_results"] = []
            ui_state.goto(state, ui_state.DEFAULT_PAGE)
            st.rerun()

        sidebar_eyebrow("Backend")
        if client.is_available():
            st.success("API connected", icon=":material/check_circle:")
        else:
            st.error("API unreachable", icon=":material/cloud_off:")
        st.caption(client.base_url)

        sidebar_eyebrow("How to read this")
        st.caption(
            "Similarity is a ranking signal, not a hiring probability. Skill coverage "
            "is a separate count. Recommendations are coarse labels, never decisions. "
            "Read the evidence."
        )

    return page


def main() -> None:
    """Configure the page, resolve state, and render."""
    st.set_page_config(
        page_title=f"{APP_TITLE} — {APP_TAGLINE}",
        page_icon=":material/person_search:",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    inject_theme()

    state = st.session_state
    ui_state.init_state(state)
    # Must happen before the navigation widget exists; see ui_state.goto.
    ui_state.apply_pending_navigation(state)

    client = get_client()

    render_sidebar(client, state)

    try:
        pool = client.list_candidates()
    except APIUnavailableError:
        api_unavailable_state(client.base_url)
        return
    except APIClientError as error:
        # A missing resume directory is a server configuration problem, not a
        # recruiter's mistake, so it is reported rather than silently empty.
        error_state(error.message)
        pool = {"candidates": [], "count": 0, "unreadable": []}

    render_page(ui_state.current_page(state), client, state, pool)


main()
