"""Streamlit recruiter dashboard (Phase 6).

A **client**, not a second implementation. Every number this dashboard shows
came over HTTP from the FastAPI backend; nothing here parses a PDF, embeds text,
searches an index, extracts a skill or calls a language model. The only module
that speaks to the backend is :mod:`app.ui.api_client`, so the rest of the UI
never touches a URL, a status code or a JSON key it did not ask for.

Run it with::

    uvicorn app.api.main:app --reload      # terminal 1
    streamlit run app/ui/dashboard.py      # terminal 2

Nothing is imported at package level: importing :mod:`app.ui.dashboard` starts
a Streamlit page, and a bare ``import app.ui.formatting`` should not.
"""
