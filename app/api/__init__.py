"""REST API layer (Phase 5).

A thin HTTP shell over the existing application: every endpoint delegates to the
Phase 1-4 modules and none of the parsing, embedding, ranking, extraction or
retrieval logic is reimplemented here.

Run it with::

    uvicorn app.api.main:app --reload

Nothing is imported at package level on purpose -- importing :mod:`app.api.main`
constructs the application, and a bare ``import app.api.config`` should not do
that.
"""
