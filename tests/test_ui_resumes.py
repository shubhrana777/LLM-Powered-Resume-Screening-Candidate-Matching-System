"""Tests for the resume pool management page and the session reset.

Two distinctions carry the weight here, and both are easy to get wrong:

* **A new screening session is not a delete.** It clears the job description,
  the ranking and the analyses -- the work done for one role -- and leaves every
  stored resume where it is. Emptying the pool is a separate, confirmed action.
* **Deleting invalidates what was derived from it.** A ranking that included a
  deleted candidate no longer describes anything, and their analysis is about a
  resume that is gone. Both are dropped rather than left on screen.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from app.ui import state as ui_state
from app.ui.api_client import APIError
from tests.test_ui_app import ANALYSIS, JOB, POOL, RANKING, StubClient, run_app, text_of

APP_FILE = str(Path(__file__).resolve().parents[1] / "app" / "ui" / "dashboard.py")


class DeletingStubClient(StubClient):
    """A stub whose pool actually shrinks when candidates are deleted."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._pool = {
            "candidates": [dict(entry) for entry in POOL["candidates"]],
            "count": len(POOL["candidates"]),
            "unreadable": [],
        }
        self.deleted: list[str] = []
        self.fail_on: set[str] = set()

    def _remove(self, candidate_ids):
        removed, failed = [], []
        for candidate_id in candidate_ids:
            if candidate_id in self.fail_on:
                failed.append(f"{candidate_id}: that resume could not be deleted.")
                continue
            before = len(self._pool["candidates"])
            self._pool["candidates"] = [
                entry for entry in self._pool["candidates"]
                if entry["candidate_id"] != candidate_id
            ]
            if len(self._pool["candidates"]) < before:
                removed.append(candidate_id)
                self.deleted.append(candidate_id)
            else:
                failed.append(f"{candidate_id}: no candidate matching that name.")
        self._pool["count"] = len(self._pool["candidates"])
        return {"deleted": removed, "failed": failed, "remaining": self._pool["count"]}

    def delete_candidate(self, candidate: str) -> dict[str, Any]:
        self.calls.append(f"delete_candidate:{candidate}")
        result = self._remove([candidate])
        if not result["deleted"]:
            raise APIError(
                f"No candidate matching {candidate!r}.", status_code=404, code="candidate_not_found"
            )
        return result

    def delete_candidates(self, candidates) -> dict[str, Any]:
        self.calls.append(f"delete_candidates:{','.join(candidates)}")
        return self._remove(list(candidates))

    def clear_candidates(self) -> dict[str, Any]:
        self.calls.append("clear_candidates")
        ids = [entry["candidate_id"] for entry in self._pool["candidates"]]
        return self._remove(ids)


@pytest.fixture(autouse=True)
def _clear_streamlit_caches():
    st.cache_resource.clear()
    yield
    st.cache_resource.clear()


def open_resumes(monkeypatch, client=None, **session) -> AppTest:
    """Run the app on the Resumes page."""
    payload = {"page": "Resumes"}
    payload.update(session)
    return run_app(monkeypatch, client or DeletingStubClient(), payload)


def click(app: AppTest, label: str) -> AppTest:
    """Click the first button with this label and rerun."""
    return next(button for button in app.button if button.label == label).click().run()


# --------------------------------------------------------------------------
# The page exists and lists the pool
# --------------------------------------------------------------------------


def test_resumes_is_a_page(monkeypatch):
    app = run_app(monkeypatch, DeletingStubClient())
    assert "Resumes" in app.sidebar.radio[0].options


def test_the_page_renders(monkeypatch):
    assert not open_resumes(monkeypatch).exception


def test_every_stored_candidate_is_listed(monkeypatch):
    rendered = text_of(open_resumes(monkeypatch))
    for name in ("Sarah Wilson", "James Patel", "Nina Volkov"):
        assert name in rendered


def test_filenames_are_shown(monkeypatch):
    assert "sarah_wilson.pdf" in text_of(open_resumes(monkeypatch))


def test_the_stored_count_is_shown(monkeypatch):
    rendered = text_of(open_resumes(monkeypatch))
    assert "Stored resumes" in rendered


def test_an_empty_pool_says_so(monkeypatch):
    empty = DeletingStubClient()
    empty._pool = {"candidates": [], "count": 0, "unreadable": []}

    assert "The pool is empty" in text_of(open_resumes(monkeypatch, empty))


def test_unreadable_files_are_reported(monkeypatch):
    client = DeletingStubClient()
    client._pool["unreadable"] = [{"filename": "broken.pdf", "reason": "not a PDF"}]

    rendered = text_of(open_resumes(monkeypatch, client))
    assert "broken.pdf" in rendered


# --------------------------------------------------------------------------
# Delete one
# --------------------------------------------------------------------------


def test_each_candidate_has_a_remove_button(monkeypatch):
    app = open_resumes(monkeypatch)
    assert len([b for b in app.button if b.label == "Remove"]) == 3


def test_removing_one_candidate_calls_the_api(monkeypatch):
    client = DeletingStubClient()
    app = open_resumes(monkeypatch, client)

    next(b for b in app.button if b.key == "delete_james_patel").click().run()

    assert "delete_candidate:james_patel" in client.calls
    assert client.deleted == ["james_patel"]


def test_a_removed_candidate_disappears_from_the_list(monkeypatch):
    client = DeletingStubClient()
    app = open_resumes(monkeypatch, client)

    app = next(b for b in app.button if b.key == "delete_james_patel").click().run()

    rendered = text_of(app)
    assert "James Patel" not in rendered
    assert "Sarah Wilson" in rendered


def test_a_successful_removal_is_confirmed(monkeypatch):
    client = DeletingStubClient()
    app = open_resumes(monkeypatch, client)

    app = next(b for b in app.button if b.key == "delete_james_patel").click().run()

    assert any("Removed 1 resume" in str(item.value) for item in app.success)


def test_a_failed_removal_is_reported_without_crashing(monkeypatch):
    client = DeletingStubClient()
    client.fail_on = {"james_patel"}
    app = open_resumes(monkeypatch, client)

    app = next(b for b in app.button if b.key == "delete_james_patel").click().run()

    assert not app.exception
    assert any("No candidate matching" in str(item.value) for item in app.error)


# --------------------------------------------------------------------------
# Delete several
# --------------------------------------------------------------------------


def test_candidates_can_be_selected(monkeypatch):
    app = open_resumes(monkeypatch)
    assert len(app.checkbox) == 3


def test_selecting_reveals_a_bulk_remove_button(monkeypatch):
    app = open_resumes(monkeypatch)
    app = app.checkbox[0].set_value(True).run()

    assert any("selected resume" in str(b.label) for b in app.button)


def test_removing_several_calls_the_batch_endpoint(monkeypatch):
    client = DeletingStubClient()
    app = open_resumes(monkeypatch, client)

    app = app.checkbox[0].set_value(True).run()
    app = app.checkbox[1].set_value(True).run()
    app = click(app, "Remove 2 selected resumes")

    assert any(call.startswith("delete_candidates:") for call in client.calls)
    assert len(client.deleted) == 2


def test_removing_several_leaves_the_rest(monkeypatch):
    client = DeletingStubClient()
    app = open_resumes(monkeypatch, client)

    app = app.checkbox[0].set_value(True).run()
    app = app.checkbox[1].set_value(True).run()
    app = click(app, "Remove 2 selected resumes")

    assert client._pool["count"] == 1


def test_a_partial_batch_failure_is_reported(monkeypatch):
    client = DeletingStubClient()
    client.fail_on = {"sarah_wilson"}
    app = open_resumes(monkeypatch, client)

    app = app.checkbox[0].set_value(True).run()
    app = app.checkbox[1].set_value(True).run()
    app = click(app, "Remove 2 selected resumes")

    assert not app.exception
    assert any("could not be removed" in str(item.value) for item in app.error)


# --------------------------------------------------------------------------
# Clear all, behind a confirmation
# --------------------------------------------------------------------------


def test_clearing_asks_for_confirmation_first(monkeypatch):
    client = DeletingStubClient()
    app = open_resumes(monkeypatch, client)

    app = click(app, "Clear resume pool")

    assert "clear_candidates" not in client.calls
    assert any("cannot be undone" in str(item.value) for item in app.warning)


def test_confirming_clears_the_pool(monkeypatch):
    client = DeletingStubClient()
    app = open_resumes(monkeypatch, client)

    app = click(app, "Clear resume pool")
    app = click(app, "Yes, delete everything")

    assert "clear_candidates" in client.calls
    assert client._pool["count"] == 0


def test_cancelling_deletes_nothing(monkeypatch):
    client = DeletingStubClient()
    app = open_resumes(monkeypatch, client)

    app = click(app, "Clear resume pool")
    app = click(app, "Cancel")

    assert "clear_candidates" not in client.calls
    assert client._pool["count"] == 3


def test_the_confirmation_resets_after_cancelling(monkeypatch):
    app = open_resumes(monkeypatch)
    app = click(app, "Clear resume pool")
    app = click(app, "Cancel")

    assert app.session_state["confirm_clear_pool"] is False
    assert any(b.label == "Clear resume pool" for b in app.button)


def test_clearing_reports_what_went(monkeypatch):
    app = open_resumes(monkeypatch)
    app = click(app, "Clear resume pool")
    app = click(app, "Yes, delete everything")

    assert any("Cleared 3 resumes" in str(item.value) for item in app.success)


# --------------------------------------------------------------------------
# New screening session
# --------------------------------------------------------------------------


def test_a_new_session_clears_the_job_description(monkeypatch):
    app = open_resumes(monkeypatch, job_description=JOB, ranking=RANKING)
    app = click(app, "New screening session")

    assert app.session_state["job_description"] == ""


def test_a_new_session_clears_the_ranking_and_analyses(monkeypatch):
    app = open_resumes(
        monkeypatch,
        job_description=JOB,
        ranking=RANKING,
        analyses={"sarah_wilson": ANALYSIS},
        selected_candidate="sarah_wilson",
    )
    app = click(app, "New screening session")

    assert app.session_state["ranking"] is None
    assert app.session_state["analyses"] == {}
    assert app.session_state["selected_candidate"] is None


def test_a_new_session_clears_the_upload_report(monkeypatch):
    app = open_resumes(
        monkeypatch,
        job_description=JOB,
        upload_results=[{"filename": "x.pdf", "ok": True, "message": "done"}],
    )
    app = click(app, "New screening session")

    assert app.session_state["upload_results"] == []


def test_a_new_session_keeps_every_resume(monkeypatch):
    """The whole distinction: a session reset is not a delete."""
    client = DeletingStubClient()
    app = open_resumes(monkeypatch, client, job_description=JOB, ranking=RANKING)

    app = click(app, "New screening session")

    assert client.deleted == []
    assert client._pool["count"] == 3
    assert not any(call.startswith(("delete", "clear")) for call in client.calls)


def test_a_new_session_moves_to_screening(monkeypatch):
    app = open_resumes(monkeypatch, job_description=JOB)
    app = click(app, "New screening session")

    assert app.session_state["page"] == "Screening"


# --------------------------------------------------------------------------
# Deleting invalidates what was derived from it
# --------------------------------------------------------------------------


def test_deleting_drops_a_ranking_that_included_the_candidate(monkeypatch):
    client = DeletingStubClient()
    app = open_resumes(
        monkeypatch, client, job_description=JOB, ranking=RANKING,
        analyses={"sarah_wilson": ANALYSIS},
    )

    app = next(b for b in app.button if b.key == "delete_james_patel").click().run()

    assert app.session_state["ranking"] is None


def test_deleting_drops_that_candidates_analysis(monkeypatch):
    client = DeletingStubClient()
    app = open_resumes(
        monkeypatch, client, job_description=JOB, ranking=RANKING,
        analyses={"sarah_wilson": ANALYSIS, "james_patel": dict(ANALYSIS, candidate_id="james_patel")},
    )

    app = next(b for b in app.button if b.key == "delete_james_patel").click().run()

    assert "james_patel" not in app.session_state["analyses"]


def test_deleting_keeps_the_analyses_of_surviving_candidates(monkeypatch):
    client = DeletingStubClient()
    app = open_resumes(
        monkeypatch, client, job_description=JOB, ranking=RANKING,
        analyses={"sarah_wilson": ANALYSIS, "james_patel": dict(ANALYSIS, candidate_id="james_patel")},
    )

    app = next(b for b in app.button if b.key == "delete_james_patel").click().run()

    assert "sarah_wilson" in app.session_state["analyses"]


def test_deleting_the_selected_candidate_clears_the_selection(monkeypatch):
    client = DeletingStubClient()
    app = open_resumes(
        monkeypatch, client, job_description=JOB, ranking=RANKING,
        selected_candidate="james_patel",
    )

    app = next(b for b in app.button if b.key == "delete_james_patel").click().run()

    assert app.session_state["selected_candidate"] is None


def test_the_job_description_survives_a_delete(monkeypatch):
    """Deleting a resume is not abandoning the role being screened for."""
    client = DeletingStubClient()
    app = open_resumes(monkeypatch, client, job_description=JOB, ranking=RANKING)

    app = next(b for b in app.button if b.key == "delete_james_patel").click().run()

    assert app.session_state["job_description"] == JOB


# --------------------------------------------------------------------------
# The state helpers, directly
# --------------------------------------------------------------------------


@pytest.fixture
def state() -> dict:
    fresh: dict = {}
    ui_state.init_state(fresh)
    return fresh


def test_new_session_clears_session_work_only(state):
    ui_state.set_job_description(state, "Analyst")
    ui_state.set_ranking(state, RANKING)
    ui_state.store_analysis(state, "sarah_wilson", ANALYSIS)
    state["upload_results"] = [{"filename": "x.pdf"}]

    ui_state.new_session(state)

    assert ui_state.get_job_description(state) == ""
    assert ui_state.get_ranking(state) is None
    assert ui_state.all_analyses(state) == {}
    assert state["upload_results"] == []


def test_forget_candidates_removes_only_the_named_ones(state):
    ui_state.store_analysis(state, "a", ANALYSIS)
    ui_state.store_analysis(state, "b", ANALYSIS)

    ui_state.forget_candidates(state, ["a"])

    assert "a" not in ui_state.all_analyses(state)
    assert "b" in ui_state.all_analyses(state)


def test_forget_candidates_drops_the_ranking(state):
    """Ranks and the considered count describe a pool that has changed."""
    ui_state.set_ranking(state, RANKING)

    ui_state.forget_candidates(state, ["sarah_wilson"])

    assert ui_state.get_ranking(state) is None


def test_forgetting_nothing_changes_nothing(state):
    ui_state.set_ranking(state, RANKING)

    ui_state.forget_candidates(state, [])

    assert ui_state.get_ranking(state) is not None


def test_forget_candidates_clears_a_matching_selection(state):
    ui_state.select_candidate(state, "sarah_wilson")

    ui_state.forget_candidates(state, ["sarah_wilson"])

    assert ui_state.get_selected_candidate(state) is None


def test_forget_candidates_keeps_an_unrelated_selection(state):
    ui_state.select_candidate(state, "sarah_wilson")

    ui_state.forget_candidates(state, ["james_patel"])

    assert ui_state.get_selected_candidate(state) == "sarah_wilson"
