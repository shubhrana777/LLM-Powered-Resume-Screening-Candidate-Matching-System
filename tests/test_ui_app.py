"""End-to-end tests for the Streamlit app itself.

These run the real script through Streamlit's ``AppTest`` harness -- the widget
tree is built, every ``render_*`` function executes, and the markup is produced
-- with the API client replaced by a stub. No browser, no server, no model.

They exist because the pure-logic tests cannot catch the failure that actually
matters in a Streamlit app: a page that raises while rendering. Each state a
recruiter can reach is rendered here and asserted to produce no exception.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from app.ui.api_client import APIUnavailableError

# AppTest resolves a relative path against the calling file, so it is made
# absolute here rather than depending on the working directory.
APP_FILE = str(Path(__file__).resolve().parents[1] / "app" / "ui" / "dashboard.py")

POOL = {
    "candidates": [
        {"candidate_id": "sarah_wilson", "name": "Sarah Wilson", "filename": "sarah_wilson.pdf", "text_length": 1987},
        {"candidate_id": "james_patel", "name": "James Patel", "filename": "james_patel.pdf", "text_length": 1170},
        {"candidate_id": "nina_volkov", "name": "Nina Volkov", "filename": "nina_volkov.pdf", "text_length": 1385},
    ],
    "count": 3,
    "unreadable": [],
}

RANKING = {
    "results": [
        {"rank": 1, "candidate": "Sarah Wilson", "candidate_id": "sarah_wilson", "similarity_score": 0.6024},
        {"rank": 2, "candidate": "James Patel", "candidate_id": "james_patel", "similarity_score": 0.4702},
        {"rank": 3, "candidate": "Nina Volkov", "candidate_id": "nina_volkov", "similarity_score": 0.1201},
    ],
    "count": 3,
    "candidates_considered": 3,
    "score_type": "cosine_similarity",
    "score_note": "Cosine similarity between the job-description embedding and the resume "
    "embedding, in [-1.0, 1.0]. A semantic similarity score: not a probability of being hired.",
}

ANALYSIS = {
    "candidate": "Sarah Wilson",
    "candidate_id": "sarah_wilson",
    "recommendation": "STRONG_MATCH",
    "recommendation_note": "A coarse ordinal label, not a score and not a probability.",
    "summary": "Sarah Wilson matches 12 of 13 skills identified in the job description.",
    "matched_skills": ["Python", "SQL", "Excel", "Power BI"],
    "skill_gaps": ["Investment Analysis"],
    "experience_assessment": "The resume states 4 years (stated on resume); the job asks for "
    "3 years. Requirement met: yes.",
    "education": ["MBA - Finance"],
    "evidence": [
        {
            "candidate_id": "sarah_wilson",
            "chunk_id": "sarah_wilson#4",
            "text": "Python, Power BI, Tableau, budgeting, risk analysis, data analysis",
            "retrieval_score": 0.6709,
        }
    ],
    "limitations": [],
    "warnings": [],
    "is_grounded": True,
    "model": "fake/deterministic-v1",
}

JOB = "Financial analyst with 3+ years of experience. Strong Excel and SQL."


class StubClient:
    """Stands in for :class:`~app.ui.api_client.ScreeningAPIClient`.

    Records what the UI asked for, so a test can assert the dashboard called
    the endpoint it claimed to.
    """

    def __init__(self, available: bool = True, pool: dict[str, Any] | None = None) -> None:
        from app.ui.config import UISettings

        self.settings = UISettings()
        self.base_url = self.settings.api_base_url
        self._available = available
        self._pool = pool if pool is not None else POOL
        self.calls: list[str] = []

    def is_available(self) -> bool:
        return self._available

    def health(self) -> dict[str, Any]:
        return {"status": "healthy", "service": "resume-screening-api", "llm_provider": "fake"}

    def list_candidates(self) -> dict[str, Any]:
        self.calls.append("list_candidates")
        if not self._available:
            raise APIUnavailableError(f"Could not reach the API at {self.base_url}.")
        return self._pool

    def match_candidates(self, job_description: str, top_k: int) -> dict[str, Any]:
        self.calls.append("match_candidates")
        return RANKING

    def analyze_candidate(self, candidate: str, job_description: str) -> dict[str, Any]:
        self.calls.append(f"analyze_candidate:{candidate}")
        return ANALYSIS

    def upload_resume(self, filename: str, content: bytes, store: bool = True) -> dict[str, Any]:
        self.calls.append(f"upload_resume:{filename}")
        return {"filename": filename, "status": "success", "text_length": 1200,
                "word_count": 190, "preview": "…", "stored": store,
                "candidate_id": filename.removesuffix(".pdf")}


@pytest.fixture(autouse=True)
def _clear_streamlit_caches():
    """Stop one test's cached client leaking into the next."""
    st.cache_resource.clear()
    yield
    st.cache_resource.clear()


def run_app(monkeypatch, client: StubClient, session: dict[str, Any] | None = None) -> AppTest:
    """Run the dashboard with ``client`` standing in for the real one."""
    monkeypatch.setattr("app.ui.api_client.ScreeningAPIClient", lambda *a, **k: client)

    app = AppTest.from_file(APP_FILE, default_timeout=60)
    for key, value in (session or {}).items():
        app.session_state[key] = value
    app.run()
    return app


def text_of(app: AppTest) -> str:
    """Flatten everything the app rendered into one searchable string."""
    parts: list[str] = []
    for element in app.markdown:
        parts.append(str(element.value))
    for element in app.caption:
        parts.append(str(element.value))
    for collection in (app.error, app.warning, app.info, app.success):
        for element in collection:
            parts.append(str(element.value))
    for element in app.subheader:
        parts.append(str(element.value))
    return "\n".join(parts)


# --------------------------------------------------------------------------
# The app renders at all
# --------------------------------------------------------------------------


def test_the_app_starts_without_error(monkeypatch):
    app = run_app(monkeypatch, StubClient())
    assert not app.exception


def test_the_sidebar_reports_a_connected_api(monkeypatch):
    app = run_app(monkeypatch, StubClient())
    assert any("API connected" in str(item.value) for item in app.sidebar.success)


def test_navigation_offers_every_page(monkeypatch):
    from app.ui.state import PAGES

    app = run_app(monkeypatch, StubClient())
    assert tuple(app.sidebar.radio[0].options) == PAGES


@pytest.mark.parametrize("page", ["Overview", "Screening", "Ranking", "Candidate"])
def test_every_page_renders_without_error(monkeypatch, page: str):
    app = run_app(monkeypatch, StubClient(), {"page": page})
    assert not app.exception, f"{page} raised: {app.exception}"


@pytest.mark.parametrize("page", ["Overview", "Screening", "Ranking", "Candidate"])
def test_every_page_renders_with_a_full_session(monkeypatch, page: str):
    """The same pages, but with a job, a ranking and an analysis already present."""
    app = run_app(
        monkeypatch,
        StubClient(),
        {
            "page": page,
            "job_description": JOB,
            "ranking": RANKING,
            "analyses": {"sarah_wilson": ANALYSIS},
            "selected_candidate": "sarah_wilson",
        },
    )
    assert not app.exception, f"{page} raised: {app.exception}"


# --------------------------------------------------------------------------
# Overview shows real values, not invented ones
# --------------------------------------------------------------------------


def test_the_overview_counts_the_real_pool(monkeypatch):
    app = run_app(monkeypatch, StubClient())
    rendered = text_of(app)
    assert "Candidates in pool" in rendered
    assert ">3<" in rendered


def test_the_overview_does_not_invent_an_average(monkeypatch):
    """With no ranking there is no average, and none is shown."""
    app = run_app(monkeypatch, StubClient())
    rendered = text_of(app)
    assert "Average similarity" in rendered
    assert "Not analyzed yet" in rendered


def test_the_overview_reports_the_real_average_once_ranked(monkeypatch):
    app = run_app(monkeypatch, StubClient(), {"job_description": JOB, "ranking": RANKING})
    expected = (0.6024 + 0.4702 + 0.1201) / 3
    assert f"{expected * 100:.2f}%" in text_of(app)


def test_an_empty_pool_shows_an_empty_state_not_zeroes(monkeypatch):
    empty = {"candidates": [], "count": 0, "unreadable": []}
    app = run_app(monkeypatch, StubClient(pool=empty))

    assert "No resumes loaded" in text_of(app)
    assert not app.exception


def test_unreadable_files_are_surfaced(monkeypatch):
    pool = dict(POOL, unreadable=[{"filename": "broken.pdf", "reason": "not a PDF"}])
    app = run_app(monkeypatch, StubClient(pool=pool))

    assert "broken.pdf" in text_of(app)


# --------------------------------------------------------------------------
# The API-unavailable state
# --------------------------------------------------------------------------


def test_an_unreachable_api_shows_how_to_start_it(monkeypatch):
    app = run_app(monkeypatch, StubClient(available=False))

    assert not app.exception
    rendered = text_of(app)
    assert "Cannot reach the screening API" in rendered
    # The start script is the primary instruction; the raw command is the fallback.
    assert any("start_app.ps1" in str(block.value) for block in app.code)
    assert "uvicorn app.api.main:app" in rendered


def test_an_unreachable_api_does_not_render_a_page(monkeypatch):
    """Better an honest failure than a dashboard of zeroes."""
    app = run_app(monkeypatch, StubClient(available=False))
    assert "Screening overview" not in text_of(app)


# --------------------------------------------------------------------------
# Screening workflow
# --------------------------------------------------------------------------


def test_the_screening_page_offers_a_job_description_field(monkeypatch):
    app = run_app(monkeypatch, StubClient(), {"page": "Screening"})
    assert len(app.text_area) == 1


def test_the_screening_page_offers_a_file_uploader(monkeypatch):
    app = run_app(monkeypatch, StubClient(), {"page": "Screening"})
    assert not app.exception
    assert "Candidate pool" in text_of(app)


def test_the_upload_area_explains_what_happens_to_a_resume(monkeypatch):
    rendered = text_of(run_app(monkeypatch, StubClient(), {"page": "Screening"}))
    assert "added to the candidate pool" in rendered
    assert "PDF" in rendered


def test_the_screening_page_shows_the_three_step_flow(monkeypatch):
    """Describe role, rank, analyze -- the recruiter workflow, in order."""
    from app.ui.pages import STEP_LABELS

    rendered = text_of(run_app(monkeypatch, StubClient(), {"page": "Screening"}))
    for label in STEP_LABELS:
        assert label in rendered


def test_the_current_step_is_marked(monkeypatch):
    rendered = text_of(run_app(monkeypatch, StubClient(), {"page": "Screening"}))
    assert "Current step" in rendered


def test_a_finished_step_is_marked_done(monkeypatch):
    app = run_app(monkeypatch, StubClient(), {"page": "Screening", "job_description": JOB})
    assert "Done" in text_of(app)


def test_an_empty_job_description_is_rejected_in_the_ui(monkeypatch):
    app = run_app(monkeypatch, StubClient(), {"page": "Screening"})
    app.text_area[0].set_value("   ")
    app.button[0].click().run()

    assert any("Enter a job description" in str(item.value) for item in app.error)


def test_saving_a_job_description_records_it(monkeypatch):
    app = run_app(monkeypatch, StubClient(), {"page": "Screening"})
    app.text_area[0].set_value(JOB)
    app.button[0].click().run()

    assert app.session_state["job_description"] == JOB


def test_ranking_cannot_start_without_a_job_description(monkeypatch):
    app = run_app(monkeypatch, StubClient(), {"page": "Screening"})
    assert "Nothing to rank against yet" in text_of(app)


def test_ranking_cannot_start_without_resumes(monkeypatch):
    empty = {"candidates": [], "count": 0, "unreadable": []}
    app = run_app(
        monkeypatch, StubClient(pool=empty), {"page": "Screening", "job_description": JOB}
    )
    assert "No candidates in the pool" in text_of(app)


def test_an_absent_job_description_gets_its_own_empty_state(monkeypatch):
    rendered = text_of(run_app(monkeypatch, StubClient(), {"page": "Screening"}))
    assert "No role described yet" in rendered


# --------------------------------------------------------------------------
# Ranking
# --------------------------------------------------------------------------


def test_the_ranking_page_is_empty_before_ranking(monkeypatch):
    app = run_app(monkeypatch, StubClient(), {"page": "Ranking"})
    assert "Nothing ranked yet" in text_of(app)


def _ranked(monkeypatch, **extra) -> AppTest:
    session = {"page": "Ranking", "job_description": JOB, "ranking": RANKING}
    session.update(extra)
    return run_app(monkeypatch, StubClient(), session)


def test_the_ranking_page_explains_what_similarity_is(monkeypatch):
    """The percent sign never travels without the wording that qualifies it."""
    rendered = text_of(_ranked(monkeypatch))
    assert "ranking signal" in rendered
    assert "not a hiring probability" in rendered


def test_the_ranking_page_shows_a_summary_row(monkeypatch):
    rendered = text_of(_ranked(monkeypatch))
    for label in ("Candidates ranked", "Analyzed", "Strong matches", "Average similarity"):
        assert label in rendered


def test_the_ranking_page_identifies_the_top_candidate(monkeypatch):
    rendered = text_of(_ranked(monkeypatch))
    assert "Top match" in rendered
    assert "Sarah Wilson" in rendered


def test_similarity_is_shown_as_a_percentage(monkeypatch):
    assert "60.24%" in text_of(_ranked(monkeypatch))


def test_the_ranking_table_holds_percentages_not_raw_scores(monkeypatch):
    frame = _ranked(monkeypatch).dataframe[0].value
    assert frame["Similarity"].iloc[0] == pytest.approx(60.24)


def test_unanalysed_rows_say_not_analyzed_yet(monkeypatch):
    frame = _ranked(monkeypatch).dataframe[0].value
    assert frame["Recommendation"].iloc[0] == "Not analyzed yet"
    assert frame["Skill coverage"].iloc[0] == "Not analyzed yet"


def test_analysed_rows_update_naturally(monkeypatch):
    """After analysis the same row carries coverage, experience and a label."""
    frame = _ranked(monkeypatch, analyses={"sarah_wilson": ANALYSIS}).dataframe[0].value
    row = frame[frame["Candidate"] == "Sarah Wilson"].iloc[0]

    assert row["Skill coverage"] == "4 / 5"
    assert row["Experience"] == "Requirement met"
    assert row["Recommendation"] == "Strong match"


def test_skill_coverage_is_never_a_percentage(monkeypatch):
    """Coverage and similarity are different measures and must look different."""
    frame = _ranked(monkeypatch, analyses={"sarah_wilson": ANALYSIS}).dataframe[0].value
    assert "%" not in str(frame["Skill coverage"].iloc[0])


def test_the_ranking_page_renders_a_table(monkeypatch):
    app = run_app(
        monkeypatch, StubClient(), {"page": "Ranking", "job_description": JOB, "ranking": RANKING}
    )
    assert len(app.dataframe) >= 1


def test_the_ranking_page_offers_filters(monkeypatch):
    app = run_app(
        monkeypatch, StubClient(), {"page": "Ranking", "job_description": JOB, "ranking": RANKING}
    )
    assert len(app.multiselect) >= 1
    assert len(app.selectbox) >= 1
    assert len(app.slider) >= 1


def test_an_empty_ranking_shows_a_no_results_state(monkeypatch):
    empty_ranking = dict(RANKING, results=[], count=0)
    app = run_app(
        monkeypatch,
        StubClient(),
        {"page": "Ranking", "job_description": JOB, "ranking": empty_ranking},
    )
    assert "No candidates matched" in text_of(app)


# --------------------------------------------------------------------------
# Candidate detail
# --------------------------------------------------------------------------


def test_the_candidate_page_asks_for_a_selection_first(monkeypatch):
    app = run_app(monkeypatch, StubClient(), {"page": "Candidate"})
    assert "No candidate selected" in text_of(app)


def _detail(monkeypatch) -> AppTest:
    return run_app(
        monkeypatch,
        StubClient(),
        {
            "page": "Candidate",
            "job_description": JOB,
            "ranking": RANKING,
            "analyses": {"sarah_wilson": ANALYSIS},
            "selected_candidate": "sarah_wilson",
        },
    )


def test_the_detail_view_shows_every_required_section(monkeypatch):
    rendered = text_of(_detail(monkeypatch))
    for section in (
        "Skills",
        "Experience",
        "Education",
        "AI-generated interpretation",
        "Evidence from resume",
        "Gaps",
    ):
        assert section in rendered, f"missing section: {section}"


def test_the_detail_view_shows_similarity_as_a_percentage(monkeypatch):
    assert "60.24%" in text_of(_detail(monkeypatch))


def test_the_detail_view_still_shows_the_raw_cosine_value(monkeypatch):
    """The percentage is a presentation, not a replacement."""
    assert "0.6024" in text_of(_detail(monkeypatch))


def test_the_detail_view_shows_four_distinct_measures(monkeypatch):
    """Similarity, coverage, experience and recommendation are separate cards."""
    rendered = text_of(_detail(monkeypatch))
    for label in ("Similarity", "Skill coverage", "Experience", "Recommendation"):
        assert label in rendered
    assert "60.24%" in rendered
    assert "4 / 5" in rendered
    assert "Requirement met" in rendered
    assert "Strong match" in rendered


def test_the_detail_view_shows_matched_skills_and_gaps(monkeypatch):
    rendered = text_of(_detail(monkeypatch))
    assert "Power BI" in rendered
    assert "Investment Analysis" in rendered


def test_the_detail_view_shows_education(monkeypatch):
    assert "MBA - Finance" in text_of(_detail(monkeypatch))


def test_the_detail_view_shows_the_evidence_text(monkeypatch):
    assert "budgeting, risk analysis" in text_of(_detail(monkeypatch))


def test_evidence_is_marked_as_coming_from_the_resume(monkeypatch):
    rendered = text_of(_detail(monkeypatch))
    assert "From resume" in rendered
    assert "sarah_wilson#4" in rendered


def test_generated_prose_is_marked_as_generated(monkeypatch):
    """A recruiter must be able to tell the model's words from the resume's."""
    rendered = text_of(_detail(monkeypatch))
    assert "AI-generated interpretation" in rendered
    assert "fake/deterministic-v1" in rendered


def test_evidence_is_labelled_as_coming_from_the_resume_not_the_model(monkeypatch):
    rendered = text_of(_detail(monkeypatch))
    assert "Evidence from resume" in rendered
    assert "Source text only" in rendered


def test_evidence_and_generated_text_use_different_containers(monkeypatch):
    rendered = text_of(_detail(monkeypatch))
    assert "rs-evidence" in rendered
    assert "rs-generated" in rendered


def test_the_detail_view_reports_grounding(monkeypatch):
    assert "Grounded" in text_of(_detail(monkeypatch))


def test_a_corrected_analysis_is_flagged(monkeypatch):
    corrected = dict(ANALYSIS, is_grounded=False, warnings=["removed unsupported skill 'AWS'"])
    app = run_app(
        monkeypatch,
        StubClient(),
        {
            "page": "Candidate",
            "job_description": JOB,
            "ranking": RANKING,
            "analyses": {"sarah_wilson": corrected},
            "selected_candidate": "sarah_wilson",
        },
    )
    assert "Corrected claims" in text_of(app)


def test_an_unanalysed_selection_is_fetched_on_demand(monkeypatch):
    client = StubClient()
    app = run_app(
        monkeypatch,
        client,
        {
            "page": "Candidate",
            "job_description": JOB,
            "ranking": RANKING,
            "selected_candidate": "sarah_wilson",
        },
    )

    assert not app.exception
    assert "analyze_candidate:sarah_wilson" in client.calls
    assert app.session_state["analyses"]["sarah_wilson"] == ANALYSIS


def test_a_candidate_with_no_evidence_says_so(monkeypatch):
    """Missing evidence is stated, not silently rendered as an empty section."""
    thin = dict(ANALYSIS, evidence=[])
    app = run_app(
        monkeypatch,
        StubClient(),
        {
            "page": "Candidate",
            "job_description": JOB,
            "ranking": RANKING,
            "analyses": {"sarah_wilson": thin},
            "selected_candidate": "sarah_wilson",
        },
    )
    assert "No evidence was retrieved" in text_of(app)


# --------------------------------------------------------------------------
# The whole session
# --------------------------------------------------------------------------


def test_start_over_clears_the_session(monkeypatch):
    app = run_app(
        monkeypatch,
        StubClient(),
        {
            "page": "Ranking",
            "job_description": JOB,
            "ranking": RANKING,
            "analyses": {"sarah_wilson": ANALYSIS},
        },
    )

    reset = next(button for button in app.sidebar.button if button.label == "Start over")
    reset.click().run()

    assert app.session_state["job_description"] == ""
    assert app.session_state["ranking"] is None
    assert app.session_state["analyses"] == {}


@pytest.mark.parametrize("page", ["Overview", "Screening", "Ranking", "Candidate"])
def test_no_page_ever_calls_similarity_a_probability(monkeypatch, page: str):
    """The one claim the whole project has been careful never to make."""
    app = run_app(
        monkeypatch,
        StubClient(),
        {
            "page": page,
            "job_description": JOB,
            "ranking": RANKING,
            "analyses": {"sarah_wilson": ANALYSIS},
            "selected_candidate": "sarah_wilson",
        },
    )
    rendered = text_of(app).lower()

    # Strip the disclaimers, then nothing resembling a probability claim
    # may remain.
    stripped = rendered.replace("not a hiring probability", "")
    stripped = stripped.replace("not a probability of being hired", "")
    stripped = stripped.replace("not a score and not a probability", "")
    stripped = stripped.replace("not a probability", "")

    for phrase in ("probability", "chance of being hired", "likelihood of"):
        assert phrase not in stripped, f"{page} implies probability: {phrase}"


@pytest.mark.parametrize("page", ["Ranking", "Candidate"])
def test_similarity_never_appears_without_its_explanation(monkeypatch, page: str):
    app = run_app(
        monkeypatch,
        StubClient(),
        {
            "page": page,
            "job_description": JOB,
            "ranking": RANKING,
            "analyses": {"sarah_wilson": ANALYSIS},
            "selected_candidate": "sarah_wilson",
        },
    )
    rendered = text_of(app)

    assert "60.24%" in rendered
    assert "ranking signal" in rendered


# --------------------------------------------------------------------------
# Navigation between pages
# --------------------------------------------------------------------------


def test_ranking_moves_the_recruiter_to_the_ranking_page(monkeypatch):
    """A programmatic move must not be undone by the sidebar's sticky value."""
    app = run_app(
        monkeypatch, StubClient(), {"page": "Screening", "job_description": JOB}
    )
    rank = next(button for button in app.button if button.label == "Rank candidates")
    rank.click().run()

    assert app.session_state["page"] == "Ranking"
    assert app.session_state["ranking"] == RANKING


def test_opening_a_candidate_moves_to_the_detail_page(monkeypatch):
    app = run_app(
        monkeypatch,
        StubClient(),
        {"page": "Ranking", "job_description": JOB, "ranking": RANKING},
    )
    view = next(button for button in app.button if button.label == "View full analysis")
    view.click().run()

    assert app.session_state["page"] == "Candidate"
    assert app.session_state["selected_candidate"] == "sarah_wilson"


def test_the_sidebar_still_navigates(monkeypatch):
    app = run_app(monkeypatch, StubClient())
    app.sidebar.radio[0].set_value("Screening").run()

    assert app.session_state["page"] == "Screening"
    assert not app.exception
