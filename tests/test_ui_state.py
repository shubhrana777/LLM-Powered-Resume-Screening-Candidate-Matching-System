"""Tests for the dashboard's session state and ranking-row construction.

The behaviour worth protecting here is invalidation. A ranking and its analyses
belong to one job description; if the recruiter edits the role and the old
numbers stay on screen under the new heading, the dashboard is lying. Several
tests below exist only to prove that cannot happen.
"""

from __future__ import annotations

import pytest

from app.ui import state as ui_state
from app.ui.formatting import NOT_ANALYZED
from app.ui.pages import build_ranking_rows

RANKING = {
    "results": [
        {"rank": 1, "candidate": "Sarah Wilson", "candidate_id": "sarah_wilson", "similarity_score": 0.61},
        {"rank": 2, "candidate": "James Patel", "candidate_id": "james_patel", "similarity_score": 0.47},
        {"rank": 3, "candidate": "Nina Volkov", "candidate_id": "nina_volkov", "similarity_score": 0.12},
    ],
    "count": 3,
    "candidates_considered": 3,
    "score_note": "Cosine similarity ... not a probability ...",
}

ANALYSIS = {
    "candidate": "Sarah Wilson",
    "candidate_id": "sarah_wilson",
    "recommendation": "STRONG_MATCH",
    "matched_skills": ["Python", "SQL", "Excel"],
    "skill_gaps": ["Tableau"],
    "experience_assessment": "The resume states 4 years; the job asks for 3 years. Requirement met: yes.",
    "is_grounded": True,
}


@pytest.fixture
def state() -> dict:
    fresh: dict = {}
    ui_state.init_state(fresh)
    return fresh


# --------------------------------------------------------------------------
# Initialisation
# --------------------------------------------------------------------------


def test_init_creates_every_key_the_pages_read(state):
    for key in ("page", "job_description", "ranking", "analyses", "selected_candidate"):
        assert key in state


def test_init_does_not_overwrite_existing_values():
    existing = {"job_description": "already typed"}
    ui_state.init_state(existing)
    assert existing["job_description"] == "already typed"


def test_init_gives_each_session_its_own_containers():
    first, second = {}, {}
    ui_state.init_state(first)
    ui_state.init_state(second)

    first["analyses"]["x"] = {}

    assert second["analyses"] == {}


def test_the_default_page_is_the_overview(state):
    assert ui_state.current_page(state) == "Overview"


# --------------------------------------------------------------------------
# Job description drives invalidation
# --------------------------------------------------------------------------


def test_a_job_description_is_stored_stripped(state):
    ui_state.set_job_description(state, "  Financial analyst  ")
    assert ui_state.get_job_description(state) == "Financial analyst"


def test_setting_a_new_job_description_reports_the_change(state):
    assert ui_state.set_job_description(state, "Analyst") is True


def test_setting_the_same_job_description_reports_no_change(state):
    ui_state.set_job_description(state, "Analyst")
    assert ui_state.set_job_description(state, "Analyst") is False


def test_whitespace_only_differences_are_not_a_change(state):
    ui_state.set_job_description(state, "Analyst")
    assert ui_state.set_job_description(state, "  Analyst  ") is False


def test_changing_the_role_discards_the_ranking(state):
    ui_state.set_job_description(state, "Analyst")
    ui_state.set_ranking(state, RANKING)

    ui_state.set_job_description(state, "Backend engineer")

    assert ui_state.get_ranking(state) is None


def test_changing_the_role_discards_every_analysis(state):
    ui_state.set_job_description(state, "Analyst")
    ui_state.store_analysis(state, "sarah_wilson", ANALYSIS)

    ui_state.set_job_description(state, "Backend engineer")

    assert ui_state.all_analyses(state) == {}


def test_changing_the_role_clears_the_selection(state):
    ui_state.set_job_description(state, "Analyst")
    ui_state.select_candidate(state, "sarah_wilson")

    ui_state.set_job_description(state, "Backend engineer")

    assert ui_state.get_selected_candidate(state) is None


def test_re_saving_the_same_role_keeps_the_results(state):
    """Results are only invalid when the role actually changed."""
    ui_state.set_job_description(state, "Analyst")
    ui_state.set_ranking(state, RANKING)

    ui_state.set_job_description(state, "Analyst")

    assert ui_state.get_ranking(state) is not None


def test_a_missing_job_description_is_reported(state):
    assert ui_state.has_job_description(state) is False
    ui_state.set_job_description(state, "Analyst")
    assert ui_state.has_job_description(state) is True


# --------------------------------------------------------------------------
# Analyses
# --------------------------------------------------------------------------


def test_an_analysis_can_be_stored_and_read_back(state):
    ui_state.store_analysis(state, "sarah_wilson", ANALYSIS)
    assert ui_state.get_analysis(state, "sarah_wilson") == ANALYSIS


def test_an_unanalysed_candidate_returns_none(state):
    assert ui_state.get_analysis(state, "nobody") is None


def test_analyses_are_counted(state):
    ui_state.store_analysis(state, "a", ANALYSIS)
    ui_state.store_analysis(state, "b", ANALYSIS)
    assert ui_state.analysed_count(state) == 2


# --------------------------------------------------------------------------
# Navigation
# --------------------------------------------------------------------------


@pytest.mark.parametrize("page", ui_state.PAGES)
def test_every_page_can_be_reached(state, page: str):
    ui_state.goto(state, page)
    assert ui_state.current_page(state) == page


def test_an_unknown_page_is_ignored(state):
    ui_state.goto(state, "Nowhere")
    assert ui_state.current_page(state) == "Overview"


def test_a_corrupted_page_value_falls_back(state):
    state["page"] = 42
    assert ui_state.current_page(state) == "Overview"


# --------------------------------------------------------------------------
# Step progress
# --------------------------------------------------------------------------


def test_no_steps_are_complete_at_the_start(state):
    assert ui_state.completed_steps(state) == (False, False, False)


def test_describing_the_role_completes_the_first_step(state):
    ui_state.set_job_description(state, "Analyst")
    assert ui_state.completed_steps(state) == (True, False, False)


def test_ranking_completes_the_second_step(state):
    ui_state.set_job_description(state, "Analyst")
    ui_state.set_ranking(state, RANKING)
    assert ui_state.completed_steps(state) == (True, True, False)


def test_analysing_completes_the_third_step(state):
    """Ranking and analysis are separate steps, and the indicator says so."""
    ui_state.set_job_description(state, "Analyst")
    ui_state.set_ranking(state, RANKING)
    ui_state.store_analysis(state, "sarah_wilson", ANALYSIS)
    assert ui_state.completed_steps(state) == (True, True, True)


def test_ranking_alone_does_not_complete_analysis(state):
    ui_state.set_job_description(state, "Analyst")
    ui_state.set_ranking(state, RANKING)
    assert ui_state.completed_steps(state)[2] is False


# --------------------------------------------------------------------------
# Ranking rows
# --------------------------------------------------------------------------


def test_every_ranked_candidate_becomes_a_row():
    rows = build_ranking_rows(RANKING["results"], {})
    assert len(rows) == 3


def test_rank_comes_straight_from_the_api():
    rows = build_ranking_rows(RANKING["results"], {})
    assert rows[0]["Rank"] == 1


def test_the_table_holds_similarity_scaled_for_display():
    """Scaled to 0-100 so the column sorts numerically and renders as a percent."""
    rows = build_ranking_rows(RANKING["results"], {})
    assert rows[0]["Similarity"] == pytest.approx(61.0)


def test_the_row_keeps_the_untouched_backend_value():
    """The raw cosine value travels with the row for charts and detail views."""
    rows = build_ranking_rows(RANKING["results"], {})
    assert rows[0]["similarity_score"] == 0.61


def test_an_unanalysed_candidate_is_marked_not_analyzed_yet():
    rows = build_ranking_rows(RANKING["results"], {})
    assert rows[0]["Skill coverage"] == "Not analyzed yet"
    assert rows[0]["Recommendation"] == "Not analyzed yet"
    assert rows[0]["analysed"] is False


def test_an_unanalysed_candidate_still_has_a_similarity():
    """Ranking and analysis are separate: one can exist without the other."""
    rows = build_ranking_rows(RANKING["results"], {})
    assert rows[0]["Similarity"] is not None
    assert rows[0]["Recommendation"] == "Not analyzed yet"


def test_an_unanalysed_candidate_is_never_shown_as_zero_coverage():
    """Not analysed and no matching skills are different facts."""
    rows = build_ranking_rows(RANKING["results"], {})
    assert rows[0]["Skill coverage"] != "0 / 0"
    assert "0" not in rows[0]["Skill coverage"]


def test_an_analysed_candidate_gains_its_columns():
    rows = build_ranking_rows(RANKING["results"], {"sarah_wilson": ANALYSIS})
    row = rows[0]

    assert row["Skill coverage"] == "3 / 4"
    assert row["Recommendation"] == "Strong match"
    assert row["Experience"] == "Requirement met"
    assert row["analysed"] is True


def test_analysing_one_candidate_leaves_the_others_alone():
    rows = build_ranking_rows(RANKING["results"], {"sarah_wilson": ANALYSIS})
    assert rows[1]["Recommendation"] == NOT_ANALYZED


def test_a_band_is_attached_to_every_row():
    rows = build_ranking_rows(RANKING["results"], {})
    assert rows[0]["Band"] == "Strong similarity"
    assert rows[2]["Band"] == "Low similarity"


def test_rows_carry_the_id_needed_to_open_a_candidate():
    rows = build_ranking_rows(RANKING["results"], {})
    assert rows[0]["candidate_id"] == "sarah_wilson"


def test_an_empty_ranking_produces_no_rows():
    assert build_ranking_rows([], {}) == []
