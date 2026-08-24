"""The four screens.

Each ``render_*`` function takes the API client, the session state and the
candidate pool, and draws one page. They fetch through
:class:`~app.ui.api_client.ScreeningAPIClient` and draw through
:mod:`app.ui.components`; there is no HTTP and no markup written inline here.

The screening flow is three steps -- describe the role, rank the pool, analyse
the shortlist -- and every screen shows where the recruiter is in it. Ranking
and analysis are deliberately separate: ranking is cheap and covers everyone,
analysis costs a model call each and is run on a shortlist. A candidate who has
not reached the second step reads *Not analyzed yet*, never zero and never a
failure.
"""

from __future__ import annotations

from typing import Any, Mapping, MutableMapping, Sequence

import streamlit as st

from app.ui import state as ui_state
from app.ui.api_client import (
    APIClientError,
    APIError,
    APITimeoutError,
    APIUnavailableError,
    ScreeningAPIClient,
)
from app.ui.components import (
    api_unavailable_state,
    badge,
    badge_html,
    card,
    chips,
    coverage_chart,
    empty_state,
    error_state,
    evidence_block,
    footnote,
    generated_block,
    lead_candidate,
    masthead,
    ranking_chart,
    ranking_table,
    rule,
    section,
    similarity_footnote,
    stat_cards,
    step_indicator,
)
from app.ui.formatting import (
    NOT_ANALYZED,
    NOT_ANALYZED_HINT,
    SIMILARITY_MEANING,
    experience_status,
    format_coverage,
    format_similarity,
    format_similarity_raw,
    grounding_label,
    plural,
    recommendation_label,
    recommendation_tone,
    score_band,
    to_percent,
    truncate,
)

__all__ = [
    "MAX_UPLOAD_MB",
    "STEP_LABELS",
    "render_overview",
    "render_screening",
    "render_ranking",
    "render_candidate",
    "render_resumes",
    "render_page",
    "build_ranking_rows",
]

MAX_UPLOAD_MB = 5

STEP_LABELS = ("Describe role", "Rank candidates", "Analyze candidates")

TABLE_COLUMNS = (
    "Rank",
    "Candidate",
    "Similarity",
    "Band",
    "Skill coverage",
    "Experience",
    "Recommendation",
)


def _report(error: APIClientError, base_url: str) -> None:
    """Render whichever failure state matches ``error``."""
    if isinstance(error, APIUnavailableError):
        api_unavailable_state(base_url)
    elif isinstance(error, APITimeoutError):
        st.warning(error.message, icon=":material/hourglass_top:")
    elif isinstance(error, APIError):
        error_state(error.message, error.details)
    else:  # pragma: no cover - the three subclasses are exhaustive today
        error_state(error.message)


def _steps(state: MutableMapping[str, Any]) -> tuple[int, tuple]:
    """Return the active step index and the labelled step states."""
    done = ui_state.completed_steps(state)
    active = next((index for index, complete in enumerate(done) if not complete), len(done) - 1)
    return active, tuple(zip(STEP_LABELS, done))


def _strong_matches(analyses: Mapping[str, Mapping[str, Any]]) -> int:
    """Count analysed candidates the model labelled a strong match."""
    return sum(1 for item in analyses.values() if item.get("recommendation") == "STRONG_MATCH")


def _mean_similarity(results: Sequence[Mapping[str, Any]]) -> float | None:
    """Mean similarity across a ranking, or ``None`` when there is none."""
    scores = [
        float(row["similarity_score"])
        for row in results
        if isinstance(row.get("similarity_score"), (int, float))
    ]
    return sum(scores) / len(scores) if scores else None


# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------


def render_overview(
    client: ScreeningAPIClient,
    state: MutableMapping[str, Any],
    pool: Mapping[str, Any],
) -> None:
    """Render the dashboard home screen.

    Every figure is derived from a real API response or from work done in this
    session. Where a number does not exist yet the screen says so and points at
    the step that would produce it.

    Args:
        client: The API client.
        state: Session state.
        pool: The ``GET /candidates`` response.
    """
    masthead(
        "Screening overview",
        "Where this session stands: the candidate pool, the role being screened for, "
        "and what has been analysed so far.",
    )

    candidates = list(pool.get("candidates") or [])
    ranking = ui_state.get_ranking(state)
    analyses = ui_state.all_analyses(state)
    results = list(ranking.get("results") or []) if ranking else []
    mean = _mean_similarity(results)

    stat_cards(
        [
            (
                "Candidates in pool",
                str(len(candidates)),
                "PDF resumes the backend can rank.",
            ),
            (
                "Ranked",
                str(len(results)) if results else NOT_ANALYZED,
                "Candidates scored against the current role.",
            ),
            (
                "Analyzed",
                str(len(analyses)) if analyses else NOT_ANALYZED,
                "Full analyses run this session. One model call each.",
            ),
            (
                "Average similarity",
                format_similarity(mean),
                "Mean across the current ranking. Comparable only within it.",
            ),
        ]
    )

    active, steps = _steps(state)
    st.markdown("")
    step_indicator(steps, active)

    left, right = st.columns([3, 2], gap="medium")

    with left:
        section("Current role", "The job description every candidate is scored against.")
        job = ui_state.get_job_description(state)
        if job:
            card("Job description", f"<p class='rs-wrap'>{truncate(job, 420)}</p>")
            if results:
                st.markdown(
                    f"<p class='rs-note'>Ranked {plural(len(results), 'candidate')} of "
                    f"{ranking.get('candidates_considered', len(results))} considered.</p>",
                    unsafe_allow_html=True,
                )
            else:
                st.info("No ranking yet for this role.", icon=":material/info:")
        else:
            empty_state(
                "No role described yet",
                "Add a job description on the Screening page to begin. "
                "Everything else follows from it.",
            )

        st.markdown("")
        if st.button("Go to screening", type="primary"):
            ui_state.goto(state, "Screening")
            st.rerun()

    with right:
        section("Candidate pool", "Resumes currently loaded on the server.")
        if candidates:
            for candidate in candidates[:6]:
                card(
                    candidate.get("filename", ""),
                    f"<p class='rs-wrap' style='margin:0;font-weight:600'>"
                    f"{candidate.get('name', candidate.get('candidate_id', ''))}</p>"
                    f"<p class='rs-note' style='margin-top:4px'>"
                    f"{candidate.get('text_length', 0):,} characters extracted</p>",
                )
            if len(candidates) > 6:
                st.caption(f"And {len(candidates) - 6} more.")
        else:
            empty_state("No resumes loaded", "Upload PDF resumes on the Screening page.")

        unreadable = list(pool.get("unreadable") or [])
        if unreadable:
            st.warning(
                f"{plural(len(unreadable), 'file')} in the pool could not be read.",
                icon=":material/warning:",
            )
            for item in unreadable:
                st.caption(f"{item.get('filename', 'unknown')} — {item.get('reason', '')}")


# ---------------------------------------------------------------------------
# Screening
# ---------------------------------------------------------------------------


def render_screening(
    client: ScreeningAPIClient,
    state: MutableMapping[str, Any],
    pool: Mapping[str, Any],
) -> None:
    """Render the job-description, upload and ranking workflow.

    Args:
        client: The API client.
        state: Session state.
        pool: The ``GET /candidates`` response.
    """
    masthead(
        "Screen candidates",
        "Describe the role, make sure the resumes are loaded, then rank the pool against it.",
    )

    candidates = list(pool.get("candidates") or [])
    active, steps = _steps(state)
    step_indicator(steps, active)

    _job_description_form(state)
    rule()
    _candidate_pool_panel(client, state, candidates)
    rule()
    _run_ranking(client, state, len(candidates))


def _job_description_form(state: MutableMapping[str, Any]) -> None:
    """Step 1: capture and validate the job description."""
    section(
        "Step 1 · Describe the role",
        "Required skills and any minimum years of experience are read from this text, "
        "so be explicit about both.",
    )

    if not ui_state.has_job_description(state):
        empty_state(
            "No role described yet",
            "Paste a job description below. Nothing can be ranked or analysed until "
            "there is a role to score against.",
        )
        st.markdown("")

    with st.form("job_description_form", border=False):
        text = st.text_area(
            "Job description",
            value=ui_state.get_job_description(state),
            height=240,
            max_chars=20_000,
            placeholder="Paste the full job description here, including required skills "
            "and any minimum years of experience.",
            label_visibility="visible",
        )
        submitted = st.form_submit_button("Save job description", type="primary")

    if not submitted:
        return

    if not text.strip():
        st.error(
            "Enter a job description before saving. It is what candidates are ranked against.",
            icon=":material/error:",
        )
        return

    if ui_state.set_job_description(state, text):
        st.success("Job description saved.", icon=":material/check_circle:")
        st.caption(
            "Any earlier ranking and analyses were cleared: they belonged to a different role."
        )
    else:
        st.info("Job description unchanged.", icon=":material/info:")


def _candidate_pool_panel(
    client: ScreeningAPIClient,
    state: MutableMapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
) -> None:
    """The candidate pool: what is loaded, and how to add more."""
    section(
        "Candidate pool",
        f"{plural(len(candidates), 'resume')} currently loaded on the server.",
    )

    uploads = st.file_uploader(
        "Add PDF resumes",
        type=["pdf"],
        accept_multiple_files=True,
        help=f"Text-based PDF resumes only, up to {MAX_UPLOAD_MB} MB each. "
        "A scanned image has no selectable text to read.",
    )

    st.caption(
        "PDF resumes are accepted. Uploaded files are added to the candidate pool so "
        "they can be ranked and analysed, and are stored on the server that runs the "
        "API rather than in this browser. A file that cannot be read is reported with "
        "the reason and nothing is added."
    )

    if uploads:
        listed = "".join(
            f"<span class='rs-chip rs-chip--have'>{upload.name} · {upload.size / 1024:,.0f} KB</span>"
            for upload in uploads
        )
        st.markdown(f"<div class='rs-chips'>{listed}</div>", unsafe_allow_html=True)
        st.markdown("")

    if st.button(
        f"Add {plural(len(uploads), 'resume')} to pool" if uploads else "Add resumes to pool",
        type="primary" if uploads else "secondary",
        disabled=not uploads,
        help=None if uploads else "Select at least one PDF first.",
    ):
        _perform_uploads(client, state, uploads or [])

    _render_upload_results(state)


def _perform_uploads(
    client: ScreeningAPIClient,
    state: MutableMapping[str, Any],
    uploads: Sequence[Any],
) -> None:
    """Send each selected file to the API, reporting progress as it goes."""
    results: list[dict[str, Any]] = []
    progress = st.progress(0.0, text="Uploading…")

    for index, upload in enumerate(uploads, start=1):
        progress.progress(
            (index - 1) / len(uploads),
            text=f"Uploading {upload.name} ({index} of {len(uploads)})",
        )
        try:
            response = client.upload_resume(upload.name, upload.getvalue(), store=True)
        except APIClientError as error:
            results.append({"filename": upload.name, "ok": False, "message": error.message})
        else:
            results.append(
                {
                    "filename": upload.name,
                    "ok": True,
                    "candidate_id": response.get("candidate_id"),
                    "message": f"{response.get('text_length', 0):,} characters extracted",
                }
            )

    progress.progress(1.0, text="Upload complete")
    state["upload_results"] = results

    if any(item["ok"] for item in results):
        # A new resume changes the pool, so anything ranked before is stale.
        ui_state.clear_results(state)

    st.rerun()


def _render_upload_results(state: MutableMapping[str, Any]) -> None:
    """Show the outcome of the last upload batch."""
    results = state.get("upload_results") or []
    if not results:
        return

    accepted = [item for item in results if item["ok"]]
    rejected = [item for item in results if not item["ok"]]

    if accepted:
        st.success(
            f"{plural(len(accepted), 'resume')} added to the pool.",
            icon=":material/check_circle:",
        )
        for item in accepted:
            st.markdown(
                f"<p class='rs-note rs-wrap'><strong>{item['filename']}</strong> → "
                f"<code>{item.get('candidate_id', '')}</code> · {item['message']}</p>",
                unsafe_allow_html=True,
            )

    if rejected:
        st.error(f"{plural(len(rejected), 'file')} could not be added.", icon=":material/error:")
        for item in rejected:
            st.markdown(
                f"<p class='rs-note rs-wrap'><strong>{item['filename']}</strong> — "
                f"{item['message']}</p>",
                unsafe_allow_html=True,
            )


def _run_ranking(
    client: ScreeningAPIClient,
    state: MutableMapping[str, Any],
    candidate_count: int,
) -> None:
    """Step 2: rank the pool against the saved job description."""
    section(
        "Step 2 · Rank candidates",
        "Scores every resume against the role by semantic similarity. Fast, and it "
        "covers the whole pool.",
    )

    if not ui_state.has_job_description(state):
        empty_state("Nothing to rank against yet", "Save a job description in step 1 first.")
        return

    if candidate_count == 0:
        empty_state("No candidates in the pool", "Add at least one PDF resume above.")
        return

    columns = st.columns([3, 1], gap="medium")
    with columns[0]:
        top_k = st.slider(
            "How many candidates to rank",
            min_value=1,
            max_value=max(1, min(candidate_count, 100)),
            value=min(client.settings.default_top_k, candidate_count),
            help="Ranking is cheap. Analysis, which comes next, costs one model call "
            "per candidate.",
        )
    with columns[1]:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        run = st.button("Rank candidates", type="primary")

    if not run:
        return

    with st.spinner("Embedding the job description and ranking the pool…"):
        try:
            ranking = client.match_candidates(ui_state.get_job_description(state), top_k)
        except APIClientError as error:
            _report(error, client.base_url)
            return

    ui_state.set_ranking(state, ranking)
    ui_state.goto(state, "Ranking")
    st.rerun()


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------


def build_ranking_rows(
    results: Sequence[Mapping[str, Any]],
    analyses: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Merge match results with any analyses into display rows.

    Skill coverage, experience and recommendation come from the analysis step,
    which is separate from ranking and costs a model call. A candidate who has
    not been through it reads *Not analyzed yet* -- a step not taken, not a
    failure and not a zero.

    ``Similarity`` is the cosine value scaled to 0-100 so the column sorts
    numerically while rendering as a percentage. Display only; the API response
    is untouched.

    Args:
        results: Match results from the API.
        analyses: Cached analyses, keyed by candidate id.

    Returns:
        One dictionary per candidate, ready for the table.
    """
    rows: list[dict[str, Any]] = []

    for result in results:
        candidate_id = str(result.get("candidate_id", ""))
        analysis = analyses.get(candidate_id) or {}
        score = result.get("similarity_score")

        rows.append(
            {
                "Rank": result.get("rank"),
                "Candidate": result.get("candidate") or candidate_id,
                "Similarity": to_percent(score),
                "Band": score_band(score).label,
                "Skill coverage": (
                    format_coverage(analysis.get("matched_skills"), analysis.get("skill_gaps"))
                    if analysis
                    else NOT_ANALYZED
                ),
                "Experience": experience_status(analysis.get("experience_assessment")).label,
                "Recommendation": (
                    recommendation_label(analysis.get("recommendation"))
                    if analysis
                    else NOT_ANALYZED
                ),
                "candidate_id": candidate_id,
                "similarity_score": score,
                "analysed": bool(analysis),
            }
        )

    return rows


def render_ranking(
    client: ScreeningAPIClient,
    state: MutableMapping[str, Any],
    pool: Mapping[str, Any],
) -> None:
    """Render the ranked candidate list: summary, filters, table, charts.

    Args:
        client: The API client.
        state: Session state.
        pool: The ``GET /candidates`` response.
    """
    masthead(
        "Candidate ranking",
        "Every resume scored against the current role by semantic similarity. "
        "Analyse a candidate to add skill coverage, experience fit and a recommendation.",
    )

    ranking = ui_state.get_ranking(state)
    if ranking is None:
        empty_state(
            "Nothing ranked yet",
            "Save a job description and rank the pool on the Screening page.",
        )
        st.markdown("")
        if st.button("Go to screening", type="primary"):
            ui_state.goto(state, "Screening")
            st.rerun()
        return

    results = list(ranking.get("results") or [])
    if not results:
        empty_state(
            "No candidates matched",
            "The pool is empty, or every resume failed to parse. Add resumes on the "
            "Screening page.",
        )
        return

    analyses = ui_state.all_analyses(state)
    rows = build_ranking_rows(results, analyses)

    # --- Summary ---------------------------------------------------------
    mean = _mean_similarity(results)
    stat_cards(
        [
            ("Candidates ranked", str(len(results)), "Scored against the current role."),
            (
                "Analyzed",
                str(len(analyses)) if analyses else NOT_ANALYZED,
                NOT_ANALYZED_HINT if not analyses else "Full analyses run this session.",
            ),
            (
                "Strong matches",
                str(_strong_matches(analyses)) if analyses else NOT_ANALYZED,
                "Analysed candidates labelled a strong match. A coarse label, not a decision.",
            ),
            ("Average similarity", format_similarity(mean), "Mean across this ranking."),
        ]
    )

    st.markdown("")
    lead = rows[0]
    lead_candidate(
        str(lead["Candidate"]),
        lead["similarity_score"],
        extra=(
            f"Skill coverage {lead['Skill coverage']}"
            if lead["analysed"]
            else NOT_ANALYZED
        ),
    )

    rule()

    # --- Analysis --------------------------------------------------------
    _analysis_controls(client, state, results)

    rule()

    # --- Filters and table ----------------------------------------------
    section("Ranked candidates", "Sort or filter the shortlist, then open a candidate.")
    visible = _filter_controls(rows)

    if not visible:
        empty_state(
            "No candidates match these filters",
            "Widen the similarity range or clear the recommendation filter.",
        )
        return

    ranking_table([{key: row[key] for key in TABLE_COLUMNS} for row in visible])
    similarity_footnote()

    # --- Charts ----------------------------------------------------------
    st.markdown("")
    left, right = st.columns(2, gap="medium")

    with left:
        section("Similarity spread", "How far ahead the leader is, and where the field falls away.")
        ranking_chart(
            [
                {"candidate": row["Candidate"], "similarity_score": row["similarity_score"]}
                for row in visible
            ]
        )

    with right:
        analysed_rows = [
            analyses[row["candidate_id"]] for row in visible if row["candidate_id"] in analyses
        ]
        if analysed_rows:
            section(
                "Skill coverage",
                "Similarity and coverage answer different questions, and often disagree.",
            )
            coverage_chart(analysed_rows)
        else:
            section("Skill coverage", "Available once candidates have been analysed.")
            empty_state("Nothing analysed yet", NOT_ANALYZED_HINT)

    rule()
    _selection_controls(state, visible)


def _analysis_controls(
    client: ScreeningAPIClient,
    state: MutableMapping[str, Any],
    results: Sequence[Mapping[str, Any]],
) -> None:
    """Offer to analyse the top candidates, with progress."""
    analysed = ui_state.analysed_count(state)
    section(
        "Step 3 · Analyze candidates",
        f"{plural(analysed, 'candidate')} analysed of {len(results)} ranked. "
        "Analysis retrieves evidence from each resume and calculates skill coverage, "
        "experience fit and a recommendation — one model call per candidate.",
    )

    columns = st.columns([3, 1], gap="medium")
    with columns[0]:
        how_many = st.slider(
            "Analyze the top",
            min_value=1,
            max_value=len(results),
            value=min(3, len(results)),
            help="Runs the retrieval-augmented analysis for the highest-ranked candidates. "
            "Already-analysed candidates are skipped.",
        )
    with columns[1]:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        run = st.button("Analyze", type="primary")

    if not run:
        return

    targets = [str(row.get("candidate_id", "")) for row in results[:how_many]]
    job = ui_state.get_job_description(state)
    progress = st.progress(0.0, text="Analyzing…")
    failures: list[str] = []

    for index, candidate_id in enumerate(targets, start=1):
        progress.progress(
            (index - 1) / len(targets),
            text=f"Analyzing {candidate_id} ({index} of {len(targets)})",
        )
        if ui_state.get_analysis(state, candidate_id):
            continue
        try:
            analysis = client.analyze_candidate(candidate_id, job)
        except APIClientError as error:
            failures.append(f"{candidate_id}: {error.message}")
        else:
            ui_state.store_analysis(state, candidate_id, analysis)

    progress.progress(1.0, text="Analysis complete")

    if failures:
        error_state("Some candidates could not be analysed.", failures)
    else:
        st.rerun()


def _filter_controls(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Render filter and sort controls, and apply them.

    Args:
        rows: All display rows.

    Returns:
        The rows to show, in the chosen order.
    """
    with st.expander("Filter and sort", expanded=True):
        columns = st.columns([2, 2, 2], gap="medium")

        with columns[0]:
            query = st.text_input(
                "Search by name", placeholder="Type part of a name", key="ranking_search"
            )
        with columns[1]:
            recommendations = sorted({row["Recommendation"] for row in rows})
            chosen = st.multiselect(
                "Recommendation",
                options=recommendations,
                default=[],
                help="Leave empty to show every candidate, analysed or not.",
            )
        with columns[2]:
            sort_by = st.selectbox(
                "Sort by",
                options=["Rank", "Similarity (high to low)", "Candidate (A-Z)", "Skill coverage"],
            )

        minimum = st.slider(
            "Minimum similarity",
            min_value=0,
            max_value=100,
            value=0,
            step=5,
            format="%d%%",
            help=SIMILARITY_MEANING + " Filtering hides candidates; it changes no score.",
        )

    visible = [
        row
        for row in rows
        if (not query or query.strip().casefold() in str(row["Candidate"]).casefold())
        and (not chosen or row["Recommendation"] in chosen)
        and (row["Similarity"] is None or float(row["Similarity"]) >= minimum)
    ]

    if sort_by == "Similarity (high to low)":
        visible.sort(key=lambda row: float(row["Similarity"] or 0.0), reverse=True)
    elif sort_by == "Candidate (A-Z)":
        visible.sort(key=lambda row: str(row["Candidate"]).casefold())
    elif sort_by == "Skill coverage":
        # Unanalysed candidates have no coverage; they sort last rather than
        # being treated as zero coverage.
        visible.sort(key=_coverage_sort_key, reverse=True)
    else:
        visible.sort(key=lambda row: row["Rank"] or 0)

    return visible


def _coverage_sort_key(row: Mapping[str, Any]) -> float:
    """Sort key for skill coverage, placing unanalysed candidates last."""
    raw = str(row.get("Skill coverage", ""))
    if "/" not in raw:
        return -1.0
    have, _, total = raw.partition("/")
    try:
        denominator = float(total)
        return float(have) / denominator if denominator else -1.0
    except ValueError:
        return -1.0


def _selection_controls(
    state: MutableMapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> None:
    """Let the recruiter open one candidate's detail view."""
    section("Open a candidate", "See skills, experience, education, evidence and the summary.")

    labels = {f"{row['Rank']}. {row['Candidate']}": row["candidate_id"] for row in rows}
    columns = st.columns([3, 1], gap="medium")

    with columns[0]:
        chosen = st.selectbox("Candidate", options=list(labels), key="candidate_picker")
    with columns[1]:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        open_it = st.button("View full analysis", type="primary")

    if open_it:
        ui_state.select_candidate(state, labels[chosen])
        ui_state.goto(state, "Candidate")
        st.rerun()


# ---------------------------------------------------------------------------
# Candidate detail
# ---------------------------------------------------------------------------


def render_candidate(
    client: ScreeningAPIClient,
    state: MutableMapping[str, Any],
    pool: Mapping[str, Any],
) -> None:
    """Render one candidate's full analysis.

    Args:
        client: The API client.
        state: Session state.
        pool: The ``GET /candidates`` response.
    """
    candidate_id = ui_state.get_selected_candidate(state)

    if not candidate_id:
        masthead("Candidate detail", "Select a candidate from the ranking to see their analysis.")
        empty_state("No candidate selected", "Pick one on the Ranking page.")
        st.markdown("")
        if st.button("Go to ranking", type="primary"):
            ui_state.goto(state, "Ranking")
            st.rerun()
        return

    analysis = ui_state.get_analysis(state, candidate_id)

    if analysis is None:
        analysis = _fetch_analysis(client, state, candidate_id)
        if analysis is None:
            return

    _render_analysis(state, analysis)


def _fetch_analysis(
    client: ScreeningAPIClient,
    state: MutableMapping[str, Any],
    candidate_id: str,
) -> dict[str, Any] | None:
    """Fetch and cache one analysis, rendering the failure state if it fails."""
    masthead("Candidate detail", f"Analyzing {candidate_id}…")

    with st.spinner("Retrieving evidence from the resume and generating the analysis…"):
        try:
            analysis = client.analyze_candidate(candidate_id, ui_state.get_job_description(state))
        except APIClientError as error:
            _report(error, client.base_url)
            if st.button("Back to ranking"):
                ui_state.goto(state, "Ranking")
                st.rerun()
            return None

    ui_state.store_analysis(state, candidate_id, analysis)
    return analysis


def _render_analysis(state: MutableMapping[str, Any], analysis: Mapping[str, Any]) -> None:
    """Draw a completed analysis."""
    name = str(analysis.get("candidate") or analysis.get("candidate_id", ""))
    candidate_id = analysis.get("candidate_id")

    ranking = ui_state.get_ranking(state)
    results = list((ranking or {}).get("results") or [])
    match = next((row for row in results if row.get("candidate_id") == candidate_id), {})
    score = match.get("similarity_score")
    rank = match.get("rank")

    context = f"Rank {rank} of {len(results)} for the current role" if rank else "Current role"
    masthead(name, context)

    if st.button("Back to ranking", icon=":material/arrow_back:"):
        ui_state.goto(state, "Ranking")
        st.rerun()

    st.markdown("")
    _render_summary_cards(analysis, score)
    similarity_footnote()

    rule()
    _render_skills(analysis)

    rule()
    _render_background(analysis)

    rule()
    _render_interpretation(analysis)

    rule()
    _render_evidence(analysis)


def _render_summary_cards(analysis: Mapping[str, Any], score: float | None) -> None:
    """The four headline figures: similarity, coverage, experience, recommendation."""
    matched = list(analysis.get("matched_skills") or [])
    gaps = list(analysis.get("skill_gaps") or [])
    experience = experience_status(analysis.get("experience_assessment"))
    recommendation = analysis.get("recommendation")

    stat_cards(
        [
            (
                "Similarity",
                format_similarity(score),
                f"Cosine value {format_similarity_raw(score)}. {score_band(score).label}.",
            ),
            (
                "Skill coverage",
                format_coverage(matched, gaps),
                "Skills found on the resume, out of those named in the job description.",
            ),
            (
                "Experience",
                experience.label,
                "Compared only when both the resume and the role state a figure.",
            ),
            (
                "Recommendation",
                recommendation_label(recommendation),
                "A coarse label, not a score and never a hiring decision.",
            ),
        ]
    )

    columns = st.columns(4, gap="small")
    with columns[2]:
        badge(experience.label, experience.tone)
    with columns[3]:
        badge(recommendation_label(recommendation), recommendation_tone(recommendation))


def _render_skills(analysis: Mapping[str, Any]) -> None:
    """Matched skills and gaps, side by side."""
    matched = list(analysis.get("matched_skills") or [])
    gaps = list(analysis.get("skill_gaps") or [])

    section(
        "Skills",
        f"Coverage {format_coverage(matched, gaps)}. Matched against the skills named in "
        "the job description, using the deterministic extractor — not the model.",
    )

    columns = st.columns(2, gap="medium")
    with columns[0]:
        st.markdown("<p class='rs-card__label'>Matched skills</p>", unsafe_allow_html=True)
        chips(matched, kind="have", empty_message="No required skills were supported by the resume.")
    with columns[1]:
        st.markdown("<p class='rs-card__label'>Gaps · missing skills</p>", unsafe_allow_html=True)
        chips(gaps, kind="gap", empty_message="No gaps against the named requirements.")


def _render_background(analysis: Mapping[str, Any]) -> None:
    """Experience and education, both deterministic."""
    columns = st.columns(2, gap="medium")

    with columns[0]:
        section("Experience")
        status = experience_status(analysis.get("experience_assessment"))
        badge(status.label, status.tone)
        st.markdown(
            f"<p class='rs-note rs-wrap'>{analysis.get('experience_assessment', '')}</p>",
            unsafe_allow_html=True,
        )

    with columns[1]:
        section("Education")
        education = list(analysis.get("education") or [])
        if education:
            for entry in education:
                st.markdown(f"- {entry}")
            st.caption("Extracted from the resume text, not generated by the model.")
        else:
            st.markdown(
                "<p class='rs-note'>No degree found in the resume text.</p>",
                unsafe_allow_html=True,
            )


def _render_interpretation(analysis: Mapping[str, Any]) -> None:
    """The model's prose and the validation that was applied to it."""
    section(
        "AI-generated interpretation",
        "Written by a language model from the retrieved evidence below, then checked "
        "against the deterministic profile. Read the evidence before acting on it.",
    )

    generated_block(str(analysis.get("summary", "")))

    grounding = grounding_label(analysis.get("is_grounded"))
    warnings = list(analysis.get("warnings") or [])

    columns = st.columns([1, 3], gap="medium")
    with columns[0]:
        badge(grounding.label, grounding.tone)
    with columns[1]:
        st.caption(f"Generated by {analysis.get('model', 'unknown')}.")

    if warnings:
        with st.expander(f"{plural(len(warnings), 'claim')} corrected before display"):
            for warning in warnings:
                st.markdown(f"- {warning}")
    else:
        st.caption("No unsupported claims were found in the model's response.")

    limitations = list(analysis.get("limitations") or [])
    if limitations:
        with st.expander("What this analysis could not determine"):
            for item in limitations:
                st.markdown(f"- {item}")


def _render_evidence(analysis: Mapping[str, Any]) -> None:
    """The retrieved resume passages the analysis rests on."""
    evidence = list(analysis.get("evidence") or [])

    section(
        "Evidence from resume",
        "Verbatim passages retrieved from this candidate's resume and shown to the model. "
        "Source text only — never the model's reasoning.",
    )

    if not evidence:
        empty_state(
            "No evidence was retrieved",
            "Nothing in this resume was close enough to the job description to retrieve. "
            "Treat the interpretation above with corresponding caution.",
        )
        return

    for item in evidence:
        evidence_block(item)

    footnote(
        "Retrieval is scoped to this candidate alone: passages from another candidate's "
        "resume cannot appear here."
    )


# ---------------------------------------------------------------------------
# Resume pool management
# ---------------------------------------------------------------------------


def render_resumes(
    client: ScreeningAPIClient,
    state: MutableMapping[str, Any],
    pool: Mapping[str, Any],
) -> None:
    """Render the resume pool: what is stored, and how to remove it.

    The pool is server-side and shared across screening sessions, so it is
    managed on its own page rather than buried in the screening flow. Two
    destructive actions live here, and they are deliberately different in
    weight: removing selected resumes is one click, emptying the pool takes a
    confirmation.

    Args:
        client: The API client.
        state: Session state.
        pool: The ``GET /candidates`` response.
    """
    masthead(
        "Resumes",
        "The candidate pool stored on the server. It persists across screening "
        "sessions until you remove something here.",
    )

    candidates = list(pool.get("candidates") or [])
    unreadable = list(pool.get("unreadable") or [])

    stat_cards(
        [
            ("Stored resumes", str(len(candidates)), "Available to rank and analyse."),
            (
                "Unreadable files",
                str(len(unreadable)) if unreadable else "0",
                "In the directory but not parseable, so not candidates.",
            ),
            (
                "Analyzed this session",
                str(ui_state.analysed_count(state)) if ui_state.analysed_count(state) else NOT_ANALYZED,
                "Cleared when you start a new screening session.",
            ),
        ]
    )

    rule()
    _session_controls(state)

    rule()

    if not candidates:
        section("Stored resumes")
        empty_state(
            "The pool is empty",
            "Add PDF resumes on the Screening page. They stay here until you remove them.",
        )
        _render_unreadable(unreadable)
        return

    _resume_list(client, state, candidates)

    rule()
    _clear_pool_controls(client, state, len(candidates))

    _render_unreadable(unreadable)


def _session_controls(state: MutableMapping[str, Any]) -> None:
    """Start a fresh screening session without touching the pool."""
    section(
        "Screening session",
        "A session is one role: the job description, the ranking and the analyses "
        "produced for it. Starting a new one clears that work and keeps every "
        "stored resume.",
    )

    columns = st.columns([2, 3], gap="medium")

    with columns[0]:
        if st.button("New screening session", type="primary", icon=":material/restart_alt:"):
            ui_state.new_session(state)
            ui_state.goto(state, "Screening")
            st.rerun()

    with columns[1]:
        job = ui_state.get_job_description(state)
        if job:
            st.markdown(
                f"<p class='rs-note rs-wrap'>Current session: {truncate(job, 140)}</p>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<p class='rs-note'>No role described in this session yet.</p>",
                unsafe_allow_html=True,
            )
        st.caption("Resumes are not deleted. To empty the pool, use Clear resume pool below.")


def _resume_list(
    client: ScreeningAPIClient,
    state: MutableMapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
) -> None:
    """List every stored resume, with per-row and multi-select deletion."""
    section(
        "Stored resumes",
        f"{plural(len(candidates), 'resume')} in the pool. Removing one deletes it from "
        "the server and drops it from any ranking straight away.",
    )

    selected: list[str] = []

    for candidate in candidates:
        candidate_id = str(candidate.get("candidate_id", ""))
        columns = st.columns([1, 5, 3, 2], gap="small")

        with columns[0]:
            if st.checkbox(
                f"Select {candidate.get('name', candidate_id)}",
                key=f"select_{candidate_id}",
                label_visibility="collapsed",
            ):
                selected.append(candidate_id)

        with columns[1]:
            st.markdown(
                f"<p class='rs-wrap' style='margin:0;font-weight:600'>"
                f"{candidate.get('name', candidate_id)}</p>"
                f"<p class='rs-note rs-wrap' style='margin:2px 0 0'>"
                f"{candidate.get('filename', '')}</p>",
                unsafe_allow_html=True,
            )

        with columns[2]:
            st.markdown(
                f"<p class='rs-note rs-num' style='margin:6px 0 0'>"
                f"{candidate.get('text_length', 0):,} characters</p>",
                unsafe_allow_html=True,
            )

        with columns[3]:
            if st.button(
                "Remove",
                key=f"delete_{candidate_id}",
                help=f"Delete {candidate.get('name', candidate_id)} from the pool.",
            ):
                _delete(client, state, [candidate_id])

    st.markdown("")

    if selected:
        st.caption(f"{plural(len(selected), 'resume')} selected.")
        if st.button(
            f"Remove {plural(len(selected), 'selected resume')}",
            type="primary",
            icon=":material/delete:",
        ):
            _delete(client, state, selected)
    else:
        st.caption("Tick resumes to remove several at once.")


def _clear_pool_controls(
    client: ScreeningAPIClient,
    state: MutableMapping[str, Any],
    count: int,
) -> None:
    """The destructive action, behind an explicit confirmation."""
    section(
        "Clear resume pool",
        "Deletes every stored resume from the server. This cannot be undone, and it "
        "is separate from starting a new screening session.",
    )

    if not state.get("confirm_clear_pool"):
        if st.button("Clear resume pool", icon=":material/warning:"):
            state["confirm_clear_pool"] = True
            st.rerun()
        return

    st.warning(
        f"Delete all {plural(count, 'resume')} from the pool? This cannot be undone.",
        icon=":material/warning:",
    )

    columns = st.columns([2, 2, 4], gap="small")

    with columns[0]:
        if st.button("Yes, delete everything", type="primary"):
            state["confirm_clear_pool"] = False
            _clear(client, state)

    with columns[1]:
        if st.button("Cancel"):
            state["confirm_clear_pool"] = False
            st.rerun()


def _render_unreadable(unreadable: Sequence[Mapping[str, Any]]) -> None:
    """Report files that are present but could not be parsed."""
    if not unreadable:
        return

    section(
        "Unreadable files",
        "Present in the resume directory but not parseable, so they are not "
        "candidates. Usually a scanned image with no selectable text.",
    )
    for item in unreadable:
        st.markdown(
            f"<p class='rs-note rs-wrap'><strong>{item.get('filename', 'unknown')}</strong> — "
            f"{item.get('reason', '')}</p>",
            unsafe_allow_html=True,
        )


def _delete(
    client: ScreeningAPIClient,
    state: MutableMapping[str, Any],
    candidate_ids: Sequence[str],
) -> None:
    """Delete candidates, then forget anything derived from them."""
    try:
        if len(candidate_ids) == 1:
            result = client.delete_candidate(candidate_ids[0])
        else:
            result = client.delete_candidates(candidate_ids)
    except APIClientError as error:
        _report(error, client.base_url)
        return

    deleted = list(result.get("deleted") or [])
    failed = list(result.get("failed") or [])

    # A ranking that included a deleted candidate is no longer describable, and
    # their analysis is about a resume that no longer exists.
    ui_state.forget_candidates(state, deleted)
    _clear_selection_widgets(state, deleted)

    if deleted:
        st.session_state["_delete_notice"] = (
            f"Removed {plural(len(deleted), 'resume')} from the pool."
        )
    if failed:
        st.session_state["_delete_failures"] = failed

    st.rerun()


def _clear(client: ScreeningAPIClient, state: MutableMapping[str, Any]) -> None:
    """Empty the pool, then forget everything derived from it."""
    try:
        result = client.clear_candidates()
    except APIClientError as error:
        _report(error, client.base_url)
        return

    deleted = list(result.get("deleted") or [])
    ui_state.forget_candidates(state, deleted)
    _clear_selection_widgets(state, deleted)

    st.session_state["_delete_notice"] = f"Cleared {plural(len(deleted), 'resume')} from the pool."
    st.rerun()


def _clear_selection_widgets(state: MutableMapping[str, Any], deleted: Sequence[str]) -> None:
    """Drop the checkbox keys of candidates that no longer exist.

    Streamlit keeps a widget's value under its key until something removes it.
    Leaving the key behind would carry a tick over to whatever candidate later
    reused that id.
    """
    for candidate_id in deleted:
        for prefix in ("select_", "delete_"):
            state.pop(f"{prefix}{candidate_id}", None)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

RENDERERS = {
    "Overview": render_overview,
    "Screening": render_screening,
    "Ranking": render_ranking,
    "Candidate": render_candidate,
    "Resumes": render_resumes,
}


def render_page(
    page: str,
    client: ScreeningAPIClient,
    state: MutableMapping[str, Any],
    pool: Mapping[str, Any],
) -> None:
    """Render whichever page is current.

    Args:
        page: The page name.
        client: The API client.
        state: Session state.
        pool: The ``GET /candidates`` response.
    """
    RENDERERS.get(page, render_overview)(client, state, pool)
