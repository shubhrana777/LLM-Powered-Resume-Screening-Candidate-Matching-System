"""Reusable presentation pieces.

Each function renders one thing and returns nothing (or a small user choice).
They hold no state and make no requests: pages fetch, components display.

Every piece of markup the dashboard produces is built here, from the tokens in
:mod:`app.ui.theme`. Pages call these functions; they never write HTML inline,
so there is one place to change how a card, a badge or an evidence block looks.

Two pieces carry more weight than the rest:

:func:`evidence_block`
    Retrieved resume text. It must never look like the model's prose -- see
    :func:`generated_block`, its deliberate opposite.
:func:`badge`
    Always renders a word and a shape, not just a colour, so status survives
    greyscale, colour blindness and a screen reader.
"""

from __future__ import annotations

import html
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd
import streamlit as st

from app.ui.formatting import (
    NOT_ANALYZED,
    SIMILARITY_MEANING,
    format_similarity,
    format_similarity_raw,
    score_band,
    to_percent,
)
from app.ui.theme import TONES, stylesheet

__all__ = [
    "inject_theme",
    "brand",
    "sidebar_eyebrow",
    "masthead",
    "section",
    "rule",
    "badge",
    "badge_html",
    "chips",
    "card",
    "stat_cards",
    "lead_candidate",
    "evidence_block",
    "generated_block",
    "empty_state",
    "error_state",
    "api_unavailable_state",
    "step_indicator",
    "footnote",
    "similarity_footnote",
    "ranking_chart",
    "coverage_chart",
    "ranking_table",
]


def _esc(value: object) -> str:
    """Escape a value for interpolation into markup."""
    return html.escape(str(value))


def inject_theme() -> None:
    """Apply the dashboard stylesheet. Call once per render, before anything else."""
    st.markdown(stylesheet(), unsafe_allow_html=True)


# --- Shell ----------------------------------------------------------------


def brand(name: str, role: str, mark: str = "RS") -> None:
    """Render the product lockup at the top of the sidebar.

    Args:
        name: Product name.
        role: What this instance is, e.g. "Recruiter dashboard".
        mark: Two or three characters for the badge.
    """
    st.markdown(
        f"<div class='rs-brand'><div class='rs-brand__mark'>{_esc(mark)}</div>"
        f"<div><div class='rs-brand__name'>{_esc(name)}</div>"
        f"<div class='rs-brand__role'>{_esc(role)}</div></div></div>",
        unsafe_allow_html=True,
    )


def sidebar_eyebrow(text: str) -> None:
    """Render a small uppercase group label in the sidebar."""
    st.markdown(f"<p class='rs-eyebrow'>{_esc(text)}</p>", unsafe_allow_html=True)


def masthead(title: str, subtitle: str) -> None:
    """Render the page heading.

    Args:
        title: Page title, rendered as the page's only ``h1``.
        subtitle: One line explaining what the page is for.
    """
    st.markdown(
        f"<div class='rs-masthead'><h1>{_esc(title)}</h1>"
        f"<p>{_esc(subtitle)}</p></div>",
        unsafe_allow_html=True,
    )


def section(title: str, subtitle: str = "") -> None:
    """Render a section heading with an optional explanatory line.

    Args:
        title: The section title.
        subtitle: What the section is for, or how to read it.
    """
    extra = f"<p class='rs-section__subtitle'>{_esc(subtitle)}</p>" if subtitle else ""
    st.markdown(
        f"<div class='rs-section'><p class='rs-section__title'>{_esc(title)}</p>{extra}</div>",
        unsafe_allow_html=True,
    )


def rule() -> None:
    """Render a hairline section separator."""
    st.markdown("<hr class='rs-rule'/>", unsafe_allow_html=True)


def footnote(text: str) -> None:
    """Render an explanatory note under a section, set apart by a rule."""
    st.markdown(f"<p class='rs-footnote'>{_esc(text)}</p>", unsafe_allow_html=True)


def similarity_footnote() -> None:
    """Render the standing explanation of what a similarity figure means.

    The wording lives in one constant and appears wherever a percentage does,
    so the figure can never travel without it.
    """
    footnote(SIMILARITY_MEANING)


# --- Status ---------------------------------------------------------------


def badge_html(label: str, tone: str = "neutral") -> str:
    """Return the markup for a status badge.

    Args:
        label: The text. Always rendered -- the colour and the dot are second
            and third signals, never the only one.
        tone: A key of :data:`app.ui.theme.TONES`.

    Returns:
        A ``<span>`` element.
    """
    safe_tone = tone if tone in TONES else "neutral"
    return f"<span class='rs-badge rs-badge--{safe_tone}'>{_esc(label)}</span>"


def badge(label: str, tone: str = "neutral") -> None:
    """Render a status badge."""
    st.markdown(badge_html(label, tone), unsafe_allow_html=True)


def chips(items: Sequence[str], kind: str = "have", empty_message: str = "None") -> None:
    """Render a list of skills as chips.

    Args:
        items: The skill names.
        kind: ``"have"`` for matched skills, ``"gap"`` for missing ones. Gaps
            are dashed as well as differently coloured.
        empty_message: Shown when there is nothing to list.
    """
    if not items:
        st.markdown(f"<p class='rs-note'>{_esc(empty_message)}</p>", unsafe_allow_html=True)
        return

    rendered = "".join(
        f"<span class='rs-chip rs-chip--{kind}'>{_esc(item)}</span>" for item in items
    )
    st.markdown(f"<div class='rs-chips'>{rendered}</div>", unsafe_allow_html=True)


# --- Containers -----------------------------------------------------------


def card(label: str, body_html: str) -> None:
    """Render a labelled card.

    Args:
        label: Small uppercase label.
        body_html: Already-escaped markup for the body.
    """
    st.markdown(
        f"<div class='rs-card'><p class='rs-card__label'>{_esc(label)}</p>{body_html}</div>",
        unsafe_allow_html=True,
    )


def stat_cards(items: Sequence[tuple[str, str, str]]) -> None:
    """Render a row of headline figures.

    Used instead of ``st.metric`` so the label, value and hint share the
    dashboard's type scale and card treatment rather than Streamlit's.

    Args:
        items: ``(label, value, hint)`` triples. A value equal to
            :data:`~app.ui.formatting.NOT_ANALYZED` is de-emphasised, because a
            figure that does not exist yet should not read as loudly as one
            that does.
    """
    if not items:
        return

    columns = st.columns(len(items), gap="small")
    for column, (label, value, hint) in zip(columns, items):
        muted = " rs-stat__value--muted" if value == NOT_ANALYZED else ""
        with column:
            st.markdown(
                f"<div class='rs-stat'><p class='rs-stat__label'>{_esc(label)}</p>"
                f"<p class='rs-stat__value{muted}'>{_esc(value)}</p>"
                f"<p class='rs-stat__hint'>{_esc(hint)}</p></div>",
                unsafe_allow_html=True,
            )


def lead_candidate(name: str, score: float | None, extra: str = "") -> None:
    """Highlight the top-ranked candidate.

    Identified by position, a rule and the words "Top match" rather than by a
    colour wash, so it is obvious in greyscale and does not shout.

    Args:
        name: Candidate display name.
        score: Their similarity score.
        extra: Optional second line, e.g. skill coverage once analysed.
    """
    band = score_band(score)
    meta = f"Similarity {format_similarity(score)} · {band.label}"
    if extra:
        meta += f" · {extra}"

    st.markdown(
        f"<div class='rs-lead'><div>"
        f"<p class='rs-lead__name'>{_esc(name)}</p>"
        f"<p class='rs-lead__meta rs-num'>{_esc(meta)}</p></div>"
        f"<div class='rs-lead__spacer'></div>"
        f"{badge_html('Top match', 'info')}</div>",
        unsafe_allow_html=True,
    )


def evidence_block(item: Mapping[str, Any]) -> None:
    """Render one retrieved resume passage.

    Deliberately unlike generated text: monospace, sunken ground, solid rule.
    This is what the resume actually says.

    Args:
        item: An evidence payload with ``chunk_id``, ``text`` and
            ``retrieval_score``.
    """
    chunk_id = _esc(item.get("chunk_id", "unknown"))
    score = item.get("retrieval_score")
    meta = f"From resume · {chunk_id}"
    if isinstance(score, (int, float)):
        meta += f" · retrieval similarity {score * 100:.2f}%"

    st.markdown(
        f"<div class='rs-evidence'><span class='rs-blocklabel'>{meta}</span>"
        f"{_esc(item.get('text', ''))}</div>",
        unsafe_allow_html=True,
    )


def generated_block(text: str, label: str = "AI-generated interpretation") -> None:
    """Render model-written prose, marked as such.

    Args:
        text: The generated text.
        label: What to call it, so it is never mistaken for source material.
    """
    st.markdown(
        f"<div class='rs-generated'><span class='rs-blocklabel'>{_esc(label)}</span>"
        f"{_esc(text)}</div>",
        unsafe_allow_html=True,
    )


# --- States ---------------------------------------------------------------


def empty_state(title: str, message: str) -> None:
    """Render an empty state that says what to do next.

    Args:
        title: Short heading.
        message: What the recruiter should do to fill this screen.
    """
    st.markdown(
        f"<div class='rs-empty'><h3>{_esc(title)}</h3><p>{_esc(message)}</p></div>",
        unsafe_allow_html=True,
    )


def error_state(message: str, details: Iterable[str] = ()) -> None:
    """Render a recoverable error.

    Args:
        message: What went wrong, in the backend's own words.
        details: Optional per-field detail.
    """
    st.error(message, icon=":material/error:")
    listed = [item for item in details if item]
    if listed:
        with st.expander("Details"):
            for item in listed:
                st.markdown(f"- {item}")


def api_unavailable_state(base_url: str) -> None:
    """Render the state where the backend cannot be reached at all.

    The one failure a recruiter cannot act on without instructions, so it gets
    the command needed to fix it.

    Args:
        base_url: Where the dashboard looked.
    """
    st.error(f"Cannot reach the screening API at {base_url}.", icon=":material/cloud_off:")
    st.markdown(
        "This dashboard is a client: without the backend it has nothing to show. "
        "Start the API, then reload this page."
    )
    st.code(".\\start_app.ps1", language="powershell")
    st.caption(
        "Or run it directly: uvicorn app.api.main:app --reload. "
        "If the API runs elsewhere, set API_BASE_URL before starting Streamlit."
    )


def step_indicator(steps: Sequence[tuple[str, bool]], active_index: int) -> None:
    """Render the screening progress bar.

    Args:
        steps: ``(label, done)`` pairs.
        active_index: Which step is current.
    """
    rendered = []
    for index, (label, done) in enumerate(steps):
        if done:
            modifier, state, marker = "rs-step--done", "Done", "✓"
        elif index == active_index:
            modifier, state, marker = "rs-step--active", "Current step", str(index + 1)
        else:
            modifier, state, marker = "", "Not started", str(index + 1)

        rendered.append(
            f"<div class='rs-step {modifier}'>"
            f"<span class='rs-step__index'>{marker}</span>"
            f"<span><span class='rs-step__title'>{_esc(label)}</span><br>"
            f"<span class='rs-step__state'>{state}</span></span></div>"
        )

    st.markdown(f"<div class='rs-steps'>{''.join(rendered)}</div>", unsafe_allow_html=True)


# --- Charts ---------------------------------------------------------------
# Two, both answering a question a table hides. Altair ships with Streamlit, so
# neither costs a dependency. Both follow the skill's chart guidance: horizontal
# bars for ranked comparison, values labelled directly on the mark, and the
# sortable table beside them as the accessible equivalent.


def ranking_chart(rows: Sequence[Mapping[str, Any]]) -> None:
    """Plot similarity as a horizontal bar chart.

    Answers "how far ahead is the leader, and where does the field fall away?",
    which a column of numbers hides.

    Args:
        rows: Match results with ``candidate`` and ``similarity_score``.
    """
    if not rows:
        return

    import altair as alt

    frame = pd.DataFrame(
        [
            {
                "Candidate": str(row.get("candidate") or row.get("candidate_id", "")),
                "Similarity": to_percent(row.get("similarity_score")) or 0.0,
                "Band": score_band(row.get("similarity_score")).label,
            }
            for row in rows
        ]
    )

    base = alt.Chart(frame).encode(
        y=alt.Y("Candidate:N", sort="-x", title=None),
        x=alt.X(
            "Similarity:Q",
            title="Semantic similarity (ranking signal, not a hiring probability)",
            scale=alt.Scale(domain=[0, 100]),
            axis=alt.Axis(format=".0f", labelExpr="datum.value + '%'"),
        ),
    )

    bars = base.mark_bar(cornerRadiusEnd=2, size=18).encode(
        # Colour repeats the band the label already states; never the only
        # carrier of that information.
        color=alt.Color(
            "Band:N",
            title="Similarity band",
            scale=alt.Scale(
                domain=["Strong similarity", "Moderate similarity", "Low similarity"],
                range=["#16A34A", "#D97706", "#94A3B8"],
            ),
            legend=alt.Legend(orient="bottom"),
        ),
        tooltip=["Candidate", alt.Tooltip("Similarity:Q", format=".2f", title="Similarity %"), "Band"],
    )

    labels = base.mark_text(align="left", dx=6, fontSize=11, color="#334155").encode(
        text=alt.Text("Similarity:Q", format=".2f")
    )

    st.altair_chart((bars + labels).properties(height=max(150, 34 * len(frame))))


def coverage_chart(rows: Sequence[Mapping[str, Any]]) -> None:
    """Plot matched versus missing skills per analysed candidate.

    Answers "who covers the requirements?" -- a different question from "who
    reads as similar?", and the two often disagree.

    Args:
        rows: Analyses with ``candidate``, ``matched_skills`` and ``skill_gaps``.
    """
    if not rows:
        return

    import altair as alt

    records: list[dict[str, Any]] = []
    for row in rows:
        name = str(row.get("candidate") or row.get("candidate_id", ""))
        records.append(
            {"Candidate": name, "Skills": len(row.get("matched_skills") or []), "Status": "Matched"}
        )
        records.append(
            {"Candidate": name, "Skills": len(row.get("skill_gaps") or []), "Status": "Gap"}
        )

    frame = pd.DataFrame(records)

    chart = (
        alt.Chart(frame)
        .mark_bar(size=18)
        .encode(
            y=alt.Y("Candidate:N", title=None),
            x=alt.X("Skills:Q", title="Skills named in the job description", stack="zero"),
            color=alt.Color(
                "Status:N",
                title="Skill status",
                scale=alt.Scale(domain=["Matched", "Gap"], range=["#0369A1", "#CBD5E1"]),
                legend=alt.Legend(orient="bottom"),
            ),
            order=alt.Order("Status:N", sort="descending"),
            tooltip=["Candidate", "Status", "Skills"],
        )
        .properties(height=max(150, 34 * frame["Candidate"].nunique()))
    )

    st.altair_chart(chart)


def ranking_table(rows: Sequence[Mapping[str, Any]]) -> None:
    """Render the ranking as a sortable table.

    The accessible counterpart to :func:`ranking_chart`: every value the chart
    encodes is here as text, and the column headers sort.

    ``Similarity`` holds the cosine value scaled to 0-100 so the column sorts
    numerically while displaying a percentage. The scaling is display-only; the
    API response is untouched.

    Args:
        rows: Prepared display rows.
    """
    st.dataframe(
        pd.DataFrame(rows),
        width="stretch",
        hide_index=True,
        column_config={
            "Rank": st.column_config.NumberColumn("Rank", width="small"),
            "Similarity": st.column_config.NumberColumn(
                "Similarity",
                help=SIMILARITY_MEANING,
                format="%.2f%%",
            ),
            "Band": st.column_config.TextColumn(
                "Band", help="A word for the similarity figure, so it reads without colour."
            ),
            "Skill coverage": st.column_config.TextColumn(
                "Skill coverage",
                help="Skills found on the resume out of those named in the job description. "
                "A count, not a percentage, and separate from similarity.",
            ),
            "Experience": st.column_config.TextColumn(
                "Experience",
                help="Whether the stated years meet the stated requirement. "
                "'Not stated' means the resume gives no figure.",
            ),
            "Recommendation": st.column_config.TextColumn(
                "Recommendation",
                help="A coarse label from the analysis step. Not a score, not a hiring decision.",
            ),
        },
    )
