"""Design tokens and the stylesheet built from them.

Where this came from
--------------------
The ``ui-ux-pro-max`` skill returns **Minimalism & Swiss Style** for this
product type -- twice, for "recruitment hiring dashboard" and for "enterprise
hr analytics admin dashboard" -- with the note that its best fit is "enterprise
apps, dashboards, professional tools". Its colour strategy for that pattern is
*"neutral. Status colors (green/amber/red). Data-dense but scannable."*

That is the change from the first pass. The canvas was blue-tinted and blue was
doing too many jobs at once, which reads as a demo rather than a tool. Now:

* the canvas and every neutral is on a **slate ramp** -- the same ramp the skill
  returns for enterprise dashboards;
* **blue is reserved**: primary actions, the active nav item, links, focus
  rings, the top-match rule. Nothing decorative is blue;
* **status is green / amber / red**, and always paired with a word.

Also taken from the skill: its four-level elevation scale, of which only the
first two are used -- a professional tool wants edges and restraint, not depth.

Contrast is checked rather than assumed. The skill's own rule is 4.5:1 for body
text, so every `-text` token below is the darkened variant that clears it, while
the plain colour stays for fills, borders and chart marks, where the 3:1
non-text threshold applies.

Streamlit theming happens in two places. ``.streamlit/config.toml`` sets the
base theme for Streamlit's own widgets; this module supplies everything
Streamlit has no theme option for, built from the same tokens so the two cannot
drift.
"""

from __future__ import annotations

__all__ = [
    "TOKENS",
    "FONT_IMPORT_URL",
    "TONES",
    "stylesheet",
]

# --- Tokens ---------------------------------------------------------------
TOKENS: dict[str, str] = {
    # Brand. Used sparingly and only where it means "act" or "you are here".
    "--rs-primary": "#0369A1",
    "--rs-primary-hover": "#075985",
    "--rs-primary-soft": "#EFF6FF",
    "--rs-on-primary": "#FFFFFF",
    # Neutral ramp (slate). The canvas, not the accent.
    "--rs-canvas": "#F6F8FA",
    "--rs-surface": "#FFFFFF",
    "--rs-surface-sunken": "#F1F5F9",
    "--rs-surface-hover": "#F8FAFC",
    "--rs-ink": "#0F172A",
    "--rs-ink-secondary": "#334155",
    "--rs-ink-muted": "#64748B",
    "--rs-line": "#E2E8F0",
    "--rs-line-strong": "#CBD5E1",
    # Status. `-text` clears 4.5:1 on white; the plain value is for fills,
    # borders and chart marks only.
    "--rs-positive": "#16A34A",
    "--rs-positive-text": "#15803D",
    "--rs-positive-soft": "#F0FDF4",
    "--rs-caution": "#D97706",
    "--rs-caution-text": "#B45309",
    "--rs-caution-soft": "#FFFBEB",
    "--rs-critical": "#DC2626",
    "--rs-critical-text": "#B91C1C",
    "--rs-critical-soft": "#FEF2F2",
    "--rs-neutral-text": "#475569",
    "--rs-neutral-soft": "#F1F5F9",
    # Spacing, 4px rhythm
    "--rs-space-1": "4px",
    "--rs-space-2": "8px",
    "--rs-space-3": "12px",
    "--rs-space-4": "16px",
    "--rs-space-5": "20px",
    "--rs-space-6": "28px",
    "--rs-space-7": "40px",
    # Radii and elevation. Levels 1-2 of the skill's four-level scale; the
    # deeper ones belong to a spatial UI, which this is not.
    "--rs-radius-sm": "6px",
    "--rs-radius": "10px",
    "--rs-radius-lg": "14px",
    "--rs-elevation-1": "0 1px 2px rgba(15, 23, 42, 0.05)",
    "--rs-elevation-2": "0 4px 12px rgba(15, 23, 42, 0.07)",
    "--rs-focus-ring": "0 0 0 3px rgba(3, 105, 161, 0.30)",
    # Type
    "--rs-font-body": "'Inter', 'Fira Sans', 'Segoe UI', system-ui, sans-serif",
    "--rs-font-mono": "'Fira Code', 'Cascadia Mono', ui-monospace, monospace",
    "--rs-text-xs": "0.72rem",
    "--rs-text-sm": "0.82rem",
    "--rs-text-base": "0.94rem",
    "--rs-text-lg": "1.08rem",
    "--rs-text-xl": "1.35rem",
    "--rs-text-2xl": "1.75rem",
}

FONT_IMPORT_URL = (
    "https://fonts.googleapis.com/css2"
    "?family=Fira+Code:wght@400;500;600"
    "&family=Inter:wght@400;500;600;700&display=swap"
)

# --- Semantic tones -------------------------------------------------------
# Every tone pairs a colour with a text label at the call site. Nothing in this
# dashboard is distinguished by colour alone: the skill's `color-not-only` rule,
# applied structurally rather than remembered case by case.
TONES: dict[str, dict[str, str]] = {
    "positive": {
        "text": "var(--rs-positive-text)",
        "surface": "var(--rs-positive-soft)",
        "border": "var(--rs-positive)",
    },
    "caution": {
        "text": "var(--rs-caution-text)",
        "surface": "var(--rs-caution-soft)",
        "border": "var(--rs-caution)",
    },
    "critical": {
        "text": "var(--rs-critical-text)",
        "surface": "var(--rs-critical-soft)",
        "border": "var(--rs-critical)",
    },
    "info": {
        "text": "var(--rs-primary-hover)",
        "surface": "var(--rs-primary-soft)",
        "border": "var(--rs-primary)",
    },
    "neutral": {
        "text": "var(--rs-neutral-text)",
        "surface": "var(--rs-neutral-soft)",
        "border": "var(--rs-line-strong)",
    },
}

# Kept so existing callers and tests that speak the old tone names keep working.
_TONE_ALIASES = {"success": "positive", "warning": "caution", "danger": "critical"}
for _alias, _target in _TONE_ALIASES.items():
    TONES[_alias] = TONES[_target]


def _root_variables() -> str:
    """Render :data:`TOKENS` as a CSS ``:root`` block."""
    declarations = "\n".join(f"    {name}: {value};" for name, value in TOKENS.items())
    return f":root {{\n{declarations}\n}}"


def _tone_rules() -> str:
    """Render badge and chip rules, one pair per entry in :data:`TONES`."""
    return "\n".join(
        f".rs-badge--{name} {{ color: {tone['text']}; background: {tone['surface']}; "
        f"border-color: {tone['border']}; }}"
        for name, tone in TONES.items()
    )


def stylesheet() -> str:
    """Return the dashboard stylesheet, ready to inject.

    Returns:
        A ``<style>`` element: the font import, the token block, and every
        component rule built from those tokens. One stylesheet, injected once
        per render -- there are no per-page styles anywhere else.
    """
    return f"""<style>
@import url('{FONT_IMPORT_URL}');

{_root_variables()}

/* --- Base ------------------------------------------------------------- */
html, body, .stApp {{
    background: var(--rs-canvas);
    font-family: var(--rs-font-body);
    color: var(--rs-ink);
    font-size: var(--rs-text-base);
    -webkit-font-smoothing: antialiased;
}}
.stApp h1, .stApp h2, .stApp h3, .stApp h4 {{
    font-family: var(--rs-font-body);
    color: var(--rs-ink);
    font-weight: 600;
    letter-spacing: -0.015em;
}}
.stApp p, .stApp li, .stApp label {{ color: var(--rs-ink-secondary); }}

/* Figures never jitter between rows. */
.rs-num {{ font-family: var(--rs-font-mono); font-variant-numeric: tabular-nums; }}

/* Streamlit's own chrome is what makes an app look like a Streamlit app. */
[data-testid="stDecoration"] {{ display: none; }}
[data-testid="stToolbar"] {{ right: var(--rs-space-2); }}
[data-testid="stAppDeployButton"] {{ display: none; }}
.stAppHeader {{ background: transparent; }}
.stMainBlockContainer {{ padding-top: var(--rs-space-6); max-width: 1320px; }}

/* Keyboard focus is strengthened, never removed. */
.stApp *:focus-visible {{
    outline: 2px solid var(--rs-primary);
    outline-offset: 2px;
    box-shadow: var(--rs-focus-ring);
    border-radius: var(--rs-radius-sm);
}}
.stApp button, .stApp [role="button"], .stApp label {{ cursor: pointer; }}

@media (prefers-reduced-motion: reduce) {{
    *, *::before, *::after {{
        animation-duration: 0.001ms !important;
        transition-duration: 0.001ms !important;
    }}
}}

/* --- Sidebar ---------------------------------------------------------- */
[data-testid="stSidebar"] {{
    background: var(--rs-surface);
    border-right: 1px solid var(--rs-line);
}}
[data-testid="stSidebar"] .stRadio label {{
    font-size: var(--rs-text-sm);
    font-weight: 500;
}}
.rs-brand {{
    display: flex; align-items: center; gap: var(--rs-space-3);
    padding-bottom: var(--rs-space-4);
    margin-bottom: var(--rs-space-4);
    border-bottom: 1px solid var(--rs-line);
}}
.rs-brand__mark {{
    width: 34px; height: 34px; flex: 0 0 34px;
    border-radius: var(--rs-radius-sm);
    background: var(--rs-primary);
    color: var(--rs-on-primary);
    display: flex; align-items: center; justify-content: center;
    font-family: var(--rs-font-mono); font-weight: 600; font-size: var(--rs-text-sm);
}}
.rs-brand__name {{ font-weight: 650; font-size: var(--rs-text-base); line-height: 1.2; }}
.rs-brand__role {{ color: var(--rs-ink-muted); font-size: var(--rs-text-xs); }}
.rs-eyebrow {{
    font-size: var(--rs-text-xs); font-weight: 600; letter-spacing: 0.08em;
    text-transform: uppercase; color: var(--rs-ink-muted);
    margin: var(--rs-space-4) 0 var(--rs-space-2);
}}

/* --- Page header ------------------------------------------------------ */
.rs-masthead {{ margin-bottom: var(--rs-space-5); }}
.rs-masthead h1 {{ margin: 0; font-size: var(--rs-text-2xl); line-height: 1.25; }}
.rs-masthead p {{
    margin: var(--rs-space-2) 0 0;
    color: var(--rs-ink-muted);
    font-size: var(--rs-text-sm);
    max-width: 74ch;
}}

/* --- Sections --------------------------------------------------------- */
.rs-section {{ margin: var(--rs-space-6) 0 var(--rs-space-3); }}
.rs-section__title {{
    font-size: var(--rs-text-lg); font-weight: 600; color: var(--rs-ink); margin: 0;
}}
.rs-section__subtitle {{
    color: var(--rs-ink-muted); font-size: var(--rs-text-sm);
    margin: var(--rs-space-1) 0 0; max-width: 74ch;
}}
.rs-rule {{ height: 1px; background: var(--rs-line); border: 0; margin: var(--rs-space-5) 0; }}

/* --- Cards ------------------------------------------------------------ */
.rs-card {{
    background: var(--rs-surface);
    border: 1px solid var(--rs-line);
    border-radius: var(--rs-radius);
    padding: var(--rs-space-4);
    box-shadow: var(--rs-elevation-1);
}}
.rs-card__label {{
    font-size: var(--rs-text-xs); font-weight: 600; letter-spacing: 0.07em;
    text-transform: uppercase; color: var(--rs-ink-muted);
    margin: 0 0 var(--rs-space-2);
}}

/* --- Stat cards ------------------------------------------------------- */
.rs-stat {{
    background: var(--rs-surface);
    border: 1px solid var(--rs-line);
    border-radius: var(--rs-radius);
    padding: var(--rs-space-4);
    box-shadow: var(--rs-elevation-1);
    height: 100%;
}}
.rs-stat__label {{
    font-size: var(--rs-text-xs); font-weight: 600; letter-spacing: 0.07em;
    text-transform: uppercase; color: var(--rs-ink-muted); margin: 0;
}}
.rs-stat__value {{
    font-family: var(--rs-font-mono); font-variant-numeric: tabular-nums;
    font-size: var(--rs-text-2xl); font-weight: 600; color: var(--rs-ink);
    line-height: 1.15; margin: var(--rs-space-2) 0 0;
}}
.rs-stat__value--muted {{ font-size: var(--rs-text-lg); color: var(--rs-ink-muted); }}
.rs-stat__hint {{
    font-size: var(--rs-text-xs); color: var(--rs-ink-muted);
    margin: var(--rs-space-2) 0 0; line-height: 1.5;
}}

/* --- Badges ----------------------------------------------------------- */
.rs-badge {{
    display: inline-flex; align-items: center; gap: var(--rs-space-2);
    padding: 3px 10px;
    border-radius: 999px;
    border: 1px solid;
    font-size: var(--rs-text-xs); font-weight: 600; line-height: 1.6;
    white-space: nowrap;
}}
/* A shape as well as a colour, so the tone survives greyscale. */
.rs-badge::before {{
    content: ""; width: 6px; height: 6px; border-radius: 50%;
    background: currentColor; flex: 0 0 6px;
}}
.rs-badge--neutral::before, .rs-badge--info::before {{ border-radius: 1px; }}
{_tone_rules()}

/* --- Chips ------------------------------------------------------------ */
.rs-chips {{ display: flex; flex-wrap: wrap; gap: var(--rs-space-2); }}
.rs-chip {{
    display: inline-block; padding: 3px 10px;
    border-radius: var(--rs-radius-sm);
    font-size: var(--rs-text-sm); border: 1px solid;
    max-width: 100%; overflow-wrap: anywhere;
}}
.rs-chip--have {{
    color: var(--rs-positive-text); background: var(--rs-positive-soft);
    border-color: var(--rs-positive);
}}
.rs-chip--gap {{
    color: var(--rs-critical-text); background: var(--rs-critical-soft);
    border-color: var(--rs-critical); border-style: dashed;
}}

/* --- Evidence vs interpretation --------------------------------------- */
/* The single most important visual distinction on the page: what the resume
   said, versus what a model wrote about it. Different face, ground and rule. */
.rs-evidence {{
    background: var(--rs-surface-sunken);
    border: 1px solid var(--rs-line);
    border-left: 3px solid var(--rs-ink-secondary);
    border-radius: 0 var(--rs-radius-sm) var(--rs-radius-sm) 0;
    padding: var(--rs-space-3) var(--rs-space-4);
    margin-bottom: var(--rs-space-3);
    font-family: var(--rs-font-mono);
    font-size: var(--rs-text-sm); line-height: 1.7;
    white-space: pre-wrap; overflow-wrap: anywhere;
    color: var(--rs-ink-secondary);
}}
.rs-generated {{
    background: var(--rs-surface);
    border: 1px solid var(--rs-line);
    border-left: 3px dashed var(--rs-primary);
    border-radius: 0 var(--rs-radius-sm) var(--rs-radius-sm) 0;
    padding: var(--rs-space-4);
    margin-bottom: var(--rs-space-3);
    font-size: var(--rs-text-base); line-height: 1.7;
    box-shadow: var(--rs-elevation-1);
}}
.rs-blocklabel {{
    display: block; font-family: var(--rs-font-body);
    font-size: var(--rs-text-xs); font-weight: 600; letter-spacing: 0.07em;
    text-transform: uppercase; color: var(--rs-ink-muted);
    margin-bottom: var(--rs-space-2);
}}

/* --- Steps ------------------------------------------------------------ */
.rs-steps {{ display: flex; flex-wrap: wrap; gap: var(--rs-space-3); margin-bottom: var(--rs-space-5); }}
.rs-step {{
    flex: 1 1 200px;
    display: flex; align-items: flex-start; gap: var(--rs-space-3);
    padding: var(--rs-space-3) var(--rs-space-4);
    border: 1px solid var(--rs-line);
    border-radius: var(--rs-radius);
    background: var(--rs-surface);
}}
.rs-step__index {{
    width: 24px; height: 24px; flex: 0 0 24px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-family: var(--rs-font-mono); font-size: var(--rs-text-xs); font-weight: 600;
    background: var(--rs-surface-sunken); color: var(--rs-ink-muted);
    border: 1px solid var(--rs-line-strong);
}}
.rs-step__title {{ font-size: var(--rs-text-sm); font-weight: 600; color: var(--rs-ink-secondary); }}
.rs-step__state {{ font-size: var(--rs-text-xs); color: var(--rs-ink-muted); }}
.rs-step--done {{ border-color: var(--rs-positive); }}
.rs-step--done .rs-step__index {{
    background: var(--rs-positive-soft); color: var(--rs-positive-text);
    border-color: var(--rs-positive);
}}
.rs-step--done .rs-step__state {{ color: var(--rs-positive-text); }}
.rs-step--active {{ border-color: var(--rs-primary); box-shadow: var(--rs-elevation-2); }}
.rs-step--active .rs-step__index {{
    background: var(--rs-primary); color: var(--rs-on-primary); border-color: var(--rs-primary);
}}
.rs-step--active .rs-step__title {{ color: var(--rs-ink); }}
.rs-step--active .rs-step__state {{ color: var(--rs-primary-hover); font-weight: 600; }}

/* --- Lead candidate --------------------------------------------------- */
/* Identifies the top match with a rule and a label rather than a colour wash. */
.rs-lead {{
    display: flex; flex-wrap: wrap; align-items: center; gap: var(--rs-space-4);
    background: var(--rs-surface);
    border: 1px solid var(--rs-line);
    border-left: 3px solid var(--rs-primary);
    border-radius: var(--rs-radius);
    padding: var(--rs-space-4) var(--rs-space-5);
    box-shadow: var(--rs-elevation-1);
}}
.rs-lead__name {{ font-size: var(--rs-text-xl); font-weight: 650; color: var(--rs-ink); margin: 0; }}
.rs-lead__meta {{ color: var(--rs-ink-muted); font-size: var(--rs-text-sm); margin: 2px 0 0; }}
.rs-lead__spacer {{ flex: 1 1 auto; }}

/* --- Empty / states --------------------------------------------------- */
.rs-empty {{
    border: 1px dashed var(--rs-line-strong);
    border-radius: var(--rs-radius);
    background: var(--rs-surface);
    padding: var(--rs-space-7) var(--rs-space-5);
    text-align: center;
}}
.rs-empty h3 {{ margin: 0 0 var(--rs-space-2); font-size: var(--rs-text-lg); color: var(--rs-ink); }}
.rs-empty p {{ margin: 0 auto; color: var(--rs-ink-muted); font-size: var(--rs-text-sm); max-width: 56ch; }}

.rs-note {{
    color: var(--rs-ink-muted); font-size: var(--rs-text-sm);
    line-height: 1.65; margin-top: var(--rs-space-2);
}}
.rs-footnote {{
    border-top: 1px solid var(--rs-line);
    padding-top: var(--rs-space-3); margin-top: var(--rs-space-4);
    color: var(--rs-ink-muted); font-size: var(--rs-text-xs); line-height: 1.65;
    max-width: 90ch;
}}
.rs-wrap {{ overflow-wrap: anywhere; }}

/* --- Streamlit widgets ------------------------------------------------ */
.stButton > button, .stFormSubmitButton > button, .stDownloadButton > button {{
    border-radius: var(--rs-radius-sm);
    font-weight: 550;
    font-size: var(--rs-text-sm);
    padding: 0.45rem 1rem;
    min-height: 40px;               /* comfortable target, 8px+ gaps around */
    transition: background-color 180ms ease, border-color 180ms ease;
}}
.stButton > button[kind="secondary"] {{
    background: var(--rs-surface);
    border: 1px solid var(--rs-line-strong);
    color: var(--rs-ink-secondary);
}}
.stButton > button[kind="secondary"]:hover {{
    background: var(--rs-surface-hover);
    border-color: var(--rs-primary);
    color: var(--rs-primary-hover);
}}
.stTextArea textarea, .stTextInput input {{
    border-radius: var(--rs-radius-sm);
    border-color: var(--rs-line-strong);
    font-size: var(--rs-text-sm);
}}
.stTextArea textarea {{ font-family: var(--rs-font-body); line-height: 1.6; }}

[data-testid="stExpander"] {{
    border: 1px solid var(--rs-line);
    border-radius: var(--rs-radius);
    background: var(--rs-surface);
    box-shadow: var(--rs-elevation-1);
}}
[data-testid="stExpander"] summary {{ font-size: var(--rs-text-sm); font-weight: 550; }}

[data-testid="stDataFrame"] {{
    border: 1px solid var(--rs-line);
    border-radius: var(--rs-radius);
    overflow: hidden;
    box-shadow: var(--rs-elevation-1);
}}

[data-testid="stFileUploaderDropzone"] {{
    background: var(--rs-surface);
    border: 1px dashed var(--rs-line-strong);
    border-radius: var(--rs-radius);
}}
[data-testid="stFileUploaderDropzone"]:hover {{ border-color: var(--rs-primary); }}

.stAlert {{ border-radius: var(--rs-radius); font-size: var(--rs-text-sm); }}
.stProgress > div > div > div {{ background-color: var(--rs-primary); }}
hr {{ border-color: var(--rs-line); }}

/* --- Responsive ------------------------------------------------------- */
@media (max-width: 900px) {{
    .rs-steps {{ flex-direction: column; }}
    .rs-lead {{ flex-direction: column; align-items: flex-start; }}
    .rs-masthead h1 {{ font-size: var(--rs-text-xl); }}
}}
</style>"""
