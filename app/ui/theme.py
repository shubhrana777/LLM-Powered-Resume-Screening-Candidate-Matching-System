"""Design tokens and the stylesheet built from them.

One system, one place
---------------------
Every colour, size, space and radius the dashboard uses is a token in
:data:`TOKENS`, and every component rule below is written in terms of those
tokens. Pages and components never carry styling of their own, so a change here
reaches all five screens at once.

The design is **Minimalism & Swiss Style**: a neutral slate canvas, one accent,
status carried by green / amber / red *and always a word*. Blue is reserved for
things that act or say "you are here" -- primary buttons, the active nav item,
links, focus rings, the top-match rule. Nothing decorative is blue.

Readability over density
------------------------
The type scale starts at 16px for body text and 15px for helper text. Nothing
in the interface is set below 13px, and 13px is used only for the letterspaced
uppercase micro-labels, where it is a deliberate typographic register rather
than a way to fit more in.

Contrast is checked, not assumed. Body text clears 4.5:1 on its own ground and
every `-text` token is the darkened variant that does so; the plain colour is
for fills, borders and chart marks, where the 3:1 non-text threshold applies.
Amber is the clearest case: ``--rs-caution`` (#F59E0B) is only ~2:1 on white and
is therefore **never** used for text -- ``--rs-caution-text`` is.

Button contrast
---------------
A primary button is white on blue, enforced here rather than left to Streamlit.
This is also where a real bug lived: the base rule that colours ``p`` elements
also caught the ``<p>`` Streamlit puts *inside* every button label, which is why
blue buttons could render with dark text. Button labels now inherit their
button's colour explicitly, and cannot be overridden by the base rule again.

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
    "NAV_ICONS",
    "stylesheet",
]

# --- Tokens ---------------------------------------------------------------
TOKENS: dict[str, str] = {
    # Brand. Used only where it means "act" or "you are here".
    "--rs-primary": "#2563EB",
    "--rs-primary-hover": "#1D4ED8",
    "--rs-primary-active": "#1E40AF",
    "--rs-primary-soft": "#EFF6FF",
    "--rs-primary-soft-border": "#BFDBFE",
    "--rs-on-primary": "#FFFFFF",
    # Neutral ramp (slate). The canvas, not the accent.
    "--rs-canvas": "#F8FAFC",
    "--rs-surface": "#FFFFFF",
    "--rs-surface-sunken": "#F1F5F9",
    "--rs-surface-hover": "#F1F5F9",
    "--rs-ink": "#0F172A",
    "--rs-ink-secondary": "#475569",
    "--rs-ink-muted": "#64748B",
    "--rs-line": "#E2E8F0",
    "--rs-line-strong": "#CBD5E1",
    # Status. `-text` clears 4.5:1; the plain value is for fills, borders and
    # chart marks only. Never set text in --rs-caution.
    "--rs-positive": "#16A34A",
    "--rs-positive-text": "#15803D",
    "--rs-positive-soft": "#F0FDF4",
    "--rs-caution": "#F59E0B",
    "--rs-caution-text": "#B45309",
    "--rs-caution-soft": "#FFFBEB",
    "--rs-critical": "#DC2626",
    "--rs-critical-text": "#B91C1C",
    "--rs-critical-soft": "#FEF2F2",
    "--rs-info": "#0284C7",
    "--rs-info-text": "#075985",
    "--rs-info-soft": "#F0F9FF",
    "--rs-neutral-text": "#475569",
    "--rs-neutral-soft": "#F1F5F9",
    # Spacing, 4px rhythm.
    "--rs-space-1": "4px",
    "--rs-space-2": "8px",
    "--rs-space-3": "12px",
    "--rs-space-4": "16px",
    "--rs-space-5": "20px",
    "--rs-space-6": "24px",
    "--rs-space-7": "32px",
    "--rs-space-8": "40px",
    # Radii and elevation. Two levels only: a professional tool wants edges and
    # restraint, not depth.
    "--rs-radius-sm": "8px",
    "--rs-radius": "12px",
    "--rs-radius-lg": "16px",
    "--rs-elevation-1": "0 1px 2px rgba(15, 23, 42, 0.04)",
    "--rs-elevation-2": "0 2px 8px rgba(15, 23, 42, 0.06)",
    "--rs-focus-ring": "0 0 0 3px rgba(37, 99, 235, 0.28)",
    # Type. Nothing below --rs-text-xs, and that is for uppercase labels only.
    "--rs-font-body": (
        "'Inter', 'Segoe UI', Roboto, system-ui, -apple-system, sans-serif"
    ),
    "--rs-font-mono": "'Fira Code', 'Cascadia Mono', ui-monospace, monospace",
    "--rs-text-xs": "0.8125rem",   # 13px -- uppercase micro-labels only
    "--rs-text-sm": "0.9375rem",   # 15px -- helper and secondary text
    "--rs-text-base": "1rem",      # 16px -- body
    "--rs-text-md": "1.0625rem",   # 17px -- prominent body
    "--rs-text-lg": "1.1875rem",   # 19px -- card headings
    "--rs-text-xl": "1.375rem",    # 22px -- section headings
    "--rs-text-2xl": "1.75rem",    # 28px -- page titles
    "--rs-text-3xl": "2rem",       # 32px -- headline figures
    "--rs-leading-tight": "1.25",
    "--rs-leading-heading": "1.35",
    "--rs-leading-body": "1.6",
    "--rs-leading-helper": "1.5",
}

FONT_IMPORT_URL = (
    "https://fonts.googleapis.com/css2"
    "?family=Fira+Code:wght@400;500;600"
    "&family=Inter:wght@400;500;600;700&display=swap"
)

# --- Semantic tones -------------------------------------------------------
# Every tone pairs a colour with a text label at the call site. Nothing in this
# dashboard is distinguished by colour alone.
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


# --- Navigation icons -----------------------------------------------------
# Drawn as CSS masks from inline SVG, so they need no icon font and no network,
# and they take their colour from the nav item's own `color`. Purely
# decorative: the readable page name is always the label beside them, and the
# radio options themselves are untouched, so navigation behaviour is unchanged.
#
# Order matches app.ui.state.PAGES. An item with no icon simply has none.
NAV_ICONS: tuple[str, ...] = (
    # Overview -- layout grid
    "%3Crect x='3' y='3' width='7' height='7' rx='1.5'/%3E"
    "%3Crect x='14' y='3' width='7' height='7' rx='1.5'/%3E"
    "%3Crect x='3' y='14' width='7' height='7' rx='1.5'/%3E"
    "%3Crect x='14' y='14' width='7' height='7' rx='1.5'/%3E",
    # Screening -- magnifier
    "%3Ccircle cx='11' cy='11' r='7'/%3E%3Cpath d='M21 21l-4.3-4.3'/%3E",
    # Ranking -- bar chart
    "%3Cpath d='M4 20V10M10 20V4M16 20v-7M22 20H2'/%3E",
    # Candidate -- person
    "%3Ccircle cx='12' cy='8' r='4'/%3E%3Cpath d='M4 21a8 8 0 0 1 16 0'/%3E",
    # Resumes -- document
    "%3Cpath d='M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z'/%3E"
    "%3Cpath d='M14 3v5h5M9 13h6M9 17h6'/%3E",
)

_SVG_OPEN = (
    "data:image/svg+xml;utf8,%3Csvg xmlns='http://www.w3.org/2000/svg' "
    "viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='1.8' "
    "stroke-linecap='round' stroke-linejoin='round'%3E"
)


def _root_variables() -> str:
    """Render :data:`TOKENS` as a CSS ``:root`` block."""
    declarations = "\n".join(f"    {name}: {value};" for name, value in TOKENS.items())
    return f":root {{\n{declarations}\n}}"


def _tone_rules() -> str:
    """Render badge rules, one per entry in :data:`TONES`."""
    return "\n".join(
        f".rs-badge--{name} {{ color: {tone['text']}; background: {tone['surface']}; "
        f"border-color: {tone['border']}; }}"
        for name, tone in TONES.items()
    )


def _nav_icon_rules() -> str:
    """Render one mask rule per navigation item, in :data:`NAV_ICONS` order."""
    rules = []
    for index, path in enumerate(NAV_ICONS, start=1):
        mask = f"url(\"{_SVG_OPEN}{path}%3C/svg%3E\")"
        rules.append(
            f'[data-testid="stSidebar"] [role="radiogroup"] > label:nth-of-type({index})'
            f"::before {{ -webkit-mask-image: {mask}; mask-image: {mask}; }}"
        )
    return "\n".join(rules)


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
    line-height: var(--rs-leading-body);
    -webkit-font-smoothing: antialiased;
}}

/* Heading hierarchy, one scale, applied everywhere. */
.stApp h1, .stApp h2, .stApp h3, .stApp h4 {{
    font-family: var(--rs-font-body);
    color: var(--rs-ink);
    line-height: var(--rs-leading-heading);
    letter-spacing: -0.015em;
}}
.stApp h1 {{ font-size: var(--rs-text-2xl); font-weight: 700; }}
.stApp h2 {{ font-size: var(--rs-text-xl); font-weight: 650; }}
.stApp h3 {{ font-size: var(--rs-text-lg); font-weight: 600; }}
.stApp h4 {{ font-size: var(--rs-text-md); font-weight: 600; }}

.stApp p, .stApp li {{
    color: var(--rs-ink-secondary);
    font-size: var(--rs-text-base);
    line-height: var(--rs-leading-body);
}}

/* Form labels: their own register, clearly above helper text. */
.stApp label, .stApp [data-testid="stWidgetLabel"] p {{
    color: var(--rs-ink);
    font-size: var(--rs-text-sm);
    font-weight: 600;
    line-height: var(--rs-leading-helper);
}}

/* st.caption is the helper-text register. Readable, never tiny. */
.stApp [data-testid="stCaptionContainer"],
.stApp [data-testid="stCaptionContainer"] p {{
    color: var(--rs-ink-muted);
    font-size: var(--rs-text-sm);
    line-height: var(--rs-leading-helper);
}}

/* Figures never jitter between rows. */
.rs-num {{ font-family: var(--rs-font-mono); font-variant-numeric: tabular-nums; }}

a, .stApp a {{ color: var(--rs-primary-hover); text-decoration-thickness: 1px; }}
a:hover, .stApp a:hover {{ color: var(--rs-primary-active); }}

/* Streamlit's own chrome is what makes an app look like a Streamlit app. */
[data-testid="stDecoration"] {{ display: none; }}
[data-testid="stToolbar"] {{ right: var(--rs-space-2); }}
[data-testid="stAppDeployButton"] {{ display: none; }}
.stAppHeader {{ background: transparent; }}
.stMainBlockContainer {{
    padding-top: var(--rs-space-7);
    padding-bottom: var(--rs-space-8);
    max-width: 1320px;
}}

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
[data-testid="stSidebar"] > div {{ padding-top: var(--rs-space-5); }}

/* Navigation. The radio still *is* the navigation -- only its appearance
   changes. The input keeps its place in the DOM so it stays focusable and
   announced; the BaseWeb dot is what is hidden. */
[data-testid="stSidebar"] [role="radiogroup"] {{
    gap: var(--rs-space-1);
    display: flex;
    flex-direction: column;
}}
[data-testid="stSidebar"] [role="radiogroup"] > label {{
    display: flex;
    align-items: center;
    gap: var(--rs-space-3);
    width: 100%;
    padding: 9px var(--rs-space-3);
    border-radius: var(--rs-radius-sm);
    border-left: 3px solid transparent;
    color: var(--rs-ink-secondary);
    transition: background-color 140ms ease, color 140ms ease;
}}
[data-testid="stSidebar"] [role="radiogroup"] > label > div:last-child p,
[data-testid="stSidebar"] [role="radiogroup"] > label p {{
    font-size: var(--rs-text-base);
    font-weight: 500;
    color: inherit;
    margin: 0;
    line-height: var(--rs-leading-tight);
}}
/* Decorative leading icon, coloured by the item's own text colour. */
[data-testid="stSidebar"] [role="radiogroup"] > label::before {{
    content: "";
    width: 19px; height: 19px; flex: 0 0 19px;
    background-color: currentColor;
    -webkit-mask-repeat: no-repeat; mask-repeat: no-repeat;
    -webkit-mask-position: center; mask-position: center;
    -webkit-mask-size: contain; mask-size: contain;
    opacity: 0.85;
}}
{_nav_icon_rules()}
/* Hide BaseWeb's radio dot. If the selector ever stops matching, the dot
   simply returns -- navigation keeps working either way. */
[data-testid="stSidebar"] [role="radiogroup"] div[data-baseweb="radio"] > div:first-of-type {{
    display: none;
}}
[data-testid="stSidebar"] [role="radiogroup"] > label:hover {{
    background: var(--rs-surface-hover);
    color: var(--rs-ink);
}}
/* Active page: soft blue ground, blue text, heavier weight, blue edge. */
[data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked) {{
    background: var(--rs-primary-soft);
    border-left-color: var(--rs-primary);
    color: var(--rs-primary-hover);
}}
[data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked) p {{
    font-weight: 650;
}}
[data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked)::before {{
    opacity: 1;
}}
[data-testid="stSidebar"] [role="radiogroup"] > label:has(input:focus-visible) {{
    outline: 2px solid var(--rs-primary);
    outline-offset: 1px;
}}

[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {{
    font-size: var(--rs-text-sm);
    line-height: var(--rs-leading-helper);
    color: var(--rs-ink-secondary);
}}
[data-testid="stSidebar"] .stAlert {{ font-size: var(--rs-text-sm); }}
[data-testid="stSidebar"] .stButton > button {{ width: 100%; }}

.rs-brand {{
    display: flex; align-items: center; gap: var(--rs-space-3);
    padding-bottom: var(--rs-space-4);
    margin-bottom: var(--rs-space-2);
    border-bottom: 1px solid var(--rs-line);
}}
.rs-brand__mark {{
    width: 40px; height: 40px; flex: 0 0 40px;
    border-radius: var(--rs-radius-sm);
    background: var(--rs-primary);
    color: var(--rs-on-primary);
    display: flex; align-items: center; justify-content: center;
    font-family: var(--rs-font-body); font-weight: 700; font-size: var(--rs-text-base);
    letter-spacing: 0.02em;
}}
.rs-brand__name {{
    font-weight: 700; font-size: var(--rs-text-md); color: var(--rs-ink);
    line-height: var(--rs-leading-tight);
}}
.rs-brand__role {{
    color: var(--rs-ink-muted); font-size: var(--rs-text-sm);
    line-height: var(--rs-leading-tight); margin-top: 2px;
}}
.rs-eyebrow {{
    font-size: var(--rs-text-xs); font-weight: 700; letter-spacing: 0.09em;
    text-transform: uppercase; color: var(--rs-ink-muted);
    margin: var(--rs-space-6) 0 var(--rs-space-2);
}}

/* --- Page header ------------------------------------------------------ */
.rs-masthead {{ margin-bottom: var(--rs-space-6); }}
.rs-masthead h1 {{
    margin: 0; font-size: var(--rs-text-2xl); font-weight: 700;
    line-height: var(--rs-leading-tight); color: var(--rs-ink);
}}
.rs-masthead p {{
    margin: var(--rs-space-3) 0 0;
    color: var(--rs-ink-secondary);
    font-size: var(--rs-text-md);
    line-height: var(--rs-leading-body);
    max-width: 78ch;
}}

/* --- Sections --------------------------------------------------------- */
.rs-section {{ margin: var(--rs-space-7) 0 var(--rs-space-4); }}
.rs-section__title {{
    font-size: var(--rs-text-xl); font-weight: 650; color: var(--rs-ink);
    margin: 0; line-height: var(--rs-leading-heading); letter-spacing: -0.015em;
}}
.rs-section__subtitle {{
    color: var(--rs-ink-secondary); font-size: var(--rs-text-base);
    line-height: var(--rs-leading-body);
    margin: var(--rs-space-2) 0 0; max-width: 78ch;
}}
.rs-rule {{
    height: 1px; background: var(--rs-line); border: 0;
    margin: var(--rs-space-7) 0;
}}

/* --- Cards ------------------------------------------------------------ */
.rs-card {{
    background: var(--rs-surface);
    border: 1px solid var(--rs-line);
    border-radius: var(--rs-radius);
    padding: var(--rs-space-5);
    box-shadow: var(--rs-elevation-1);
}}
.rs-card__label {{
    font-size: var(--rs-text-xs); font-weight: 700; letter-spacing: 0.08em;
    text-transform: uppercase; color: var(--rs-ink-muted);
    margin: 0 0 var(--rs-space-3);
}}

/* An informational panel: quieter than an alert, louder than a caption. */
.rs-panel {{
    display: flex; gap: var(--rs-space-3);
    background: var(--rs-primary-soft);
    border: 1px solid var(--rs-primary-soft-border);
    border-radius: var(--rs-radius);
    padding: var(--rs-space-4) var(--rs-space-5);
}}
.rs-panel__icon {{
    flex: 0 0 20px; width: 20px; height: 20px; margin-top: 1px;
    border-radius: 50%;
    background: var(--rs-primary); color: var(--rs-on-primary);
    display: flex; align-items: center; justify-content: center;
    font-size: var(--rs-text-xs); font-weight: 700; font-style: normal;
}}
.rs-panel__title {{
    margin: 0; font-size: var(--rs-text-base); font-weight: 650;
    color: var(--rs-primary-hover); line-height: var(--rs-leading-helper);
}}
.rs-panel__body {{
    margin: var(--rs-space-1) 0 0; font-size: var(--rs-text-sm);
    color: var(--rs-ink-secondary); line-height: var(--rs-leading-helper);
}}

/* --- Stat cards ------------------------------------------------------- */
.rs-stat {{
    background: var(--rs-surface);
    border: 1px solid var(--rs-line);
    border-radius: var(--rs-radius);
    padding: var(--rs-space-5);
    box-shadow: var(--rs-elevation-1);
    height: 100%;
}}
.rs-stat__label {{
    font-size: var(--rs-text-xs); font-weight: 700; letter-spacing: 0.08em;
    text-transform: uppercase; color: var(--rs-ink-muted); margin: 0;
}}
.rs-stat__value {{
    font-family: var(--rs-font-body); font-variant-numeric: tabular-nums;
    font-size: var(--rs-text-3xl); font-weight: 700; color: var(--rs-ink);
    line-height: var(--rs-leading-tight); margin: var(--rs-space-3) 0 0;
    letter-spacing: -0.02em;
}}
/* A figure that does not exist yet must not read as loudly as one that does. */
.rs-stat__value--muted {{
    font-size: var(--rs-text-lg); font-weight: 600; color: var(--rs-ink-muted);
    letter-spacing: 0;
}}
.rs-stat__hint {{
    font-size: var(--rs-text-sm); color: var(--rs-ink-muted);
    margin: var(--rs-space-3) 0 0; line-height: var(--rs-leading-helper);
}}

/* --- Badges ----------------------------------------------------------- */
.rs-badge {{
    display: inline-flex; align-items: center; gap: var(--rs-space-2);
    padding: 4px 12px;
    border-radius: 999px;
    border: 1px solid;
    font-size: var(--rs-text-sm); font-weight: 600; line-height: 1.5;
    white-space: nowrap;
}}
/* A shape as well as a colour, so the tone survives greyscale. */
.rs-badge::before {{
    content: ""; width: 7px; height: 7px; border-radius: 50%;
    background: currentColor; flex: 0 0 7px;
}}
.rs-badge--neutral::before, .rs-badge--info::before {{ border-radius: 1px; }}
{_tone_rules()}

/* --- Chips ------------------------------------------------------------ */
.rs-chips {{ display: flex; flex-wrap: wrap; gap: var(--rs-space-2); }}
.rs-chip {{
    display: inline-block; padding: 5px 12px;
    border-radius: var(--rs-radius-sm);
    font-size: var(--rs-text-sm); font-weight: 500; border: 1px solid;
    max-width: 100%; overflow-wrap: anywhere; line-height: 1.5;
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
    padding: var(--rs-space-4) var(--rs-space-5);
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
    padding: var(--rs-space-5);
    margin-bottom: var(--rs-space-3);
    font-size: var(--rs-text-base); line-height: var(--rs-leading-body);
    color: var(--rs-ink-secondary);
    box-shadow: var(--rs-elevation-1);
}}
.rs-blocklabel {{
    display: block; font-family: var(--rs-font-body);
    font-size: var(--rs-text-xs); font-weight: 700; letter-spacing: 0.08em;
    text-transform: uppercase; color: var(--rs-ink-muted);
    margin-bottom: var(--rs-space-2);
}}

/* --- Steps ------------------------------------------------------------ */
.rs-steps {{
    display: flex; flex-wrap: wrap; gap: var(--rs-space-3);
    margin-bottom: var(--rs-space-6);
}}
.rs-step {{
    flex: 1 1 220px;
    display: flex; align-items: center; gap: var(--rs-space-3);
    padding: var(--rs-space-4) var(--rs-space-5);
    border: 1px solid var(--rs-line);
    border-radius: var(--rs-radius);
    background: var(--rs-surface);
    box-shadow: var(--rs-elevation-1);
}}
.rs-step__index {{
    width: 30px; height: 30px; flex: 0 0 30px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-family: var(--rs-font-body); font-size: var(--rs-text-sm); font-weight: 700;
    background: var(--rs-surface-sunken); color: var(--rs-ink-muted);
    border: 1px solid var(--rs-line-strong);
}}
.rs-step__title {{
    font-size: var(--rs-text-base); font-weight: 600; color: var(--rs-ink-secondary);
    line-height: var(--rs-leading-tight);
}}
.rs-step__state {{
    font-size: var(--rs-text-sm); color: var(--rs-ink-muted);
    line-height: var(--rs-leading-tight); margin-top: 2px;
}}
.rs-step--done {{ border-color: var(--rs-positive); }}
.rs-step--done .rs-step__index {{
    background: var(--rs-positive-soft); color: var(--rs-positive-text);
    border-color: var(--rs-positive);
}}
.rs-step--done .rs-step__state {{ color: var(--rs-positive-text); }}
.rs-step--active {{
    border-color: var(--rs-primary);
    background: var(--rs-primary-soft);
    box-shadow: var(--rs-elevation-2);
}}
.rs-step--active .rs-step__index {{
    background: var(--rs-primary); color: var(--rs-on-primary); border-color: var(--rs-primary);
}}
.rs-step--active .rs-step__title {{ color: var(--rs-ink); font-weight: 700; }}
.rs-step--active .rs-step__state {{ color: var(--rs-primary-hover); font-weight: 600; }}

/* --- Lead candidate --------------------------------------------------- */
/* Identifies the top match with a rule and a label rather than a colour wash. */
.rs-lead {{
    display: flex; flex-wrap: wrap; align-items: center; gap: var(--rs-space-4);
    background: var(--rs-surface);
    border: 1px solid var(--rs-line);
    border-left: 3px solid var(--rs-primary);
    border-radius: var(--rs-radius);
    padding: var(--rs-space-5) var(--rs-space-6);
    box-shadow: var(--rs-elevation-1);
}}
.rs-lead__name {{
    font-size: var(--rs-text-xl); font-weight: 700; color: var(--rs-ink);
    margin: 0; line-height: var(--rs-leading-tight);
}}
.rs-lead__meta {{
    color: var(--rs-ink-muted); font-size: var(--rs-text-sm);
    margin: var(--rs-space-2) 0 0;
}}
.rs-lead__spacer {{ flex: 1 1 auto; }}

/* --- Empty / states --------------------------------------------------- */
.rs-empty {{
    border: 1px dashed var(--rs-line-strong);
    border-radius: var(--rs-radius);
    background: var(--rs-surface);
    padding: var(--rs-space-8) var(--rs-space-6);
    text-align: center;
}}
.rs-empty__icon {{
    width: 44px; height: 44px; margin: 0 auto var(--rs-space-4);
    border-radius: var(--rs-radius-sm);
    background: var(--rs-primary-soft); color: var(--rs-primary);
    display: flex; align-items: center; justify-content: center;
}}
.rs-empty__icon svg {{ width: 22px; height: 22px; display: block; }}
.rs-empty h3 {{
    margin: 0 0 var(--rs-space-2); font-size: var(--rs-text-lg);
    font-weight: 650; color: var(--rs-ink);
}}
.rs-empty p {{
    margin: 0 auto; color: var(--rs-ink-secondary); font-size: var(--rs-text-base);
    line-height: var(--rs-leading-body); max-width: 56ch;
}}

.rs-note {{
    color: var(--rs-ink-muted); font-size: var(--rs-text-sm);
    line-height: var(--rs-leading-helper); margin-top: var(--rs-space-2);
}}
.rs-footnote {{
    border-top: 1px solid var(--rs-line);
    padding-top: var(--rs-space-4); margin-top: var(--rs-space-5);
    color: var(--rs-ink-muted); font-size: var(--rs-text-sm);
    line-height: var(--rs-leading-helper);
    max-width: 90ch;
}}
.rs-wrap {{ overflow-wrap: anywhere; }}

/* --- Buttons ---------------------------------------------------------- */
/* Every button looks like a button: real padding, a real target, one radius. */
.stButton > button, .stFormSubmitButton > button, .stDownloadButton > button {{
    border-radius: var(--rs-radius-sm);
    font-family: var(--rs-font-body);
    font-weight: 600;
    font-size: var(--rs-text-sm);
    padding: 10px 18px;
    min-height: 44px;
    line-height: var(--rs-leading-tight);
    box-shadow: none;
    transition: background-color 160ms ease, border-color 160ms ease, color 160ms ease;
}}
/* Streamlit puts the label in a <p>. Without this it inherits the base
   paragraph colour instead of the button's -- which is what produced dark
   text on blue buttons. It must inherit, and nothing may override it. */
.stButton > button p,
.stFormSubmitButton > button p,
.stDownloadButton > button p,
.stButton > button div,
.stFormSubmitButton > button div,
.stDownloadButton > button div {{
    color: inherit !important;
    font-size: var(--rs-text-sm);
    font-weight: 600;
    line-height: var(--rs-leading-tight);
    margin: 0;
}}

/* Primary: white on blue, always. */
.stButton > button[kind^="primary"],
.stFormSubmitButton > button[kind^="primary"],
.stDownloadButton > button[kind^="primary"] {{
    background: var(--rs-primary);
    border: 1px solid var(--rs-primary);
    color: var(--rs-on-primary) !important;
}}
.stButton > button[kind^="primary"]:hover,
.stFormSubmitButton > button[kind^="primary"]:hover,
.stDownloadButton > button[kind^="primary"]:hover {{
    background: var(--rs-primary-hover);
    border-color: var(--rs-primary-hover);
    color: var(--rs-on-primary) !important;
}}
.stButton > button[kind^="primary"]:active,
.stFormSubmitButton > button[kind^="primary"]:active {{
    background: var(--rs-primary-active);
    border-color: var(--rs-primary-active);
}}

/* Secondary: an outlined control, still obviously a control. */
.stButton > button[kind^="secondary"],
.stFormSubmitButton > button[kind^="secondary"],
.stDownloadButton > button[kind^="secondary"] {{
    background: var(--rs-surface);
    border: 1px solid var(--rs-line-strong);
    color: var(--rs-ink-secondary) !important;
}}
.stButton > button[kind^="secondary"]:hover,
.stFormSubmitButton > button[kind^="secondary"]:hover,
.stDownloadButton > button[kind^="secondary"]:hover {{
    background: var(--rs-primary-soft);
    border-color: var(--rs-primary);
    color: var(--rs-primary-hover) !important;
}}

/* Disabled reads as disabled, in either variant. */
.stButton > button:disabled,
.stFormSubmitButton > button:disabled,
.stDownloadButton > button:disabled,
.stButton > button[disabled] {{
    background: var(--rs-surface-sunken) !important;
    border-color: var(--rs-line) !important;
    color: var(--rs-ink-muted) !important;
    cursor: not-allowed;
    opacity: 1;
}}

/* --- Inputs ----------------------------------------------------------- */
.stTextArea textarea, .stTextInput input, .stNumberInput input {{
    border-radius: var(--rs-radius-sm);
    border: 1px solid var(--rs-line-strong);
    background: var(--rs-surface);
    color: var(--rs-ink);
    font-size: var(--rs-text-base);
    padding: var(--rs-space-3) var(--rs-space-4);
    line-height: var(--rs-leading-body);
}}
.stTextArea textarea {{ font-family: var(--rs-font-body); min-height: 200px; }}
.stTextArea textarea:focus, .stTextInput input:focus {{
    border-color: var(--rs-primary);
    box-shadow: var(--rs-focus-ring);
}}
.stTextArea textarea::placeholder, .stTextInput input::placeholder {{
    color: var(--rs-ink-muted);
}}
[data-testid="stWidgetLabel"] {{ margin-bottom: var(--rs-space-2); }}

/* Select, multiselect and slider share the input register. */
.stSelectbox [data-baseweb="select"] > div,
.stMultiSelect [data-baseweb="select"] > div {{
    border-radius: var(--rs-radius-sm);
    border-color: var(--rs-line-strong);
    font-size: var(--rs-text-base);
    min-height: 44px;
}}
.stSlider [data-testid="stTickBarMin"], .stSlider [data-testid="stTickBarMax"] {{
    font-size: var(--rs-text-xs); color: var(--rs-ink-muted);
}}
.stCheckbox label p {{ font-size: var(--rs-text-base); font-weight: 500; }}

[data-testid="stExpander"] {{
    border: 1px solid var(--rs-line);
    border-radius: var(--rs-radius);
    background: var(--rs-surface);
    box-shadow: var(--rs-elevation-1);
}}
[data-testid="stExpander"] summary {{
    font-size: var(--rs-text-base); font-weight: 600; padding: var(--rs-space-3) var(--rs-space-4);
}}

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
    padding: var(--rs-space-5);
}}
[data-testid="stFileUploaderDropzone"]:hover {{
    border-color: var(--rs-primary);
    background: var(--rs-primary-soft);
}}
[data-testid="stFileUploaderDropzone"] button {{ min-height: 40px; }}

.stAlert {{
    border-radius: var(--rs-radius);
    font-size: var(--rs-text-base);
    padding: var(--rs-space-4) var(--rs-space-5);
}}
.stAlert p {{ font-size: var(--rs-text-base); line-height: var(--rs-leading-helper); }}
.stProgress > div > div > div {{ background-color: var(--rs-primary); }}
hr {{ border-color: var(--rs-line); }}

/* --- Responsive ------------------------------------------------------- */
/* Content reflows rather than scrolling sideways. */
.stMainBlockContainer, .rs-card, .rs-stat, .rs-panel {{ max-width: 100%; }}
@media (max-width: 1100px) {{
    .rs-step {{ flex: 1 1 100%; }}
}}
@media (max-width: 900px) {{
    .rs-steps {{ flex-direction: column; }}
    .rs-lead {{ flex-direction: column; align-items: flex-start; }}
    .rs-masthead h1 {{ font-size: var(--rs-text-xl); }}
    .rs-stat__value {{ font-size: var(--rs-text-2xl); }}
    .stMainBlockContainer {{ padding-top: var(--rs-space-5); }}
}}
</style>"""
