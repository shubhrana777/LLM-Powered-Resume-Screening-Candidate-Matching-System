"""Prompt templates for grounded candidate analysis.

Kept separate from business logic so the wording can be reviewed, diffed and
tested on its own. Nothing in this module calls an LLM or touches a resume.

Grounding is enforced in two independent places, because a prompt alone is a
request, not a guarantee:

1. **Here**, by instructing the model to use only the supplied material and to
   write "Not stated" rather than guess.
2. **In** :mod:`app.analysis_parser`, by checking the returned claims against
   the candidate profile and the supplied evidence, and correcting them when
   they do not hold.

An LLM can still ignore instruction (1). Layer (2) is what makes the safeguard
more than a polite request.
"""

from __future__ import annotations

from app.models import NOT_STATED, Recommendation

__all__ = [
    "SYSTEM_PROMPT",
    "ANALYSIS_TEMPLATE",
    "RESPONSE_SCHEMA_DESCRIPTION",
    "GROUNDING_RULES",
    "build_analysis_prompt",
]

# Individually addressable so tests can assert each rule survives editing.
GROUNDING_RULES: tuple[str, ...] = (
    "Use ONLY the job description, candidate profile, and retrieved resume "
    "evidence supplied in this message.",
    "Do not invent candidate information.",
    f'If a fact is not present in the supplied material, write exactly "{NOT_STATED}".',
    "Do not infer skills from unrelated terminology, from job titles, or from "
    "the employer's industry.",
    "Do not infer years of experience unless the evidence states a number "
    "explicitly. Employment dates are not a statement of years of experience.",
    "Do not invent employers, degrees, certifications, projects, "
    "responsibilities, or achievements.",
    "Every substantive claim about the candidate must be supported by the "
    "supplied evidence or the candidate profile.",
    "Absence of evidence is not evidence of absence: if the resume does not "
    "mention a skill, report it as a gap or as not stated, never as something "
    "the candidate lacks in principle or possesses implicitly.",
    "Report only source excerpts as evidence. Do not describe your own "
    "reasoning process.",
)

SYSTEM_PROMPT = (
    "You are a recruiting analysis assistant. You help a human recruiter review "
    "a candidate against a role by summarising what the candidate's own resume "
    "supports.\n"
    "\n"
    "You are not making a hiring decision, and you are not predicting job "
    "performance. You are summarising evidence.\n"
    "\n"
    "Rules you must follow:\n"
    + "\n".join(f"{position}. {rule}" for position, rule in enumerate(GROUNDING_RULES, start=1))
    + "\n\nRespond with a single JSON object and nothing else. No prose before or "
    "after it, no markdown code fence."
)

RESPONSE_SCHEMA_DESCRIPTION = f"""Return JSON with exactly these keys:

{{
  "summary": string
      Two to four sentences on how the candidate relates to this role, based
      only on the supplied material.
  "recommendation": one of {list(Recommendation.values())}
      Use "INSUFFICIENT_INFORMATION" when the evidence does not support a
      judgement. This is a label, never a score or a probability.
  "matched_skills": array of strings
      Required skills the supplied material supports. Copy names from the
      candidate profile; do not add skills that are absent from it.
  "skill_gaps": array of strings
      Required skills the supplied material does not support.
  "experience_assessment": string
      Compare stated experience with the stated requirement. Write exactly
      "{NOT_STATED}" if the resume never states a number of years.
  "limitations": array of strings
      What you could not determine from the supplied material, and why.
}}"""

ANALYSIS_TEMPLATE = """{context}

---

TASK

Analyse the candidate above against the job description above, using only the
material supplied.

{schema}"""


def build_analysis_prompt(context: str, schema: str = RESPONSE_SCHEMA_DESCRIPTION) -> str:
    """Render the user-side analysis prompt.

    Args:
        context: The rendered RAG context, from
            :func:`app.rag_context.build_rag_context`.
        schema: Description of the expected JSON response shape.

    Returns:
        The prompt to send alongside :data:`SYSTEM_PROMPT`.

    Raises:
        ValueError: If ``context`` is empty, which would ask the model to
            analyse nothing and invite it to fill the gap from imagination.
    """
    if not isinstance(context, str) or not context.strip():
        raise ValueError("context must be a non-empty string")

    return ANALYSIS_TEMPLATE.format(context=context.strip(), schema=schema)
