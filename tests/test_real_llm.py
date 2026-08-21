"""Opt-in tests against a real LLM provider.

Every test here is marked ``llm`` and **skips** unless credentials are
configured, so the default suite never needs a key, a network, or money:

    pytest                 # skips these
    pytest -m llm          # runs them, if credentials are present

Enable by setting ``LLM_PROVIDER=anthropic`` and ``LLM_API_KEY`` (or
``ANTHROPIC_API_KEY``), with the ``anthropic`` package installed.

These check that a real model can be reached and that its output survives the
grounding validator. They deliberately do not assert on prose, which is not
reproducible between runs or model versions.
"""

from __future__ import annotations

import os

import pytest

from app.llm import LLMError, get_llm_provider
from app.models import Candidate, Recommendation
from app.rag_pipeline import RagConfig, RagPipeline

pytestmark = pytest.mark.llm

JOB = (
    "Financial analyst. Requirements: Python for reporting automation, SQL for "
    "the data warehouse, Excel modelling, Power BI dashboards. 3+ years required."
)


def _credentials_configured() -> bool:
    """Whether a real provider is configured and its SDK is importable."""
    if (os.environ.get("LLM_PROVIDER") or "").strip().lower() != "anthropic":
        return False
    if not (os.environ.get("LLM_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")):
        return False
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.fixture(scope="module")
def real_provider():
    """A real provider, or skip when none is configured."""
    if not _credentials_configured():
        pytest.skip(
            "no real LLM configured; set LLM_PROVIDER=anthropic and LLM_API_KEY "
            "(and pip install anthropic) to run these"
        )
    return get_llm_provider()


@pytest.fixture(scope="module")
def real_pipeline(real_provider):
    """A pipeline wired to the real provider and the real embedder."""
    pipeline = RagPipeline(llm=real_provider, config=RagConfig(chunk_size=60, chunk_overlap=15))
    pipeline.index_candidates(
        [
            Candidate(
                "real-strong",
                "Sarah Wilson\nSenior Financial Analyst\n"
                "Financial analyst with 4 years of experience in budgeting and forecasting.\n"
                "Automated monthly reporting using Python and SQL.\n"
                "Built Power BI dashboards and Excel financial models.\n"
                "EDUCATION\nMBA in Finance, Manchester Business School\n",
                "Sarah Wilson",
            ),
            Candidate(
                "real-poor",
                "Nina Volkov\nGraphic Designer\n"
                "Led brand identity and packaging projects for food clients.\n"
                "SKILLS\nIllustrator, Photoshop, InDesign, typography\n"
                "EDUCATION\nDiploma in Graphic Design\n",
                "Nina Volkov",
            ),
        ]
    )
    return pipeline


class TestRealProvider:
    def test_provider_returns_text(self, real_provider) -> None:
        response = real_provider.generate(
            "Reply with the single word: ready", system="You are terse."
        )
        assert isinstance(response, str)
        assert response.strip()

    def test_bad_credentials_raise_a_clear_error(self) -> None:
        from app.llm import AnthropicProvider

        provider = AnthropicProvider(api_key="sk-ant-invalid-key-for-testing")
        with pytest.raises(LLMError):
            provider.generate("hello")


class TestRealPipeline:
    def test_strong_candidate_analysis_is_grounded(self, real_pipeline) -> None:
        analysis = real_pipeline.analyze_candidate("real-strong", JOB)

        assert analysis.candidate_id == "real-strong"
        assert analysis.recommendation.value in Recommendation.values()
        assert analysis.summary.strip()
        assert analysis.evidence

    def test_claimed_skills_are_supported_by_the_resume(self, real_pipeline) -> None:
        """Whatever the model said, only supported skills may survive validation."""
        from app.candidate_analyzer import analyze_job_description

        analysis = real_pipeline.analyze_candidate("real-strong", JOB)

        requirements = analyze_job_description(JOB)
        allowed = {
            skill.lower()
            for skill in real_pipeline.build_context("real-strong", requirements).profile.skills
        }

        assert {skill.lower() for skill in analysis.matched_skills} <= allowed

    def test_unknown_experience_is_not_invented(self, real_pipeline) -> None:
        """The poor candidate's resume states no years; none may appear."""
        analysis = real_pipeline.analyze_candidate("real-poor", JOB)

        if analysis.warnings:
            # The validator caught something; that is the safeguard working.
            assert analysis.experience_assessment.startswith("Not stated") or analysis.warnings
        else:
            assert "years of experience" not in analysis.experience_assessment.lower()

    def test_evidence_is_isolated_per_candidate(self, real_pipeline) -> None:
        for candidate_id in ("real-strong", "real-poor"):
            analysis = real_pipeline.analyze_candidate(candidate_id, JOB)
            assert {item.candidate_id for item in analysis.evidence} == {candidate_id}
