"""Tests for POST /analyze-candidate.

The point of these tests is not that the endpoint returns JSON -- it is that
routing the Phase 4 pipeline through HTTP does not weaken it. Two properties
matter most and are asserted directly rather than assumed:

* **Isolation.** Evidence returned for one candidate belongs only to that
  candidate.
* **Grounding.** A model that invents facts is corrected before the response is
  built, exactly as it is on the CLI path. The client cannot supply candidate
  information, so it cannot route around the check.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.api.config import Settings
from app.api.schemas import AnalysisResponse
from app.api.service import ScreeningService
from app.llm import LLMCallError, ScriptedLLMProvider
from app.models import Recommendation
from tests.conftest import ANALYST_JOB_DESCRIPTION, build_test_client


def _analyze(client, candidate: str = "sarah_wilson", **body) -> dict:
    payload = {"candidate": candidate, "job_description": ANALYST_JOB_DESCRIPTION}
    payload.update(body)
    response = client.post("/analyze-candidate", json=payload)
    assert response.status_code == 200, response.text
    return response.json()


def _client_with_provider(settings, embedder, provider):
    service = ScreeningService(settings, embedder=embedder, llm=provider)
    return build_test_client(settings, service)


# --------------------------------------------------------------------------
# Success
# --------------------------------------------------------------------------


def test_valid_request_returns_200(api_client):
    response = api_client.post(
        "/analyze-candidate",
        json={"candidate": "sarah_wilson", "job_description": ANALYST_JOB_DESCRIPTION},
    )
    assert response.status_code == 200


def test_response_matches_its_schema(api_client):
    AnalysisResponse.model_validate(_analyze(api_client))


def test_response_identifies_the_candidate(api_client):
    body = _analyze(api_client)
    assert body["candidate_id"] == "sarah_wilson"
    assert body["candidate"] == "Sarah Wilson"


def test_recommendation_is_from_the_controlled_vocabulary(api_client):
    assert _analyze(api_client)["recommendation"] in Recommendation.values()


def test_recommendation_is_labelled_as_not_a_score(api_client):
    note = _analyze(api_client)["recommendation_note"].lower()
    assert "not a score" in note
    assert "not a probability" in note


def test_response_carries_a_summary(api_client):
    assert _analyze(api_client)["summary"].strip()


def test_response_carries_matched_skills(api_client):
    assert "Excel" in _analyze(api_client)["matched_skills"]


def test_response_carries_skill_gaps(api_client):
    """The analyst fixture has no Tableau; the job asks for it."""
    assert "Tableau" in _analyze(api_client)["skill_gaps"]


def test_response_carries_an_experience_assessment(api_client):
    assert _analyze(api_client)["experience_assessment"].strip()


def test_a_candidate_who_states_no_experience_is_not_guessed_at(api_client):
    assessment = _analyze(api_client, candidate="nina_volkov")["experience_assessment"]
    assert assessment.startswith("Not stated")


def test_response_carries_evidence(api_client):
    evidence = _analyze(api_client)["evidence"]
    assert evidence
    assert all(item["text"].strip() for item in evidence)


def test_evidence_scores_are_similarities(api_client):
    for item in _analyze(api_client)["evidence"]:
        assert -1.0 <= item["retrieval_score"] <= 1.0


def test_response_reports_the_model(api_client):
    assert _analyze(api_client)["model"] == "fake/deterministic-v1"


def test_a_clean_analysis_is_grounded(api_client):
    body = _analyze(api_client)
    assert body["is_grounded"] is True
    assert body["warnings"] == []


# --------------------------------------------------------------------------
# Candidate resolution
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "reference",
    ["sarah_wilson", "SARAH_WILSON", "Sarah Wilson", "sarah wilson", "sarah_wilson.pdf"],
)
def test_a_candidate_can_be_named_several_ways(api_client, reference: str):
    assert _analyze(api_client, candidate=reference)["candidate_id"] == "sarah_wilson"


def test_unknown_candidate_returns_404(api_client):
    response = api_client.post(
        "/analyze-candidate",
        json={"candidate": "nobody_at_all", "job_description": ANALYST_JOB_DESCRIPTION},
    )
    assert response.status_code == 404
    assert response.json()["code"] == "candidate_not_found"


def test_unknown_candidate_message_lists_what_is_available(api_client):
    response = api_client.post(
        "/analyze-candidate",
        json={"candidate": "nobody_at_all", "job_description": ANALYST_JOB_DESCRIPTION},
    )
    assert "sarah_wilson" in response.json()["detail"]


@pytest.mark.parametrize(
    "reference",
    ["../../../etc/passwd", "..\\..\\secrets.pdf", "data/resumes/sarah_wilson.pdf"],
)
def test_a_candidate_reference_may_not_be_a_path(api_client, reference: str):
    response = api_client.post(
        "/analyze-candidate",
        json={"candidate": reference, "job_description": ANALYST_JOB_DESCRIPTION},
    )
    assert response.status_code == 422


# --------------------------------------------------------------------------
# Invalid requests
# --------------------------------------------------------------------------


@pytest.mark.parametrize("job_description", ["", "   "])
def test_empty_job_description_is_rejected(api_client, job_description: str):
    response = api_client.post(
        "/analyze-candidate",
        json={"candidate": "sarah_wilson", "job_description": job_description},
    )
    assert response.status_code == 422


@pytest.mark.parametrize("candidate", ["", "   "])
def test_empty_candidate_is_rejected(api_client, candidate: str):
    response = api_client.post(
        "/analyze-candidate",
        json={"candidate": candidate, "job_description": ANALYST_JOB_DESCRIPTION},
    )
    assert response.status_code == 422


def test_missing_fields_are_rejected(api_client):
    response = api_client.post("/analyze-candidate", json={"candidate": "sarah_wilson"})
    assert response.status_code == 422


# --------------------------------------------------------------------------
# Candidate isolation
# --------------------------------------------------------------------------


def test_evidence_belongs_only_to_the_requested_candidate(api_client):
    for reference in ("sarah_wilson", "james_patel", "nina_volkov"):
        body = _analyze(api_client, candidate=reference)
        assert {item["candidate_id"] for item in body["evidence"]} == {reference}


def test_evidence_never_carries_another_candidates_text(api_client):
    """Nina is a designer; none of her vocabulary may reach Sarah's analysis."""
    evidence = " ".join(item["text"] for item in _analyze(api_client)["evidence"]).lower()

    for foreign_term in ("illustrator", "photoshop", "nina", "reconciliations", "ridgeway"):
        assert foreign_term not in evidence


def test_chunk_ids_are_namespaced_by_candidate(api_client):
    body = _analyze(api_client, candidate="james_patel")
    assert all(item["chunk_id"].startswith("james_patel#") for item in body["evidence"])


def test_analysing_one_candidate_does_not_contaminate_the_next(api_client):
    first = _analyze(api_client, candidate="sarah_wilson")
    second = _analyze(api_client, candidate="nina_volkov")
    third = _analyze(api_client, candidate="sarah_wilson")

    assert {item["candidate_id"] for item in second["evidence"]} == {"nina_volkov"}
    assert third["evidence"] == first["evidence"]


# --------------------------------------------------------------------------
# Grounding: the API must not weaken the Phase 4 safeguards
# --------------------------------------------------------------------------

LYING_RESPONSE = json.dumps(
    {
        "summary": "Sarah Wilson has 12 years of experience and holds a PhD in Econometrics.",
        "recommendation": "DEFINITELY_HIRE",
        "matched_skills": ["Kubernetes", "AWS", "Docker"],
        "skill_gaps": ["Fortran"],
        "experience_assessment": "The resume states 12 years of experience.",
        "evidence": ["A passage that was never retrieved."],
        "limitations": [],
    }
)


@pytest.fixture
def lying_client(api_settings, analyst_embedder):
    """A client whose model fabricates skills, experience, a degree and a label."""
    with _client_with_provider(
        api_settings, analyst_embedder, ScriptedLLMProvider([LYING_RESPONSE])
    ) as client:
        yield client


def test_invented_skills_are_stripped(lying_client):
    body = _analyze(lying_client)
    for invented in ("Kubernetes", "AWS", "Docker"):
        assert invented not in body["matched_skills"]


def test_invented_gaps_are_stripped(lying_client):
    assert "Fortran" not in _analyze(lying_client)["skill_gaps"]


def test_an_invalid_recommendation_becomes_insufficient_information(lying_client):
    assert _analyze(lying_client)["recommendation"] == Recommendation.INSUFFICIENT_INFORMATION.value


def test_a_fabricated_analysis_is_reported_as_ungrounded(lying_client):
    body = _analyze(lying_client)
    assert body["is_grounded"] is False
    assert body["warnings"]


def test_the_model_cannot_supply_its_own_evidence(lying_client):
    """Evidence comes from retrieval. A model's claimed citation is ignored."""
    evidence = _analyze(lying_client)["evidence"]
    assert all(item["text"] != "A passage that was never retrieved." for item in evidence)
    assert all(item["candidate_id"] == "sarah_wilson" for item in evidence)


def test_a_client_cannot_inject_candidate_information(api_client):
    """Extra fields in the request body must not reach the analysis."""
    baseline = _analyze(api_client)

    injected = api_client.post(
        "/analyze-candidate",
        json={
            "candidate": "sarah_wilson",
            "job_description": ANALYST_JOB_DESCRIPTION,
            "matched_skills": ["Kubernetes"],
            "years_experience": 30,
            "resume_text": "Sarah Wilson is a Kubernetes expert with 30 years of experience.",
            "evidence": ["fabricated"],
        },
    )

    assert injected.status_code == 200
    assert injected.json() == baseline
    assert "Kubernetes" not in injected.json()["matched_skills"]


# --------------------------------------------------------------------------
# Provider failures
# --------------------------------------------------------------------------


class FailingProvider:
    """A provider that cannot be reached."""

    name = "failing/test"

    def generate(self, prompt: str, system: str | None = None) -> str:
        raise LLMCallError("upstream refused the connection")


class BabblingProvider:
    """A provider that returns something that is not an analysis."""

    name = "babbling/test"

    def generate(self, prompt: str, system: str | None = None) -> str:
        return "I would rather not answer that."


def test_a_provider_failure_is_reported_as_a_gateway_error(api_settings, analyst_embedder):
    with _client_with_provider(api_settings, analyst_embedder, FailingProvider()) as client:
        response = client.post(
            "/analyze-candidate",
            json={"candidate": "sarah_wilson", "job_description": ANALYST_JOB_DESCRIPTION},
        )

    assert response.status_code == 502
    assert response.json()["code"] == "llm_call_failed"


def test_a_provider_failure_does_not_leak_the_upstream_message(api_settings, analyst_embedder):
    with _client_with_provider(api_settings, analyst_embedder, FailingProvider()) as client:
        response = client.post(
            "/analyze-candidate",
            json={"candidate": "sarah_wilson", "job_description": ANALYST_JOB_DESCRIPTION},
        )

    assert "upstream refused the connection" not in response.text


def test_an_unparsable_response_is_reported_not_guessed_at(api_settings, analyst_embedder):
    with _client_with_provider(api_settings, analyst_embedder, BabblingProvider()) as client:
        response = client.post(
            "/analyze-candidate",
            json={"candidate": "sarah_wilson", "job_description": ANALYST_JOB_DESCRIPTION},
        )

    assert response.status_code == 502
    assert response.json()["code"] == "llm_response_invalid"


# --------------------------------------------------------------------------
# Resume directory problems
# --------------------------------------------------------------------------


def test_missing_resume_directory_returns_404(tmp_path: Path, analyst_embedder):
    settings = Settings(resume_dir=tmp_path / "nowhere")
    service = ScreeningService(settings, embedder=analyst_embedder)

    with build_test_client(settings, service) as client:
        response = client.post(
            "/analyze-candidate",
            json={"candidate": "sarah_wilson", "job_description": ANALYST_JOB_DESCRIPTION},
        )

    assert response.status_code == 404
