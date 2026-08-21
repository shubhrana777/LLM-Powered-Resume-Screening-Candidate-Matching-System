"""Tests for POST /match-candidates."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pytest

from app.api.config import Settings
from app.api.schemas import MAX_TOP_K, MatchResponse
from app.api.service import ScreeningService
from tests.conftest import ANALYST_JOB_DESCRIPTION, FakeEmbedder, build_test_client


def _match(client, **body) -> dict:
    payload = {"job_description": ANALYST_JOB_DESCRIPTION}
    payload.update(body)
    response = client.post("/match-candidates", json=payload)
    assert response.status_code == 200, response.text
    return response.json()


class CountingEmbedder:
    """Wraps an embedder and counts how often a batch is embedded."""

    def __init__(self, inner: FakeEmbedder) -> None:
        self._inner = inner
        self.batch_calls = 0

    @property
    def dimension(self) -> int:
        return self._inner.dimension

    def embed_text(self, text: str) -> np.ndarray:
        return self._inner.embed_text(text)

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        self.batch_calls += 1
        return self._inner.embed_texts(texts)


# --------------------------------------------------------------------------
# Success
# --------------------------------------------------------------------------


def test_valid_request_returns_200(api_client):
    response = api_client.post(
        "/match-candidates", json={"job_description": ANALYST_JOB_DESCRIPTION}
    )
    assert response.status_code == 200


def test_response_matches_its_schema(api_client):
    MatchResponse.model_validate(_match(api_client))


def test_every_candidate_is_ranked(api_client):
    assert _match(api_client)["count"] == 3


def test_ranks_start_at_one_and_are_consecutive(api_client):
    ranks = [entry["rank"] for entry in _match(api_client)["results"]]
    assert ranks == [1, 2, 3]


def test_the_best_match_is_ranked_first(api_client):
    """The analyst job should rank the analyst above the graphic designer."""
    results = _match(api_client)["results"]
    assert results[0]["candidate_id"] == "sarah_wilson"
    assert results[-1]["candidate_id"] == "nina_volkov"


def test_scores_are_ordered_best_first(api_client):
    scores = [entry["similarity_score"] for entry in _match(api_client)["results"]]
    assert scores == sorted(scores, reverse=True)


def test_scores_are_within_the_cosine_range(api_client):
    for entry in _match(api_client)["results"]:
        assert -1.0 <= entry["similarity_score"] <= 1.0


def test_results_carry_display_names(api_client):
    assert _match(api_client)["results"][0]["candidate"] == "Sarah Wilson"


def test_candidates_considered_reports_the_pool_size(api_client):
    assert _match(api_client, top_k=1)["candidates_considered"] == 3


def test_score_is_labelled_as_a_similarity_not_a_probability(api_client):
    body = _match(api_client)
    assert body["score_type"] == "cosine_similarity"
    assert "not a probability" in body["score_note"].lower()


def test_the_api_score_is_the_engine_score(api_client, api_service):
    """The API rounds for display and changes nothing else."""
    from_api = {
        entry["candidate_id"]: entry["similarity_score"] for entry in _match(api_client)["results"]
    }
    from_engine, _ = api_service.match(ANALYST_JOB_DESCRIPTION, top_k=3)

    for result in from_engine:
        assert from_api[result.candidate_id] == pytest.approx(result.similarity_score, abs=1e-4)


# --------------------------------------------------------------------------
# top_k
# --------------------------------------------------------------------------


@pytest.mark.parametrize("top_k", [1, 2, 3])
def test_top_k_limits_the_result_count(api_client, top_k: int):
    assert _match(api_client, top_k=top_k)["count"] == top_k


def test_top_k_larger_than_the_pool_returns_everything(api_client):
    assert _match(api_client, top_k=50)["count"] == 3


def test_top_k_defaults_when_omitted(api_client):
    assert _match(api_client)["count"] == 3


@pytest.mark.parametrize("top_k", [0, -1, MAX_TOP_K + 1])
def test_out_of_range_top_k_is_rejected(api_client, top_k: int):
    response = api_client.post(
        "/match-candidates",
        json={"job_description": ANALYST_JOB_DESCRIPTION, "top_k": top_k},
    )
    assert response.status_code == 422


def test_non_integer_top_k_is_rejected(api_client):
    response = api_client.post(
        "/match-candidates",
        json={"job_description": ANALYST_JOB_DESCRIPTION, "top_k": "five"},
    )
    assert response.status_code == 422


# --------------------------------------------------------------------------
# Invalid requests
# --------------------------------------------------------------------------


@pytest.mark.parametrize("job_description", ["", "   ", "\n\t "])
def test_empty_job_description_is_rejected(api_client, job_description: str):
    response = api_client.post("/match-candidates", json={"job_description": job_description})
    assert response.status_code == 422
    assert response.json()["code"] == "validation_error"


def test_missing_job_description_is_rejected(api_client):
    response = api_client.post("/match-candidates", json={"top_k": 3})
    assert response.status_code == 422
    assert any(
        "job_description" in error["field"] for error in response.json()["errors"]
    )


def test_absurdly_long_job_description_is_rejected(api_client):
    response = api_client.post("/match-candidates", json={"job_description": "x" * 20_001})
    assert response.status_code == 422


def test_no_body_at_all_is_rejected(api_client):
    assert api_client.post("/match-candidates").status_code == 422


# --------------------------------------------------------------------------
# Resume directory problems
# --------------------------------------------------------------------------


def test_missing_resume_directory_returns_404(tmp_path: Path, analyst_embedder):
    settings = Settings(resume_dir=tmp_path / "nowhere")
    service = ScreeningService(settings, embedder=analyst_embedder)

    with build_test_client(settings, service) as client:
        response = client.post("/match-candidates", json={"job_description": "anything at all"})

    assert response.status_code == 404
    assert response.json()["code"] == "resume_directory_not_found"


def test_empty_resume_directory_returns_404(tmp_path: Path, analyst_embedder):
    empty = tmp_path / "empty"
    empty.mkdir()
    settings = Settings(resume_dir=empty)
    service = ScreeningService(settings, embedder=analyst_embedder)

    with build_test_client(settings, service) as client:
        response = client.post("/match-candidates", json={"job_description": "anything at all"})

    assert response.status_code == 404
    assert response.json()["code"] == "no_resumes"


def test_a_corrupt_resume_does_not_break_matching(
    analyst_resume_dir: Path, analyst_embedder
):
    (analyst_resume_dir / "broken.pdf").write_bytes(b"not a pdf")
    settings = Settings(resume_dir=analyst_resume_dir)
    service = ScreeningService(settings, embedder=analyst_embedder)

    with build_test_client(settings, service) as client:
        response = client.post(
            "/match-candidates", json={"job_description": ANALYST_JOB_DESCRIPTION}
        )

    assert response.status_code == 200
    assert response.json()["count"] == 3


# --------------------------------------------------------------------------
# Cost
# --------------------------------------------------------------------------


def test_resumes_are_embedded_once_across_requests(api_settings, analyst_embedder):
    """The model must not be re-run over the pool on every request."""
    embedder = CountingEmbedder(analyst_embedder)
    service = ScreeningService(api_settings, embedder=embedder)

    with build_test_client(api_settings, service) as client:
        for _ in range(3):
            client.post("/match-candidates", json={"job_description": ANALYST_JOB_DESCRIPTION})

    assert embedder.batch_calls == 1
