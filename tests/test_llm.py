"""Unit tests for app.llm and app.prompts.

No test here reaches the network or needs an API key.
"""

from __future__ import annotations

import json

import pytest

from app.llm import (
    DEFAULT_ANTHROPIC_MODEL,
    AnthropicProvider,
    FakeLLMProvider,
    LangChainProvider,
    LLMCallError,
    LLMConfigurationError,
    LLMError,
    LLMProvider,
    ScriptedLLMProvider,
    get_llm_provider,
)
from app.models import NOT_STATED, Recommendation
from app.prompts import (
    GROUNDING_RULES,
    RESPONSE_SCHEMA_DESCRIPTION,
    SYSTEM_PROMPT,
    build_analysis_prompt,
)

CONTEXT = """JOB DESCRIPTION:
Financial analyst with Python and SQL.

CANDIDATE PROFILE:
Candidate: Sarah Wilson
Skills found on resume: Python, SQL, Excel
Matched skills: Python, SQL
Missing skills: Tableau
Skill coverage: 2/3 required skills
Experience stated on resume: 4 years (stated on resume)
Experience required by job: 3 years
Meets stated experience requirement: yes
Education: MBA - Finance

RETRIEVED RESUME EVIDENCE:

[Chunk 1] (chunk_id=sarah#0, similarity=0.6000)
Built forecasts in Excel and automated reporting with Python and SQL.
"""


@pytest.fixture
def prompt() -> str:
    """A realistic analysis prompt built from the context above."""
    return build_analysis_prompt(CONTEXT)


class TestFakeProvider:
    def test_satisfies_the_protocol(self) -> None:
        assert isinstance(FakeLLMProvider(), LLMProvider)

    def test_reports_a_name(self) -> None:
        assert FakeLLMProvider().name == "fake/deterministic-v1"

    def test_returns_valid_json(self, prompt: str) -> None:
        payload = json.loads(FakeLLMProvider().generate(prompt))
        assert isinstance(payload, dict)

    def test_response_is_deterministic(self, prompt: str) -> None:
        assert FakeLLMProvider().generate(prompt) == FakeLLMProvider().generate(prompt)

    def test_response_carries_every_expected_key(self, prompt: str) -> None:
        payload = json.loads(FakeLLMProvider().generate(prompt))
        assert set(payload) == {
            "summary",
            "recommendation",
            "matched_skills",
            "skill_gaps",
            "experience_assessment",
            "limitations",
        }

    def test_recommendation_is_in_the_controlled_vocabulary(self, prompt: str) -> None:
        payload = json.loads(FakeLLMProvider().generate(prompt))
        assert payload["recommendation"] in Recommendation.values()

    def test_skills_are_read_from_the_supplied_context(self, prompt: str) -> None:
        """The fake must be grounded, not inventive."""
        payload = json.loads(FakeLLMProvider().generate(prompt))
        assert payload["matched_skills"] == ["Python", "SQL"]
        assert payload["skill_gaps"] == ["Tableau"]

    def test_experience_reflects_the_context(self, prompt: str) -> None:
        payload = json.loads(FakeLLMProvider().generate(prompt))
        assert "4 years" in payload["experience_assessment"]

    def test_unknown_experience_stays_unknown(self) -> None:
        context = CONTEXT.replace(
            "Experience stated on resume: 4 years (stated on resume)",
            f"Experience stated on resume: {NOT_STATED}",
        )
        payload = json.loads(FakeLLMProvider().generate(build_analysis_prompt(context)))

        assert payload["experience_assessment"].startswith(NOT_STATED)
        assert "4 years" not in payload["experience_assessment"]

    def test_records_calls(self, prompt: str) -> None:
        provider = FakeLLMProvider()
        provider.generate(prompt, system="sys")
        assert provider.calls == [(prompt, "sys")]

    def test_high_coverage_yields_a_strong_recommendation(self) -> None:
        context = CONTEXT.replace("Missing skills: Tableau", "Missing skills: none identified")
        payload = json.loads(FakeLLMProvider().generate(build_analysis_prompt(context)))
        assert payload["recommendation"] == Recommendation.STRONG_MATCH.value

    def test_no_matched_skills_yields_a_weak_recommendation(self) -> None:
        context = CONTEXT.replace("Matched skills: Python, SQL", "Matched skills: none identified")
        payload = json.loads(FakeLLMProvider().generate(build_analysis_prompt(context)))
        assert payload["recommendation"] == Recommendation.WEAK_MATCH.value

    def test_missing_evidence_yields_insufficient_information(self) -> None:
        context = CONTEXT.split("RETRIEVED RESUME EVIDENCE")[0].replace(
            "Matched skills: Python, SQL", "Matched skills: none identified"
        ).replace("Missing skills: Tableau", "Missing skills: none identified")
        payload = json.loads(FakeLLMProvider().generate(build_analysis_prompt(context)))
        assert payload["recommendation"] == Recommendation.INSUFFICIENT_INFORMATION.value

    @pytest.mark.parametrize("bad", ["", "   ", None, 42])
    def test_empty_prompt_raises(self, bad: object) -> None:
        with pytest.raises(LLMCallError):
            FakeLLMProvider().generate(bad)  # type: ignore[arg-type]


class TestScriptedProvider:
    def test_returns_responses_in_order(self) -> None:
        provider = ScriptedLLMProvider(["first", "second"])
        assert provider.generate("p") == "first"
        assert provider.generate("p") == "second"

    def test_repeats_the_last_response_by_default(self) -> None:
        provider = ScriptedLLMProvider(["only"])
        assert provider.generate("p") == "only"
        assert provider.generate("p") == "only"

    def test_can_raise_when_exhausted(self) -> None:
        provider = ScriptedLLMProvider(["only"], repeat_last=False)
        provider.generate("p")
        with pytest.raises(LLMCallError, match="exhausted"):
            provider.generate("p")

    def test_empty_script_raises(self) -> None:
        with pytest.raises(LLMConfigurationError):
            ScriptedLLMProvider([])

    def test_single_string_is_rejected(self) -> None:
        """A bare string would otherwise be read as a script of characters."""
        with pytest.raises(LLMConfigurationError):
            ScriptedLLMProvider("a response")  # type: ignore[arg-type]

    def test_satisfies_the_protocol(self) -> None:
        assert isinstance(ScriptedLLMProvider(["x"]), LLMProvider)

    def test_records_prompts(self) -> None:
        provider = ScriptedLLMProvider(["x"])
        provider.generate("the prompt", system="the system")
        assert provider.calls == [("the prompt", "the system")]


class TestAnthropicProvider:
    def test_construction_does_not_need_credentials(self) -> None:
        """Constructing must be free; only generate() touches the network."""
        provider = AnthropicProvider()
        assert provider.name == f"anthropic/{DEFAULT_ANTHROPIC_MODEL}"

    def test_satisfies_the_protocol(self) -> None:
        assert isinstance(AnthropicProvider(), LLMProvider)

    def test_custom_model_appears_in_the_name(self) -> None:
        assert AnthropicProvider(model="claude-haiku-4-5").name == "anthropic/claude-haiku-4-5"

    def test_empty_prompt_raises_before_any_network_call(self) -> None:
        with pytest.raises(LLMCallError):
            AnthropicProvider().generate("   ")

    def test_missing_sdk_reports_a_configuration_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import builtins

        real_import = builtins.__import__

        def fail_on_anthropic(name, *args, **kwargs):
            if name == "anthropic":
                raise ImportError("no module named anthropic")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fail_on_anthropic)

        with pytest.raises(LLMConfigurationError, match="pip install anthropic"):
            AnthropicProvider().generate("a prompt")

    def test_text_blocks_are_concatenated(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class Block:
            def __init__(self, text: str) -> None:
                self.type = "text"
                self.text = text

        class Response:
            content = [Block("part one "), Block("part two")]
            stop_reason = "end_turn"

        class Messages:
            @staticmethod
            def create(**_kwargs):
                return Response()

        provider = AnthropicProvider()
        monkeypatch.setattr(provider, "_load_client", lambda: type("C", (), {"messages": Messages})())

        assert provider.generate("prompt") == "part one part two"

    def test_non_text_blocks_are_ignored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class Thinking:
            type = "thinking"
            thinking = "internal reasoning that must not leak"

        class Text:
            type = "text"
            text = "the answer"

        class Response:
            content = [Thinking(), Text()]
            stop_reason = "end_turn"

        provider = AnthropicProvider()
        monkeypatch.setattr(
            provider,
            "_load_client",
            lambda: type("C", (), {"messages": type("M", (), {"create": staticmethod(lambda **k: Response())})})(),
        )

        assert provider.generate("prompt") == "the answer"

    def test_empty_response_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class Response:
            content = []
            stop_reason = "max_tokens"

        provider = AnthropicProvider()
        monkeypatch.setattr(
            provider,
            "_load_client",
            lambda: type("C", (), {"messages": type("M", (), {"create": staticmethod(lambda **k: Response())})})(),
        )

        with pytest.raises(LLMCallError, match="no text"):
            provider.generate("prompt")

    def test_system_prompt_is_forwarded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict = {}

        class Block:
            type = "text"
            text = "ok"

        class Response:
            content = [Block()]
            stop_reason = "end_turn"

        def create(**kwargs):
            captured.update(kwargs)
            return Response()

        provider = AnthropicProvider()
        monkeypatch.setattr(
            provider,
            "_load_client",
            lambda: type("C", (), {"messages": type("M", (), {"create": staticmethod(create)})})(),
        )

        provider.generate("prompt", system="be grounded")
        assert captured["system"] == "be grounded"
        assert captured["messages"] == [{"role": "user", "content": "prompt"}]


class TestLangChainProvider:
    class StubChatModel:
        """Minimal stand-in shaped like a LangChain chat model."""

        model_name = "stub-model"

        def __init__(self, content="the response") -> None:
            self.content = content
            self.received: list = []

        def invoke(self, messages):
            self.received.append(messages)
            return type("AIMessage", (), {"content": self.content})()

    def test_wraps_a_chat_model(self) -> None:
        provider = LangChainProvider(self.StubChatModel())
        assert provider.generate("prompt") == "the response"

    def test_name_includes_the_model(self) -> None:
        assert LangChainProvider(self.StubChatModel()).name == "langchain/stub-model"

    def test_satisfies_the_protocol(self) -> None:
        assert isinstance(LangChainProvider(self.StubChatModel()), LLMProvider)

    def test_system_prompt_becomes_a_system_message(self) -> None:
        model = self.StubChatModel()
        LangChainProvider(model).generate("prompt", system="be grounded")

        assert model.received[0][0] == ("system", "be grounded")
        assert model.received[0][1] == ("human", "prompt")

    def test_no_system_prompt_sends_only_the_human_turn(self) -> None:
        model = self.StubChatModel()
        LangChainProvider(model).generate("prompt")
        assert model.received[0] == [("human", "prompt")]

    def test_list_content_blocks_are_joined(self) -> None:
        model = self.StubChatModel(content=[{"text": "part one "}, {"text": "part two"}])
        assert LangChainProvider(model).generate("prompt") == "part one part two"

    def test_non_chat_model_raises(self) -> None:
        with pytest.raises(LLMConfigurationError, match="not a LangChain chat model"):
            LangChainProvider(object())

    def test_failure_is_wrapped(self) -> None:
        class Broken:
            model_name = "broken"

            def invoke(self, messages):
                raise RuntimeError("upstream exploded")

        with pytest.raises(LLMCallError, match="upstream exploded"):
            LangChainProvider(Broken()).generate("prompt")

    def test_empty_response_raises(self) -> None:
        with pytest.raises(LLMCallError, match="no text"):
            LangChainProvider(self.StubChatModel(content="")).generate("prompt")


class TestProviderConfiguration:
    @pytest.fixture(autouse=True)
    def _clear_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for name in ("LLM_PROVIDER", "LLM_MODEL", "LLM_API_KEY", "ANTHROPIC_API_KEY", "LLM_MAX_TOKENS"):
            monkeypatch.delenv(name, raising=False)

    def test_defaults_to_the_fake_provider(self) -> None:
        """No configuration and no key must still give a working provider."""
        assert isinstance(get_llm_provider(), FakeLLMProvider)

    def test_environment_selects_the_provider(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        assert isinstance(get_llm_provider(), AnthropicProvider)

    def test_argument_overrides_the_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        assert isinstance(get_llm_provider(provider="fake"), FakeLLMProvider)

    def test_provider_name_is_case_insensitive(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_PROVIDER", "  ANTHROPIC  ")
        assert isinstance(get_llm_provider(), AnthropicProvider)

    def test_model_comes_from_the_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        monkeypatch.setenv("LLM_MODEL", "claude-haiku-4-5")
        assert get_llm_provider().name == "anthropic/claude-haiku-4-5"

    def test_anthropic_provider_builds_without_a_key(self) -> None:
        """A missing key must not fail until a call is actually attempted."""
        assert get_llm_provider(provider="anthropic") is not None

    def test_unknown_provider_raises(self) -> None:
        with pytest.raises(LLMConfigurationError, match="unknown LLM_PROVIDER"):
            get_llm_provider(provider="not-a-provider")

    def test_langchain_cannot_be_built_from_configuration(self) -> None:
        with pytest.raises(LLMConfigurationError, match="LangChainProvider"):
            get_llm_provider(provider="langchain")

    def test_invalid_max_tokens_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        monkeypatch.setenv("LLM_MAX_TOKENS", "lots")
        with pytest.raises(LLMConfigurationError, match="LLM_MAX_TOKENS"):
            get_llm_provider()

    def test_all_errors_share_a_base_class(self) -> None:
        with pytest.raises(LLMError):
            get_llm_provider(provider="nope")


class TestPrompts:
    def test_system_prompt_states_the_role(self) -> None:
        assert "recruiting analysis assistant" in SYSTEM_PROMPT

    def test_grounding_instruction_is_present(self) -> None:
        assert "Use ONLY the job description, candidate profile, and retrieved resume" in SYSTEM_PROMPT

    def test_no_invention_instruction_is_present(self) -> None:
        assert "Do not invent candidate information." in SYSTEM_PROMPT

    def test_unknown_information_instruction_is_present(self) -> None:
        assert NOT_STATED in SYSTEM_PROMPT
        assert "not present" in SYSTEM_PROMPT

    def test_no_inferred_experience_instruction_is_present(self) -> None:
        assert "Do not infer years of experience" in SYSTEM_PROMPT
        assert "Employment dates are not a statement" in SYSTEM_PROMPT

    def test_no_invented_credentials_instruction_is_present(self) -> None:
        assert "Do not invent employers, degrees, certifications" in SYSTEM_PROMPT

    def test_no_inferred_skills_instruction_is_present(self) -> None:
        assert "Do not infer skills from unrelated terminology" in SYSTEM_PROMPT

    def test_evidence_support_instruction_is_present(self) -> None:
        assert "must be supported by the supplied evidence" in SYSTEM_PROMPT

    def test_chain_of_thought_is_not_requested(self) -> None:
        assert "Do not describe your own" in SYSTEM_PROMPT

    def test_every_rule_appears_in_the_system_prompt(self) -> None:
        for rule in GROUNDING_RULES:
            assert rule in SYSTEM_PROMPT

    def test_rules_are_numbered(self) -> None:
        assert "1. Use ONLY" in SYSTEM_PROMPT

    def test_json_only_instruction_is_present(self) -> None:
        assert "single JSON object" in SYSTEM_PROMPT

    def test_schema_lists_every_field(self) -> None:
        for field in (
            "summary",
            "recommendation",
            "matched_skills",
            "skill_gaps",
            "experience_assessment",
            "limitations",
        ):
            assert field in RESPONSE_SCHEMA_DESCRIPTION

    def test_schema_lists_the_controlled_vocabulary(self) -> None:
        for value in Recommendation.values():
            assert value in RESPONSE_SCHEMA_DESCRIPTION

    def test_prompt_embeds_the_context(self) -> None:
        assert "Sarah Wilson" in build_analysis_prompt(CONTEXT)

    def test_prompt_embeds_the_schema(self) -> None:
        assert "matched_skills" in build_analysis_prompt(CONTEXT)

    @pytest.mark.parametrize("bad", ["", "   ", None, 42])
    def test_empty_context_raises(self, bad: object) -> None:
        """Asking for analysis of nothing invites the model to invent."""
        with pytest.raises(ValueError):
            build_analysis_prompt(bad)  # type: ignore[arg-type]
