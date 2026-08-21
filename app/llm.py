"""LLM provider abstraction.

All model calls go through :class:`LLMProvider` so the rest of the application
never imports a vendor SDK. Swapping providers is a configuration change.

Providers
---------
``fake``
    :class:`FakeLLMProvider` -- deterministic, offline, no credentials. It reads
    the same context a real model would and produces valid grounded JSON. This
    is the **default**, so the CLI and the whole test suite work with no API key.
``anthropic``
    :class:`AnthropicProvider` -- the real Claude API. Opt-in, needs a key.
``langchain``
    :class:`LangChainProvider` -- wraps any LangChain chat model, which is where
    LangChain earns its place here: one adapter buys access to every provider
    LangChain supports without the domain layer knowing about any of them.

Configuration comes from the environment:

===================  ==========================================================
``LLM_PROVIDER``     ``fake`` (default), ``anthropic``, or ``langchain``
``LLM_MODEL``        Model id; defaults to :data:`DEFAULT_ANTHROPIC_MODEL`
``LLM_API_KEY``      API key. ``ANTHROPIC_API_KEY`` is also accepted.
``LLM_MAX_TOKENS``   Response cap; defaults to :data:`DEFAULT_MAX_TOKENS`
===================  ==========================================================

Keys are read from the environment only. Nothing here writes a key to disk or
logs one, and no key is required to run the tests.
"""

from __future__ import annotations

import json
import os
import re
from typing import Protocol, Sequence, runtime_checkable

from app.models import NOT_STATED, Recommendation

__all__ = [
    "LLMError",
    "LLMConfigurationError",
    "LLMCallError",
    "LLMProvider",
    "FakeLLMProvider",
    "ScriptedLLMProvider",
    "AnthropicProvider",
    "LangChainProvider",
    "get_llm_provider",
    "DEFAULT_ANTHROPIC_MODEL",
    "DEFAULT_MAX_TOKENS",
]

DEFAULT_ANTHROPIC_MODEL = "claude-opus-5"
DEFAULT_MAX_TOKENS = 4096
DEFAULT_PROVIDER = "fake"


class LLMError(Exception):
    """Base class for every error raised by this module."""


class LLMConfigurationError(LLMError):
    """The provider is misconfigured, e.g. an unknown name or a missing key."""


class LLMCallError(LLMError):
    """The provider was reached but the call failed."""


@runtime_checkable
class LLMProvider(Protocol):
    """Minimal interface the pipeline depends on.

    Deliberately tiny: one text-in, text-out call. Structured output is handled
    by :mod:`app.analysis_parser` rather than by provider-specific features, so
    every provider behaves identically from the pipeline's point of view.
    """

    @property
    def name(self) -> str:
        """Identifier of the provider and model, for display and audit."""

    def generate(self, prompt: str, system: str | None = None) -> str:
        """Return the model's text response to ``prompt``."""


def _validate_prompt(prompt: str) -> str:
    """Return the prompt, or raise if it carries no content."""
    if not isinstance(prompt, str) or not prompt.strip():
        raise LLMCallError("prompt must be a non-empty string")
    return prompt


class FakeLLMProvider:
    """A deterministic, offline stand-in for a real model.

    It parses the fixed-format ``CANDIDATE PROFILE`` block that
    :mod:`app.rag_context` renders and answers from that, so its output is
    always grounded in the supplied context and identical across runs.

    This makes the default CLI experience and the entire test suite work with no
    credentials. It is **not** a simulation of model quality: real prose,
    nuance, and real failure modes are absent. Tests that need to exercise
    malformed or hallucinated output should use :class:`ScriptedLLMProvider`.

    Args:
        model_name: Label reported by :attr:`name`.
    """

    def __init__(self, model_name: str = "deterministic-v1") -> None:
        self._model_name = model_name
        self.calls: list[tuple[str, str | None]] = []

    @property
    def name(self) -> str:
        """Provider and model label."""
        return f"fake/{self._model_name}"

    @staticmethod
    def _field(context: str, label: str) -> str:
        """Read one ``Label: value`` line out of the rendered profile."""
        match = re.search(rf"^{re.escape(label)}:[ \t]*(.*)$", context, re.MULTILINE)
        return match.group(1).strip() if match else ""

    @staticmethod
    def _skills(value: str) -> list[str]:
        """Split a rendered skill list, treating the empty marker as empty."""
        if not value or value == "none identified":
            return []
        return [skill.strip() for skill in value.split(",") if skill.strip()]

    def generate(self, prompt: str, system: str | None = None) -> str:
        """Produce grounded JSON derived from the supplied context.

        Args:
            prompt: The rendered analysis prompt.
            system: System prompt, recorded but not otherwise used.

        Returns:
            A JSON string matching the expected analysis schema.
        """
        _validate_prompt(prompt)
        self.calls.append((prompt, system))

        name = self._field(prompt, "Candidate") or "The candidate"
        matched = self._skills(self._field(prompt, "Matched skills"))
        missing = self._skills(self._field(prompt, "Missing skills"))
        stated = self._field(prompt, "Experience stated on resume")
        required = self._field(prompt, "Experience required by job")
        meets = self._field(prompt, "Meets stated experience requirement")
        has_evidence = "[Chunk 1]" in prompt

        total = len(matched) + len(missing)
        coverage = (len(matched) / total) if total else 0.0

        if not has_evidence or (total == 0 and not matched):
            recommendation = Recommendation.INSUFFICIENT_INFORMATION
        elif coverage >= 0.85:
            recommendation = Recommendation.STRONG_MATCH
        elif coverage >= 0.6:
            recommendation = Recommendation.GOOD_MATCH
        elif coverage >= 0.3:
            recommendation = Recommendation.PARTIAL_MATCH
        else:
            recommendation = Recommendation.WEAK_MATCH

        if stated == NOT_STATED:
            experience = (
                f"{NOT_STATED}. The resume does not state a number of years, so this "
                f"cannot be compared against the requirement of {required}."
            )
        else:
            experience = (
                f"The resume states {stated}; the job asks for {required}. "
                f"Requirement met: {meets}."
            )

        summary_parts = [
            f"{name} matches {len(matched)} of {total} skills identified in the job description."
            if total
            else f"{name} was compared against a job description with no recognised skills."
        ]
        if matched:
            summary_parts.append(f"Supported skills: {', '.join(matched)}.")
        if missing:
            summary_parts.append(f"Not supported by the resume: {', '.join(missing)}.")
        summary_parts.append(
            "This summary is derived only from the supplied profile and retrieved evidence."
        )

        limitations = [
            "Generated by the offline deterministic provider, not a real language model.",
        ]
        if stated == NOT_STATED:
            limitations.append("Years of experience are not stated on the resume.")
        if not has_evidence:
            limitations.append("No resume passages were retrieved for this candidate.")

        return json.dumps(
            {
                "summary": " ".join(summary_parts),
                "recommendation": recommendation.value,
                "matched_skills": matched,
                "skill_gaps": missing,
                "experience_assessment": experience,
                "limitations": limitations,
            },
            indent=2,
        )


class ScriptedLLMProvider:
    """Returns pre-set responses in order, for testing the layers around the LLM.

    Used to exercise malformed JSON, missing fields, and deliberately
    hallucinated claims -- cases a well-behaved provider would never produce but
    that the parser and validator must survive.

    Args:
        responses: Responses to return, one per call.
        model_name: Label reported by :attr:`name`.
        repeat_last: If ``True``, keep returning the final response once the
            script runs out instead of raising.
    """

    def __init__(
        self,
        responses: Sequence[str],
        model_name: str = "scripted",
        repeat_last: bool = True,
    ) -> None:
        if isinstance(responses, str):
            raise LLMConfigurationError("responses must be a sequence of strings, not one string")

        self._responses = list(responses)
        if not self._responses:
            raise LLMConfigurationError("responses must not be empty")

        self._model_name = model_name
        self._repeat_last = repeat_last
        self.calls: list[tuple[str, str | None]] = []

    @property
    def name(self) -> str:
        """Provider and model label."""
        return f"scripted/{self._model_name}"

    def generate(self, prompt: str, system: str | None = None) -> str:
        """Return the next scripted response.

        Raises:
            LLMCallError: If the script is exhausted and ``repeat_last`` is off.
        """
        _validate_prompt(prompt)
        self.calls.append((prompt, system))

        position = len(self.calls) - 1
        if position < len(self._responses):
            return self._responses[position]

        if self._repeat_last:
            return self._responses[-1]

        raise LLMCallError(
            f"scripted provider exhausted after {len(self._responses)} response(s)"
        )


class AnthropicProvider:
    """Calls the Anthropic Messages API.

    The SDK is imported lazily so the rest of the application, and the entire
    test suite, runs whether or not ``anthropic`` is installed.

    Args:
        model: Model id. Defaults to :data:`DEFAULT_ANTHROPIC_MODEL`.
        api_key: Explicit key. When omitted the SDK resolves credentials from
            the environment in its usual order.
        max_tokens: Response cap.

    Raises:
        LLMConfigurationError: If the SDK is not installed.
    """

    def __init__(
        self,
        model: str = DEFAULT_ANTHROPIC_MODEL,
        api_key: str | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        self._model = model
        self._max_tokens = max_tokens
        self._api_key = api_key
        self._client = None

    @property
    def name(self) -> str:
        """Provider and model label."""
        return f"anthropic/{self._model}"

    def _load_client(self):
        """Create and cache the Anthropic client."""
        if self._client is None:
            try:
                import anthropic
            except ImportError as exc:
                raise LLMConfigurationError(
                    "the 'anthropic' package is required for LLM_PROVIDER=anthropic. "
                    "Install it with: pip install anthropic"
                ) from exc

            try:
                self._client = (
                    anthropic.Anthropic(api_key=self._api_key)
                    if self._api_key
                    else anthropic.Anthropic()
                )
            except Exception as exc:
                raise LLMConfigurationError(f"could not create the Anthropic client: {exc}") from exc

        return self._client

    def generate(self, prompt: str, system: str | None = None) -> str:
        """Send one message and return the concatenated text response.

        Args:
            prompt: The user-side prompt.
            system: Optional system prompt.

        Returns:
            The model's text output.

        Raises:
            LLMConfigurationError: If the SDK is missing or credentials are rejected.
            LLMCallError: If the request fails or returns no text.
        """
        _validate_prompt(prompt)
        client = self._load_client()

        request = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            request["system"] = system

        try:
            response = client.messages.create(**request)
        except Exception as exc:
            raise self._translate(exc) from exc

        text = "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        ).strip()

        if not text:
            raise LLMCallError(
                f"the model returned no text (stop_reason={getattr(response, 'stop_reason', None)!r})"
            )

        return text

    @staticmethod
    def _translate(exc: Exception) -> LLMError:
        """Map an SDK exception onto this module's error types."""
        try:
            import anthropic
        except ImportError:  # pragma: no cover - only reachable without the SDK
            return LLMCallError(f"Anthropic request failed: {exc}")

        if isinstance(exc, anthropic.AuthenticationError):
            return LLMConfigurationError(
                "Anthropic rejected the credentials. Set LLM_API_KEY or ANTHROPIC_API_KEY "
                "to a valid key."
            )
        if isinstance(exc, anthropic.PermissionDeniedError):
            return LLMConfigurationError("the API key lacks permission for this model.")
        if isinstance(exc, anthropic.NotFoundError):
            return LLMConfigurationError(f"unknown model or endpoint: {exc}")
        if isinstance(exc, anthropic.RateLimitError):
            return LLMCallError(f"rate limited by Anthropic: {exc}")
        if isinstance(exc, anthropic.APIConnectionError):
            return LLMCallError(f"could not reach the Anthropic API: {exc}")
        return LLMCallError(f"Anthropic request failed: {exc}")


class LangChainProvider:
    """Adapts any LangChain chat model to :class:`LLMProvider`.

    This is where LangChain provides real value in this project: one adapter
    gives access to every chat backend LangChain supports, while the domain
    layer keeps depending only on the small protocol above. LangChain is an
    optional dependency -- nothing else in the application imports it.

    Args:
        chat_model: A LangChain chat model exposing ``invoke``.
        model_name: Label reported by :attr:`name`. Inferred when omitted.

    Raises:
        LLMConfigurationError: If ``chat_model`` has no ``invoke`` method.

    Example:
        >>> from langchain_anthropic import ChatAnthropic       # doctest: +SKIP
        >>> provider = LangChainProvider(ChatAnthropic(model="claude-opus-5"))
    """

    def __init__(self, chat_model: object, model_name: str | None = None) -> None:
        if not hasattr(chat_model, "invoke"):
            raise LLMConfigurationError(
                f"{type(chat_model).__name__} is not a LangChain chat model "
                "(no invoke method)"
            )

        self._chat_model = chat_model
        self._model_name = model_name or getattr(
            chat_model, "model_name", type(chat_model).__name__
        )

    @property
    def name(self) -> str:
        """Provider and model label."""
        return f"langchain/{self._model_name}"

    def generate(self, prompt: str, system: str | None = None) -> str:
        """Invoke the wrapped chat model and return its text content.

        Args:
            prompt: The user-side prompt.
            system: Optional system prompt, sent as a system message.

        Returns:
            The model's text output.

        Raises:
            LLMCallError: If the call fails or yields no text.
        """
        _validate_prompt(prompt)

        messages: list[tuple[str, str]] = []
        if system:
            messages.append(("system", system))
        messages.append(("human", prompt))

        try:
            response = self._chat_model.invoke(messages)
        except Exception as exc:
            raise LLMCallError(f"LangChain chat model failed: {exc}") from exc

        content = getattr(response, "content", response)
        if isinstance(content, list):
            content = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part) for part in content
            )

        text = str(content).strip()
        if not text:
            raise LLMCallError("the LangChain chat model returned no text")

        return text


def get_llm_provider(
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
) -> LLMProvider:
    """Build a provider from arguments, falling back to the environment.

    Args:
        provider: ``fake``, ``anthropic`` or ``langchain``. Defaults to
            ``LLM_PROVIDER``, then to ``fake`` so nothing requires credentials.
        model: Model id. Defaults to ``LLM_MODEL``.
        api_key: API key. Defaults to ``LLM_API_KEY``, then ``ANTHROPIC_API_KEY``.

    Returns:
        A ready provider.

    Raises:
        LLMConfigurationError: If the provider name is unknown, or ``langchain``
            is requested, which cannot be built from configuration alone.
    """
    name = (provider or os.environ.get("LLM_PROVIDER") or DEFAULT_PROVIDER).strip().lower()
    model_id = model or os.environ.get("LLM_MODEL") or DEFAULT_ANTHROPIC_MODEL
    key = api_key or os.environ.get("LLM_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")

    raw_max_tokens = os.environ.get("LLM_MAX_TOKENS")
    try:
        max_tokens = int(raw_max_tokens) if raw_max_tokens else DEFAULT_MAX_TOKENS
    except ValueError as exc:
        raise LLMConfigurationError(
            f"LLM_MAX_TOKENS must be an integer, got {raw_max_tokens!r}"
        ) from exc

    if name == "fake":
        return FakeLLMProvider()

    if name == "anthropic":
        return AnthropicProvider(model=model_id, api_key=key, max_tokens=max_tokens)

    if name == "langchain":
        raise LLMConfigurationError(
            "LLM_PROVIDER=langchain cannot be constructed from configuration alone. "
            "Build the chat model yourself and pass it to LangChainProvider(chat_model)."
        )

    raise LLMConfigurationError(
        f"unknown LLM_PROVIDER {name!r}; expected one of: fake, anthropic, langchain"
    )
