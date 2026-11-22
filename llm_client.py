"""
Provider-agnostic LLM client so agents never touch a vendor API directly.

Agents use three entry points, implemented by every backend:

  * complete(...)            -> plain text
  * complete_structured(...) -> a validated Pydantic object
  * run_tool_loop(...)       -> an agentic tool-use loop with your own tools

Backends (choose with config.LLM_PROVIDER / --provider):

  * "claude"            ClaudeClient - Anthropic SDK. Adaptive thinking + effort,
                        server-side refusal fallback, prompt caching, streaming.
  * "openai_compatible" OpenAICompatibleClient - any server speaking the OpenAI
                        Chat Completions API: Ollama or LM Studio (free, local),
                        vLLM, or hosted open-model APIs such as Groq / OpenRouter.

create_llm_client() builds the configured backend. Shared concerns live here
once: model selection, retries/timeouts, truncation detection, error mapping
and token accounting.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar

import anthropic
import requests
from pydantic import BaseModel, ValidationError

import config

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# Called with (tool_name, tool_input) and returns the tool result as a string.
ToolHandler = Callable[[str, dict[str, Any]], str]


class LLMError(RuntimeError):
    """Raised when the model cannot produce a usable response."""


class LLMRefusalError(LLMError):
    """Raised when the request was declined (stop_reason == 'refusal')."""


@dataclass
class Usage:
    """Running token totals across every call made through one client."""

    input_tokens: int = 0
    cache_read_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, usage: Any) -> None:
        with self._lock:
            self.calls += 1
            self.input_tokens += getattr(usage, "input_tokens", 0) or 0
            self.cache_read_tokens += getattr(usage, "cache_read_input_tokens", 0) or 0
            self.output_tokens += getattr(usage, "output_tokens", 0) or 0

    def __str__(self) -> str:
        return (f"{self.calls} calls, {self.input_tokens:,} input (+{self.cache_read_tokens:,} cached) "
                f"/ {self.output_tokens:,} output tokens")


@dataclass
class ToolLoopResult:
    final_text: str
    turns: int
    tool_calls: int
    stopped_early: bool  # True if max_turns was reached before Claude finished


class LLMClient(ABC):
    """Interface every backend implements. One instance is shared for the whole
    run and must be thread-safe (A2 runs evaluations in parallel)."""

    model: str
    usage: Usage

    @abstractmethod
    def complete(self, system: str, prompt: str, effort: str = "medium",
                 max_tokens: int = 64_000, what: str = "completion") -> str:
        """Return the model's text answer."""

    @abstractmethod
    def complete_structured(self, system: str, prompt: str, schema: type[T], effort: str = "medium",
                            max_tokens: int = 16_000, what: str = "structured completion") -> T:
        """Return a validated instance of `schema`."""

    @abstractmethod
    def run_tool_loop(self, system: str, prompt: str, tools: list[dict[str, Any]],
                      tool_handler: ToolHandler, effort: str = "medium", max_turns: int = 20,
                      max_tokens: int = 32_000, what: str = "agent loop") -> ToolLoopResult:
        """Let the model call tools until it is done.

        `tools` use the Anthropic shape ({name, description, input_schema});
        backends convert as needed. `tool_handler` executes one call and returns
        a string; exceptions it raises are reported back to the model as errors.
        """


def create_llm_client(provider: str | None = None, model: str | None = None) -> LLMClient:
    """Build the backend named by `provider` (default: config.LLM_PROVIDER)."""
    provider = (provider or config.LLM_PROVIDER).lower()
    if provider == "claude":
        return ClaudeClient(model or config.CLAUDE_MODEL)
    if provider in ("openai_compatible", "ollama"):
        return OpenAICompatibleClient(model or config.OPENAI_COMPAT_MODEL)
    raise LLMError(f"Unknown LLM_PROVIDER {provider!r}; use 'claude' or 'openai_compatible'.")


# =============================================================================
# Claude (Anthropic SDK)
# =============================================================================
class ClaudeClient(LLMClient):
    """Claude via the Anthropic SDK."""

    def __init__(self, model: str = config.CLAUDE_MODEL):
        try:
            # Credentials resolve from ANTHROPIC_API_KEY (or ANTHROPIC_AUTH_TOKEN /
            # an `ant auth login` profile). Never hardcode a key here.
            self._client = anthropic.Anthropic(
                max_retries=config.LLM_MAX_RETRIES,
                timeout=config.LLM_TIMEOUT_SECONDS,
            )
        except anthropic.AnthropicError as exc:
            raise LLMError(
                "Could not create the Anthropic client. Set ANTHROPIC_API_KEY "
                "(see README.md -> Setup)."
            ) from exc
        # Fail fast with a clear message instead of a TypeError on the first request.
        if not any(getattr(self._client, a, None) for a in ("api_key", "auth_token", "credentials")):
            raise LLMError(
                "No Claude credentials found. Set ANTHROPIC_API_KEY in your environment or in a "
                ".env file (see README.md -> Setup)."
            )
        self.model = model
        self.usage = Usage()

    # ------------------------------------------------------------------ helpers
    def _base_params(self, system: str, effort: str, max_tokens: int) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system,
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": effort},
        }
        if config.USE_REFUSAL_FALLBACK:
            params["betas"] = [config.FALLBACK_BETA]
            params["fallbacks"] = "default"
        return params

    @staticmethod
    def _text_of(message: Any) -> str:
        return "".join(b.text for b in message.content if b.type == "text").strip()

    def _check(self, message: Any, what: str) -> None:
        self.usage.add(message.usage)
        if message.stop_reason == "refusal":
            details = getattr(message, "stop_details", None)
            raise LLMRefusalError(f"{what}: request declined by the model ({details}).")
        if message.stop_reason == "max_tokens":
            raise LLMError(f"{what}: response hit max_tokens and was truncated.")

    @staticmethod
    def _wrap_api_errors(exc: Exception, what: str) -> LLMError:
        if isinstance(exc, anthropic.AuthenticationError):
            return LLMError(f"{what}: invalid or missing API key (ANTHROPIC_API_KEY).")
        if isinstance(exc, anthropic.NotFoundError):
            return LLMError(f"{what}: model not found ({exc.message}).")
        if isinstance(exc, anthropic.RateLimitError):
            return LLMError(f"{what}: rate limited even after retries; try again later.")
        if isinstance(exc, anthropic.APIStatusError):
            return LLMError(f"{what}: API error {exc.status_code}: {exc.message}")
        if isinstance(exc, anthropic.APIConnectionError):
            return LLMError(f"{what}: network error reaching the Claude API.")
        return LLMError(f"{what}: {exc}")

    # --------------------------------------------------------------- public API
    def complete(
        self,
        system: str,
        prompt: str,
        effort: str = "medium",
        max_tokens: int = 64_000,
        what: str = "completion",
    ) -> str:
        """Return Claude's text answer. Streams under the hood so long outputs don't time out."""
        params = self._base_params(system, effort, max_tokens)
        try:
            with self._client.beta.messages.stream(
                messages=[{"role": "user", "content": prompt}], **params
            ) as stream:
                message = stream.get_final_message()
        except anthropic.APIError as exc:
            raise self._wrap_api_errors(exc, what) from exc
        self._check(message, what)
        return self._text_of(message)

    def complete_structured(
        self,
        system: str,
        prompt: str,
        schema: type[T],
        effort: str = "medium",
        max_tokens: int = 16_000,
        what: str = "structured completion",
    ) -> T:
        """Return a validated instance of `schema` using structured outputs."""
        params = self._base_params(system, effort, max_tokens)
        try:
            message = self._client.beta.messages.parse(
                messages=[{"role": "user", "content": prompt}],
                output_format=schema,
                **params,
            )
        except anthropic.APIError as exc:
            raise self._wrap_api_errors(exc, what) from exc
        self._check(message, what)
        parsed = message.parsed_output
        if parsed is None:
            raise LLMError(f"{what}: no structured output returned.")
        return parsed

    def run_tool_loop(
        self,
        system: str,
        prompt: str,
        tools: list[dict[str, Any]],
        tool_handler: ToolHandler,
        effort: str = "medium",
        max_turns: int = 20,
        max_tokens: int = 32_000,
        what: str = "agent loop",
    ) -> ToolLoopResult:
        """Drive a manual agentic loop: Claude calls tools, we execute them, repeat.

        `tool_handler` executes one tool call and returns a string result; any
        exception it raises is reported back to Claude as an error tool_result
        so the agent can recover instead of crashing the run.
        """
        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        params = self._base_params(system, effort, max_tokens)
        # Each turn resends the growing history; auto-caching bills the repeated
        # prefix at the cache-read rate instead of full input price.
        params["cache_control"] = {"type": "ephemeral"}
        tool_calls = 0
        message = None

        for turn in range(1, max_turns + 1):
            try:
                with self._client.beta.messages.stream(
                    messages=messages, tools=tools, **params
                ) as stream:
                    message = stream.get_final_message()
            except anthropic.APIError as exc:
                raise self._wrap_api_errors(exc, what) from exc
            self._check(message, what)

            # Always append the full content (thinking + tool_use blocks), unmodified.
            messages.append({"role": "assistant", "content": message.content})

            if message.stop_reason == "pause_turn":
                continue
            tool_uses = [b for b in message.content if b.type == "tool_use"]
            if message.stop_reason != "tool_use" or not tool_uses:
                return ToolLoopResult(self._text_of(message), turn, tool_calls, False)

            # Execute every tool call and return all results in ONE user message.
            results = []
            for block in tool_uses:
                tool_calls += 1
                tool_input = block.input if isinstance(block.input, dict) else {}
                log.info("  [tool] %s(%s)", block.name, _short(tool_input))
                try:
                    output = tool_handler(block.name, tool_input)
                    results.append(
                        {"type": "tool_result", "tool_use_id": block.id, "content": output}
                    )
                except Exception as exc:  # noqa: BLE001 - surface any tool failure to the model
                    log.warning("  [tool] %s failed: %s", block.name, exc)
                    results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": f"Tool error: {exc}",
                            "is_error": True,
                        }
                    )
            messages.append({"role": "user", "content": results})

        final = self._text_of(message) if message is not None else ""
        return ToolLoopResult(final, max_turns, tool_calls, True)


# =============================================================================
# Open models via any OpenAI-compatible server (Ollama, LM Studio, vLLM, Groq, ...)
# =============================================================================
class OpenAICompatibleClient(LLMClient):
    """Chat Completions over plain HTTP, so no extra SDK is needed.

    Differences from Claude, handled here:
      * `effort` has no portable equivalent and is ignored.
      * Structured output asks for a JSON schema via `response_format`; if the
        server rejects that, it falls back to schema-in-prompt. The reply is
        always validated with Pydantic and repaired once on failure, because
        small open models don't always follow schemas.
      * `max_tokens` is capped at config.OPENAI_COMPAT_MAX_TOKENS.
    """

    def __init__(self, model: str = config.OPENAI_COMPAT_MODEL):
        self.model = model
        self.usage = Usage()
        self._url = config.OPENAI_COMPAT_BASE_URL.rstrip("/") + "/chat/completions"
        self._session = requests.Session()
        if config.OPENAI_COMPAT_API_KEY:
            self._session.headers["Authorization"] = f"Bearer {config.OPENAI_COMPAT_API_KEY}"
        self._schema_mode_supported = True

    # ------------------------------------------------------------------ helpers
    def _chat(self, messages: list[dict[str, Any]], max_tokens: int, what: str,
              **extra: Any) -> dict[str, Any]:
        body = {"model": self.model, "messages": messages,
                "max_tokens": min(max_tokens, config.OPENAI_COMPAT_MAX_TOKENS), **extra}
        last_error = ""
        for attempt in range(config.LLM_MAX_RETRIES + 1):
            try:
                resp = self._session.post(self._url, json=body,
                                          timeout=config.OPENAI_COMPAT_TIMEOUT_SECONDS)
            except requests.ConnectionError as exc:
                raise LLMError(
                    f"{what}: cannot reach {config.OPENAI_COMPAT_BASE_URL}. Is the server running? "
                    "(for Ollama: `ollama serve`)") from exc
            except requests.Timeout:
                last_error = f"timed out after {config.OPENAI_COMPAT_TIMEOUT_SECONDS}s"
                continue
            if resp.status_code in (429, 500, 502, 503) and attempt < config.LLM_MAX_RETRIES:
                # Free hosted tiers rate-limit hard; honour Retry-After when given.
                retry_after = resp.headers.get("retry-after", "")
                delay = float(retry_after) if retry_after.replace(".", "", 1).isdigit() else 2 ** attempt * 5
                time.sleep(min(delay, 90))
                last_error = f"HTTP {resp.status_code}"
                continue
            if resp.status_code == 401:
                raise LLMError(f"{what}: unauthorized; check OPENAI_COMPAT_API_KEY.")
            if resp.status_code == 404:
                raise LLMError(f"{what}: model {self.model!r} not found on the server "
                               f"(for Ollama: `ollama pull {self.model}`). {resp.text[:200]}")
            if resp.status_code >= 400:
                raise _HTTPError(resp.status_code, f"{what}: HTTP {resp.status_code}: {resp.text[:300]}")
            data = resp.json()
            usage = data.get("usage") or {}
            self.usage.add(_TokenCounts(usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)))
            if not data.get("choices"):
                raise LLMError(f"{what}: server returned no choices: {str(data)[:300]}")
            return data
        raise LLMError(f"{what}: request failed after retries ({last_error}).")

    @staticmethod
    def _content(choice: dict[str, Any]) -> str:
        text = (choice.get("message") or {}).get("content") or ""
        # Some reasoning models inline their chain of thought; keep only the answer.
        return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()

    @staticmethod
    def _check_finish(choice: dict[str, Any], what: str) -> None:
        if choice.get("finish_reason") == "length":
            raise LLMError(f"{what}: response hit the token limit and was truncated "
                           "(raise OPENAI_COMPAT_MAX_TOKENS or the server's context length).")

    # --------------------------------------------------------------- public API
    def complete(self, system: str, prompt: str, effort: str = "medium",
                 max_tokens: int = 64_000, what: str = "completion") -> str:
        data = self._chat([{"role": "system", "content": system},
                           {"role": "user", "content": prompt}], max_tokens, what)
        choice = data["choices"][0]
        self._check_finish(choice, what)
        return self._content(choice)

    def complete_structured(self, system: str, prompt: str, schema: type[T], effort: str = "medium",
                            max_tokens: int = 16_000, what: str = "structured completion") -> T:
        json_schema = schema.model_json_schema()
        system = (f"{system}\n\nRespond with ONLY a JSON object that matches this JSON schema "
                  f"(no prose, no code fences):\n{json.dumps(json_schema)}")
        messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
        response_format = {"type": "json_schema",
                           "json_schema": {"name": schema.__name__, "schema": json_schema, "strict": True}}

        for attempt in range(2):  # one repair round if validation fails
            extra = {"response_format": response_format} if self._schema_mode_supported else {}
            try:
                data = self._chat(messages, max_tokens, what, **extra)
            except _HTTPError as exc:
                if not (extra and exc.status == 400):
                    raise LLMError(str(exc)) from exc
                log.info("server rejected response_format=json_schema; using schema-in-prompt")
                self._schema_mode_supported = False
                data = self._chat(messages, max_tokens, what)
            choice = data["choices"][0]
            self._check_finish(choice, what)
            raw = self._content(choice)
            try:
                return schema.model_validate(_extract_json(raw))
            except (ValueError, ValidationError) as exc:
                if attempt == 1:
                    raise LLMError(f"{what}: model output did not match the schema: {exc}") from exc
                messages += [{"role": "assistant", "content": raw},
                             {"role": "user", "content": f"That was not valid. Error: {exc}\n"
                                                         "Reply again with ONLY the corrected JSON object."}]
        raise AssertionError("unreachable")

    def run_tool_loop(self, system: str, prompt: str, tools: list[dict[str, Any]],
                      tool_handler: ToolHandler, effort: str = "medium", max_turns: int = 20,
                      max_tokens: int = 32_000, what: str = "agent loop") -> ToolLoopResult:
        oa_tools = [{"type": "function", "function": {
            "name": t["name"], "description": t.get("description", ""), "parameters": t["input_schema"]}}
            for t in tools]
        messages: list[dict[str, Any]] = [{"role": "system", "content": system},
                                          {"role": "user", "content": prompt}]
        tool_calls = 0
        text = ""
        for turn in range(1, max_turns + 1):
            try:
                data = self._chat(messages, max_tokens, what, tools=oa_tools)
            except _HTTPError as exc:
                raise LLMError(f"{exc} (does this model support tool calling?)") from exc
            choice = data["choices"][0]
            self._check_finish(choice, what)
            message = choice.get("message") or {}
            text = self._content(choice)
            calls = message.get("tool_calls") or []
            messages.append({"role": "assistant", "content": message.get("content") or "",
                             **({"tool_calls": calls} if calls else {})})
            if not calls:
                return ToolLoopResult(text, turn, tool_calls, False)

            for call in calls:
                tool_calls += 1
                fn = call.get("function") or {}
                name = fn.get("name", "")
                try:
                    args = fn.get("arguments") or {}
                    args = json.loads(args) if isinstance(args, str) else args
                    if not isinstance(args, dict):
                        raise ValueError("tool arguments must be a JSON object")
                    log.info("  [tool] %s(%s)", name, _short(args))
                    output = tool_handler(name, args)
                except Exception as exc:  # noqa: BLE001 - surface any tool failure to the model
                    log.warning("  [tool] %s failed: %s", name, exc)
                    output = f"Tool error: {exc}"
                messages.append({"role": "tool", "tool_call_id": call.get("id", name),
                                 "name": name, "content": output})
        return ToolLoopResult(text, max_turns, tool_calls, True)


class _HTTPError(LLMError):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


@dataclass
class _TokenCounts:
    input_tokens: int
    output_tokens: int


def _extract_json(text: str) -> Any:
    """Parse a JSON object from model output, tolerating code fences or surrounding prose."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            return json.loads(text[start:end + 1])
        raise ValueError(f"no JSON object found in: {text[:200]!r}")


def _short(value: Any, limit: int = 120) -> str:
    text = json.dumps(value, ensure_ascii=False)
    return text if len(text) <= limit else text[: limit - 3] + "..."
