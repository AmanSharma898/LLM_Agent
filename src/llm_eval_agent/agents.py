"""Agent interface, a deterministic demo agent and optional API adapters.

Every agent receives a prompt and returns an :class:`AgentResponse` containing the
final text plus the tool calls it made. The framework only depends on this small
interface, so any LLM agent can be tested by wrapping it in an ``Agent``.
"""

from __future__ import annotations

import ast
import json
import operator
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class ToolCall:
    """A single tool invocation made by the agent."""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResponse:
    """What an agent returned for one prompt."""

    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    latency_s: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Agent(Protocol):
    """Minimal interface every agent under test must implement."""

    name: str

    def run(self, prompt: str) -> AgentResponse:
        """Answer a single prompt."""
        ...


def timed(fn: Callable[[], AgentResponse]) -> AgentResponse:
    """Run ``fn`` and fill in the response latency."""
    start = time.perf_counter()
    response = fn()
    response.latency_s = time.perf_counter() - start
    return response


# --------------------------------------------------------------------------- #
# Deterministic demo agent
# --------------------------------------------------------------------------- #

_OPERATORS: dict[type[ast.AST], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}

_ORDERS: dict[int, dict[str, Any]] = {
    42: {"order_id": 42, "status": "shipped", "items": 3},
    7: {"order_id": 7, "status": "processing", "items": 1},
}

_KM_TO_MILES = 0.621371


def safe_eval(expression: str) -> float:
    """Evaluate a basic arithmetic expression without using ``eval``."""

    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
            return float(_OPERATORS[type(node.op)](_eval(node.left), _eval(node.right)))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -_eval(node.operand)
        raise ValueError(f"Unsupported expression: {expression!r}")

    return _eval(ast.parse(expression, mode="eval"))


def _format_number(value: float) -> str:
    return str(int(value)) if value == int(value) else f"{value:.2f}"


class DemoAgent:
    """A small rule-based "assistant" with three tools.

    It behaves like a tool-using LLM agent but is fully deterministic, so the
    framework, its tests and CI run without network access or API keys.

    ``version`` simulates a model or prompt update: ``v2`` deliberately changes
    some behaviour (answer wording and rounding) so that the non-regression
    workflow can be demonstrated end to end.
    """

    def __init__(self, version: str = "v1") -> None:
        if version not in {"v1", "v2"}:
            raise ValueError(f"Unknown demo agent version: {version!r}")
        self.version = version
        self.name = f"demo:{version}"

    def run(self, prompt: str) -> AgentResponse:
        return timed(lambda: self._answer(prompt))

    def _answer(self, prompt: str) -> AgentResponse:
        text = prompt.strip().lower()

        if match := re.search(r"convert\s+([\d.]+)\s*km\s+to\s+miles", text):
            km = float(match.group(1))
            miles = km * _KM_TO_MILES
            call = ToolCall("unit_converter", {"value": km, "from": "km", "to": "miles"})
            shown = f"{miles:.2f}" if self.version == "v1" else f"{miles:.0f}"
            return AgentResponse(f"{_format_number(km)} km is {shown} miles.", [call])

        if match := re.search(r"order\s+#?(\d+)", text):
            order_id = int(match.group(1))
            call = ToolCall("order_lookup", {"order_id": order_id})
            order = _ORDERS.get(order_id)
            if order is None:
                return AgentResponse(json.dumps({"error": "order not found"}), [call])
            return AgentResponse(json.dumps(order), [call])

        if match := re.search(r"(?:what is|calculate)\s+([\d\s.+\-*/()]+)", text):
            expression = match.group(1).strip()
            call = ToolCall("calculator", {"expression": expression})
            try:
                result = safe_eval(expression)
            except (ValueError, SyntaxError, ZeroDivisionError):
                return AgentResponse("Sorry, I could not calculate that.", [call])
            if self.version == "v1":
                return AgentResponse(f"The result is {_format_number(result)}.", [call])
            return AgentResponse(f"Result: {round(result)}", [call])

        return AgentResponse(
            "Sorry, I can only help with calculations, unit conversions and order lookups."
        )


# --------------------------------------------------------------------------- #
# Optional adapters for real LLM APIs
# --------------------------------------------------------------------------- #


class OpenAIChatAgent:
    """Adapter for any OpenAI-compatible chat completions endpoint.

    Requires ``pip install llm-eval-agent[openai]`` and ``OPENAI_API_KEY``.
    """

    def __init__(self, model: str, system_prompt: str | None = None) -> None:
        from openai import OpenAI

        self._client = OpenAI()
        self.model = model
        self.system_prompt = system_prompt
        self.name = f"openai:{model}"

    def run(self, prompt: str) -> AgentResponse:
        def call() -> AgentResponse:
            messages: list[dict[str, str]] = []
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})
            messages.append({"role": "user", "content": prompt})
            completion = self._client.chat.completions.create(
                model=self.model, messages=messages, temperature=0
            )
            message = completion.choices[0].message
            tool_calls = [
                ToolCall(tc.function.name, json.loads(tc.function.arguments or "{}"))
                for tc in (message.tool_calls or [])
            ]
            return AgentResponse(message.content or "", tool_calls, metadata={"model": self.model})

        return timed(call)


class AnthropicAgent:
    """Adapter for the Anthropic Messages API.

    Requires ``pip install llm-eval-agent[anthropic]`` and ``ANTHROPIC_API_KEY``.
    """

    def __init__(self, model: str, system_prompt: str | None = None) -> None:
        import anthropic

        self._client = anthropic.Anthropic()
        self.model = model
        self.system_prompt = system_prompt
        self.name = f"anthropic:{model}"

    def run(self, prompt: str) -> AgentResponse:
        def call() -> AgentResponse:
            kwargs: dict[str, Any] = {
                "model": self.model,
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": prompt}],
            }
            if self.system_prompt:
                kwargs["system"] = self.system_prompt
            message = self._client.messages.create(**kwargs)
            text = "".join(b.text for b in message.content if b.type == "text")
            tool_calls = [
                ToolCall(b.name, dict(b.input)) for b in message.content if b.type == "tool_use"
            ]
            return AgentResponse(text, tool_calls, metadata={"model": self.model})

        return timed(call)


def build_agent(spec: str, system_prompt: str | None = None) -> Agent:
    """Create an agent from a ``provider:option`` string.

    Examples: ``demo:v1``, ``openai:<model>``, ``anthropic:<model>``.
    """
    provider, _, option = spec.partition(":")
    if provider == "demo":
        return DemoAgent(option or "v1")
    if not option:
        raise ValueError(f"Agent spec {spec!r} needs a model name, e.g. '{provider}:<model>'")
    if provider == "openai":
        return OpenAIChatAgent(option, system_prompt)
    if provider == "anthropic":
        return AnthropicAgent(option, system_prompt)
    raise ValueError(f"Unknown agent provider {provider!r} (expected demo, openai or anthropic)")
