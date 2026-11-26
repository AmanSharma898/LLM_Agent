"""Assertion library applied to an agent response.

Each check takes the response and a single argument from the YAML file and
returns a :class:`CheckResult`. New checks are added with :func:`register`.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import jsonschema

if TYPE_CHECKING:
    from llm_eval_agent.agents import AgentResponse


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    detail: str = ""


CheckFn = Callable[["AgentResponse", Any], CheckResult]
CHECKS: dict[str, CheckFn] = {}


def register(name: str) -> Callable[[CheckFn], CheckFn]:
    """Decorator that makes a check usable from YAML under ``name``."""

    def decorator(fn: CheckFn) -> CheckFn:
        CHECKS[name] = fn
        return fn

    return decorator


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else [value]


@register("contains")
def contains(response: AgentResponse, expected: Any) -> CheckResult:
    missing = [s for s in _as_list(expected) if str(s).lower() not in response.text.lower()]
    return CheckResult("contains", not missing, f"missing: {missing}" if missing else "")


@register("not_contains")
def not_contains(response: AgentResponse, forbidden: Any) -> CheckResult:
    found = [s for s in _as_list(forbidden) if str(s).lower() in response.text.lower()]
    return CheckResult("not_contains", not found, f"found: {found}" if found else "")


@register("equals")
def equals(response: AgentResponse, expected: Any) -> CheckResult:
    ok = response.text.strip() == str(expected).strip()
    return CheckResult("equals", ok, "" if ok else f"got {response.text!r}")


@register("regex")
def regex(response: AgentResponse, pattern: Any) -> CheckResult:
    ok = re.search(str(pattern), response.text) is not None
    return CheckResult("regex", ok, "" if ok else f"no match for /{pattern}/")


@register("json_schema")
def json_schema(response: AgentResponse, schema: Any) -> CheckResult:
    try:
        payload = json.loads(response.text)
    except json.JSONDecodeError as exc:
        return CheckResult("json_schema", False, f"invalid JSON: {exc.msg}")
    try:
        jsonschema.validate(payload, schema)
    except jsonschema.ValidationError as exc:
        return CheckResult("json_schema", False, exc.message)
    return CheckResult("json_schema", True)


@register("tool_called")
def tool_called(response: AgentResponse, expected: Any) -> CheckResult:
    called = {call.name for call in response.tool_calls}
    missing = [name for name in _as_list(expected) if name not in called]
    return CheckResult(
        "tool_called",
        not missing,
        f"missing tools: {missing}, called: {sorted(called)}" if missing else "",
    )


@register("no_tool_called")
def no_tool_called(response: AgentResponse, _: Any) -> CheckResult:
    names = [call.name for call in response.tool_calls]
    return CheckResult("no_tool_called", not names, f"called: {names}" if names else "")


@register("max_chars")
def max_chars(response: AgentResponse, limit: Any) -> CheckResult:
    ok = len(response.text) <= int(limit)
    return CheckResult("max_chars", ok, "" if ok else f"{len(response.text)} > {limit}")


@register("max_latency_s")
def max_latency_s(response: AgentResponse, limit: Any) -> CheckResult:
    ok = response.latency_s <= float(limit)
    return CheckResult("max_latency_s", ok, "" if ok else f"{response.latency_s:.3f}s > {limit}s")


def run_checks(response: AgentResponse, checks: list[dict[str, Any]]) -> list[CheckResult]:
    """Apply every check of a test case to a response."""
    results = []
    for check in checks:
        ((name, argument),) = check.items()
        results.append(CHECKS[name](response, argument))
    return results
