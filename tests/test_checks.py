from __future__ import annotations

from llm_eval_agent.agents import AgentResponse, ToolCall
from llm_eval_agent.checks import CHECKS, run_checks


def response(text: str, tools: list[str] | None = None, latency: float = 0.0) -> AgentResponse:
    return AgentResponse(text, [ToolCall(name) for name in tools or []], latency)


def test_contains_is_case_insensitive_and_accepts_lists() -> None:
    assert CHECKS["contains"](response("Hello World"), ["hello", "WORLD"]).passed
    result = CHECKS["contains"](response("Hello"), ["hello", "bye"])
    assert not result.passed
    assert "bye" in result.detail


def test_not_contains() -> None:
    assert CHECKS["not_contains"](response("all good"), "error").passed
    assert not CHECKS["not_contains"](response("an Error happened"), "error").passed


def test_equals_ignores_surrounding_whitespace() -> None:
    assert CHECKS["equals"](response("  42\n"), 42).passed


def test_regex() -> None:
    assert CHECKS["regex"](response("6.21 miles"), r"\d+\.\d{2} miles").passed
    assert not CHECKS["regex"](response("6 miles"), r"\d+\.\d{2} miles").passed


def test_json_schema() -> None:
    schema = {"type": "object", "required": ["id"], "properties": {"id": {"type": "integer"}}}
    assert CHECKS["json_schema"](response('{"id": 1}'), schema).passed
    assert not CHECKS["json_schema"](response('{"id": "1"}'), schema).passed
    invalid = CHECKS["json_schema"](response("not json"), schema)
    assert not invalid.passed
    assert "invalid JSON" in invalid.detail


def test_tool_checks() -> None:
    r = response("ok", ["calculator"])
    assert CHECKS["tool_called"](r, "calculator").passed
    assert not CHECKS["tool_called"](r, ["calculator", "search"]).passed
    assert not CHECKS["no_tool_called"](r, True).passed
    assert CHECKS["no_tool_called"](response("ok"), True).passed


def test_limits() -> None:
    assert CHECKS["max_chars"](response("abc"), 3).passed
    assert not CHECKS["max_chars"](response("abcd"), 3).passed
    assert not CHECKS["max_latency_s"](response("x", latency=2.0), 1).passed


def test_run_checks_applies_all_checks_in_order() -> None:
    results = run_checks(
        response("The result is 5.", ["calculator"]),
        [
            {"contains": "5"},
            {"tool_called": "calculator"},
            {"max_chars": 5},
        ],
    )
    assert [r.name for r in results] == ["contains", "tool_called", "max_chars"]
    assert [r.passed for r in results] == [True, True, False]
