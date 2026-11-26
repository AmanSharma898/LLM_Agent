from __future__ import annotations

from pathlib import Path

from llm_eval_agent.agents import AgentResponse, ToolCall
from llm_eval_agent.baseline import BaselineStore, compare


def test_store_roundtrip(tmp_path: Path) -> None:
    store = BaselineStore(tmp_path / "baselines")
    store.save("suite", "demo:v1", {"a": AgentResponse("hello", [ToolCall("t", {"x": 1})])})
    loaded = store.load("suite")
    assert loaded["a"]["text"] == "hello"
    assert loaded["a"]["tool_calls"] == [{"name": "t", "arguments": {"x": 1}}]


def test_missing_baseline_file_returns_empty(tmp_path: Path) -> None:
    assert BaselineStore(tmp_path).load("nothing") == {}


def test_compare_modes() -> None:
    baseline = {"text": "The result is 5.", "tool_calls": []}
    assert compare(AgentResponse("The result is 5."), baseline, "exact").status == "match"
    assert compare(AgentResponse("The result is 5!"), baseline, "exact").status == "drift"
    assert compare(AgentResponse("The result is 5!"), baseline, "similarity", 0.9).passed
    assert compare(AgentResponse("Result: 5"), baseline, "similarity", 0.9).status == "drift"
    assert compare(AgentResponse("anything"), baseline, "off").passed


def test_missing_baseline_does_not_fail() -> None:
    result = compare(AgentResponse("x"), None)
    assert result.status == "missing"
    assert result.passed


def test_changed_tool_calls_are_drift_even_with_same_text() -> None:
    baseline = {"text": "ok", "tool_calls": [{"name": "calculator", "arguments": {}}]}
    result = compare(AgentResponse("ok", [ToolCall("search")]), baseline)
    assert result.status == "drift"
    assert "tool calls changed" in result.detail
