from __future__ import annotations

import pytest

from llm_eval_agent.agents import Agent, DemoAgent, build_agent, safe_eval


def test_safe_eval_handles_basic_arithmetic() -> None:
    assert safe_eval("2 + 3 * 4") == 14
    assert safe_eval("-(2 - 5) ** 2") == -9


@pytest.mark.parametrize("expression", ["__import__('os')", "a + 1", "[1, 2]"])
def test_safe_eval_rejects_non_arithmetic(expression: str) -> None:
    with pytest.raises(ValueError):
        safe_eval(expression)


def test_demo_agent_uses_calculator_tool() -> None:
    response = DemoAgent("v1").run("What is 6 * 7?")
    assert response.text == "The result is 42."
    assert [call.name for call in response.tool_calls] == ["calculator"]
    assert response.latency_s >= 0


def test_demo_agent_v2_changes_behaviour() -> None:
    assert DemoAgent("v2").run("Calculate 10 / 4").text == "Result: 2"


def test_demo_agent_is_deterministic() -> None:
    agent = DemoAgent("v1")
    assert agent.run("convert 5 km to miles").text == agent.run("convert 5 km to miles").text


def test_build_agent_returns_demo_agent() -> None:
    agent = build_agent("demo:v2")
    assert isinstance(agent, Agent)
    assert agent.name == "demo:v2"


@pytest.mark.parametrize("spec", ["unknown:x", "openai", "demo:v9"])
def test_build_agent_rejects_bad_specs(spec: str) -> None:
    with pytest.raises(ValueError):
        build_agent(spec)
