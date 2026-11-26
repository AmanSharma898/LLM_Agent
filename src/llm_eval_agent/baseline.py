"""Baseline ("golden output") storage and comparison.

A baseline is the recorded response of a known-good agent version. Later runs
are compared against it so that behaviour changes are detected automatically,
even when every individual check still passes.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher, unified_diff
from pathlib import Path
from typing import Any

from llm_eval_agent.agents import AgentResponse


@dataclass(frozen=True)
class BaselineComparison:
    status: str  # "match", "drift" or "missing"
    similarity: float | None = None
    detail: str = ""

    @property
    def passed(self) -> bool:
        return self.status != "drift"


def similarity(a: str, b: str) -> float:
    """Character-level similarity ratio between 0.0 and 1.0."""
    return SequenceMatcher(None, a, b).ratio()


class BaselineStore:
    """Stores one JSON file of baseline responses per suite."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)

    def path_for(self, suite_name: str) -> Path:
        return self.directory / f"{suite_name}.json"

    def load(self, suite_name: str) -> dict[str, dict[str, Any]]:
        path = self.path_for(suite_name)
        if not path.exists():
            return {}
        data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return dict(data.get("cases", {}))

    def save(self, suite_name: str, agent_name: str, responses: dict[str, AgentResponse]) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "suite": suite_name,
            "agent": agent_name,
            "cases": {
                case_id: {
                    "text": response.text,
                    "tool_calls": [asdict(call) for call in response.tool_calls],
                }
                for case_id, response in sorted(responses.items())
            },
        }
        path = self.path_for(suite_name)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return path


def compare(
    response: AgentResponse,
    baseline: dict[str, Any] | None,
    mode: str = "similarity",
    threshold: float = 0.9,
) -> BaselineComparison:
    """Compare a response with its recorded baseline."""
    if mode == "off":
        return BaselineComparison("match", detail="baseline disabled")
    if baseline is None:
        return BaselineComparison("missing", detail="no baseline recorded")

    expected_tools = [call["name"] for call in baseline.get("tool_calls", [])]
    actual_tools = [call.name for call in response.tool_calls]
    if expected_tools != actual_tools:
        return BaselineComparison(
            "drift", detail=f"tool calls changed: {expected_tools} -> {actual_tools}"
        )

    expected_text = str(baseline.get("text", ""))
    score = similarity(expected_text, response.text)
    ok = expected_text == response.text if mode == "exact" else score >= threshold
    if ok:
        return BaselineComparison("match", score)
    if "\n" not in expected_text and "\n" not in response.text:
        return BaselineComparison("drift", score, f"{expected_text!r} -> {response.text!r}")
    diff = "\n".join(
        unified_diff(
            expected_text.splitlines(),
            response.text.splitlines(),
            "baseline",
            "current",
            lineterm="",
        )
    )
    return BaselineComparison("drift", score, diff)
