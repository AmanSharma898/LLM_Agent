"""JSON and Markdown reports for a suite run."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from llm_eval_agent.runner import SuiteResult


def to_dict(result: SuiteResult) -> dict[str, Any]:
    return {
        "suite": result.suite.name,
        "agent": result.agent_name,
        "passed": result.passed,
        "summary": result.counts,
        "cases": [
            {
                "id": r.case.id,
                "passed": r.passed,
                "prompt": r.case.prompt,
                "response": r.response.text,
                "tool_calls": [asdict(c) for c in r.response.tool_calls],
                "latency_s": round(r.response.latency_s, 4),
                "checks": [asdict(c) for c in r.checks],
                "baseline": asdict(r.baseline),
            }
            for r in result.results
        ],
    }


def to_json(result: SuiteResult) -> str:
    return json.dumps(to_dict(result), indent=2, ensure_ascii=False)


def to_markdown(result: SuiteResult) -> str:
    counts = result.counts
    status = "PASSED" if result.passed else "FAILED"
    lines = [
        f"# Evaluation report: {result.suite.name}",
        "",
        f"- **Agent:** `{result.agent_name}`",
        f"- **Status:** {status}",
        f"- **Cases:** {counts['passed']}/{counts['total']} passed, "
        f"{counts['drift']} baseline drift, {counts['missing_baseline']} without baseline",
        "",
        "| Case | Result | Baseline | Similarity | Details |",
        "|------|--------|----------|------------|---------|",
    ]
    for r in result.results:
        score = "" if r.baseline.similarity is None else f"{r.baseline.similarity:.2f}"
        details = "; ".join(r.failures).replace("\n", " ").replace("|", "\\|")
        lines.append(
            f"| `{r.case.id}` | {'pass' if r.passed else 'FAIL'} | {r.baseline.status} "
            f"| {score} | {details} |"
        )
    return "\n".join(lines) + "\n"


def to_console(result: SuiteResult) -> str:
    lines = [f"Suite '{result.suite.name}' with agent '{result.agent_name}'"]
    for r in result.results:
        mark = "PASS" if r.passed else "FAIL"
        lines.append(f"  [{mark}] {r.case.id}")
        lines.extend(f"         - {msg.splitlines()[0]}" for msg in r.failures)
    c = result.counts
    lines.append(
        f"{c['passed']}/{c['total']} passed, {c['drift']} drift, "
        f"{c['missing_baseline']} missing baseline"
    )
    return "\n".join(lines)
