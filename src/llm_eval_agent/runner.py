"""Run a suite against an agent and collect the results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from llm_eval_agent.agents import Agent, AgentResponse
from llm_eval_agent.baseline import BaselineComparison, BaselineStore, compare
from llm_eval_agent.cases import Suite, TestCase
from llm_eval_agent.checks import CheckResult, run_checks


@dataclass
class CaseResult:
    case: TestCase
    response: AgentResponse
    checks: list[CheckResult]
    baseline: BaselineComparison

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks) and self.baseline.passed

    @property
    def failures(self) -> list[str]:
        messages = [f"{c.name}: {c.detail}" for c in self.checks if not c.passed]
        if not self.baseline.passed:
            score = self.baseline.similarity
            shown = f" (similarity {score:.2f})" if score is not None else ""
            messages.append(f"baseline drift{shown}: {self.baseline.detail}")
        return messages


@dataclass
class SuiteResult:
    suite: Suite
    agent_name: str
    results: list[CaseResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results)

    @property
    def counts(self) -> dict[str, int]:
        passed = sum(r.passed for r in self.results)
        drift = sum(r.baseline.status == "drift" for r in self.results)
        missing = sum(r.baseline.status == "missing" for r in self.results)
        return {
            "total": len(self.results),
            "passed": passed,
            "failed": len(self.results) - passed,
            "drift": drift,
            "missing_baseline": missing,
        }


def evaluate_case(agent: Agent, case: TestCase, baseline: dict[str, Any] | None) -> CaseResult:
    response = agent.run(case.prompt)
    return CaseResult(
        case=case,
        response=response,
        checks=run_checks(response, case.checks),
        baseline=compare(response, baseline, case.baseline, case.threshold),
    )


def run_suite(
    agent: Agent,
    suite: Suite,
    store: BaselineStore | None = None,
    tags: list[str] | None = None,
) -> SuiteResult:
    """Run every case (optionally filtered by tag) and compare with baselines."""
    baselines = store.load(suite.name) if store else {}
    result = SuiteResult(suite=suite, agent_name=agent.name)
    for case in suite.cases:
        if tags and not set(tags) & set(case.tags):
            continue
        result.results.append(evaluate_case(agent, case, baselines.get(case.id)))
    return result


def record_baselines(agent: Agent, suite: Suite, store: BaselineStore) -> dict[str, AgentResponse]:
    """Run the suite and store every response as the new baseline."""
    responses = {case.id: agent.run(case.prompt) for case in suite.cases}
    store.save(suite.name, agent.name, responses)
    return responses
