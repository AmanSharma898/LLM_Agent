"""pytest plugin: every case in an ``eval_*.yaml`` file becomes a pytest test.

Usage::

    pytest examples/suites --llm-agent demo:v1 --llm-baselines examples/baselines
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from llm_eval_agent.agents import Agent, build_agent
from llm_eval_agent.baseline import BaselineStore
from llm_eval_agent.cases import Suite, TestCase, load_suite
from llm_eval_agent.runner import evaluate_case


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("llm-eval", "LLM agent non-regression testing")
    group.addoption("--llm-agent", default="demo:v1", help="agent spec, e.g. demo:v1")
    group.addoption("--llm-baselines", default="baselines", help="baseline directory")


def pytest_collect_file(parent: pytest.Collector, file_path: Path) -> pytest.Collector | None:
    if file_path.suffix in {".yaml", ".yml"} and file_path.name.startswith("eval_"):
        return EvalSuiteFile.from_parent(parent, path=file_path)
    return None


class EvalSuiteFile(pytest.File):
    def collect(self) -> Iterator[pytest.Item]:
        suite = load_suite(self.path)
        for case in suite.cases:
            yield EvalCaseItem.from_parent(self, name=case.id, suite=suite, case=case)


class LLMEvalFailure(Exception):
    pass


class EvalCaseItem(pytest.Item):
    def __init__(self, *, suite: Suite, case: TestCase, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.suite = suite
        self.case = case
        for tag in case.tags:
            # YAML tags become pytest markers, so `-m smoke` selects cases.
            self.config.addinivalue_line("markers", f"{tag}: llm-eval tag")
            self.add_marker(tag)

    def _agent(self) -> Agent:
        config = self.config
        cache: dict[str, Agent] = config.stash.setdefault(_AGENTS, {})
        spec = str(config.getoption("--llm-agent"))
        if spec not in cache:
            cache[spec] = build_agent(spec, self.suite.system_prompt)
        return cache[spec]

    def runtest(self) -> None:
        store = BaselineStore(str(self.config.getoption("--llm-baselines")))
        baseline = store.load(self.suite.name).get(self.case.id)
        result = evaluate_case(self._agent(), self.case, baseline)
        if not result.passed:
            raise LLMEvalFailure(result)

    def repr_failure(self, excinfo: pytest.ExceptionInfo[BaseException], style: Any = None) -> str:
        if isinstance(excinfo.value, LLMEvalFailure):
            result = excinfo.value.args[0]
            lines = [
                f"prompt:   {self.case.prompt}",
                f"response: {result.response.text}",
                *(f"- {msg}" for msg in result.failures),
            ]
            return "\n".join(lines)
        return str(super().repr_failure(excinfo))

    def reportinfo(self) -> tuple[Path, int | None, str]:
        return self.path, None, f"llm-eval: {self.suite.name}::{self.case.id}"


_AGENTS = pytest.StashKey[dict[str, Agent]]()
