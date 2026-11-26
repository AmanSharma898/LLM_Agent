from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from llm_eval_agent.agents import DemoAgent
from llm_eval_agent.baseline import BaselineStore
from llm_eval_agent.cases import load_suite
from llm_eval_agent.cli import main
from llm_eval_agent.report import to_markdown
from llm_eval_agent.runner import record_baselines, run_suite


def test_v1_passes_against_committed_baselines(
    example_suite_path: Path, example_baselines: Path
) -> None:
    result = run_suite(
        DemoAgent("v1"), load_suite(example_suite_path), BaselineStore(example_baselines)
    )
    assert result.passed, [r.failures for r in result.results]
    assert result.counts["drift"] == 0


def test_v2_regression_is_detected(example_suite_path: Path, example_baselines: Path) -> None:
    result = run_suite(
        DemoAgent("v2"), load_suite(example_suite_path), BaselineStore(example_baselines)
    )
    failed = {r.case.id for r in result.results if not r.passed}
    assert failed == {"simple_addition", "division_keeps_decimals", "km_to_miles"}
    # simple_addition passes all explicit checks: only the baseline catches the change.
    simple = next(r for r in result.results if r.case.id == "simple_addition")
    assert all(c.passed for c in simple.checks)
    assert simple.baseline.status == "drift"
    assert "FAIL" in to_markdown(result)


def test_tag_filter(example_suite_path: Path) -> None:
    result = run_suite(DemoAgent("v1"), load_suite(example_suite_path), tags=["smoke"])
    assert {r.case.id for r in result.results} == {
        "simple_addition",
        "order_lookup_returns_json",
        "out_of_scope_request",
    }


def test_record_then_run(tmp_path: Path, example_suite_path: Path) -> None:
    suite = load_suite(example_suite_path)
    store = BaselineStore(tmp_path)
    record_baselines(DemoAgent("v2"), suite, store)
    assert run_suite(DemoAgent("v2"), suite, store).counts["drift"] == 0


def test_cli_run_writes_reports_and_sets_exit_code(
    tmp_path: Path,
    example_suite_path: Path,
    example_baselines: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    base = ["run", str(example_suite_path), "--baselines", str(example_baselines)]
    assert main([*base, "--json", str(tmp_path / "r.json")]) == 0
    assert json.loads((tmp_path / "r.json").read_text())["summary"]["passed"] == 7

    assert main([*base, "--agent", "demo:v2", "--markdown", str(tmp_path / "r.md")]) == 1
    assert "baseline drift" in (tmp_path / "r.md").read_text()
    assert "4/7 passed, 2 drift" in capsys.readouterr().out


def test_cli_record_and_fail_on_missing(tmp_path: Path, example_suite_path: Path) -> None:
    suite_copy = tmp_path / "eval_assistant.yaml"
    shutil.copy(example_suite_path, suite_copy)
    baselines = tmp_path / "baselines"
    run = ["run", str(suite_copy), "--baselines", str(baselines)]
    assert main(run) == 0
    assert main([*run, "--fail-on-missing"]) == 1
    assert main(["record", str(suite_copy), "--baselines", str(baselines)]) == 0
    assert main([*run, "--fail-on-missing"]) == 0
