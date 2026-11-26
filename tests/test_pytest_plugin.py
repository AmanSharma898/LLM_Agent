from __future__ import annotations

import shutil
from pathlib import Path

import pytest


@pytest.fixture
def project(pytester: pytest.Pytester, example_suite_path: Path, example_baselines: Path) -> Path:
    shutil.copy(example_suite_path, pytester.path / "eval_assistant.yaml")
    shutil.copytree(example_baselines, pytester.path / "baselines")
    return pytester.path


def test_yaml_cases_are_collected_and_pass(pytester: pytest.Pytester, project: Path) -> None:
    result = pytester.runpytest("--llm-agent", "demo:v1")
    result.assert_outcomes(passed=7)


def test_regression_fails_with_readable_output(pytester: pytest.Pytester, project: Path) -> None:
    result = pytester.runpytest("--llm-agent", "demo:v2")
    result.assert_outcomes(passed=4, failed=3)
    result.stdout.fnmatch_lines(["*'The result is 5.' -> 'Result: 5'*"])


def test_tags_become_markers(pytester: pytest.Pytester, project: Path) -> None:
    result = pytester.runpytest("-m", "smoke")
    result.assert_outcomes(passed=3, deselected=4)
