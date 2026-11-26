from __future__ import annotations

from pathlib import Path

import pytest

from llm_eval_agent.cases import SuiteError, load_suite


def write(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "eval_suite.yaml"
    path.write_text(content, encoding="utf-8")
    return path


def test_example_suite_loads(example_suite_path: Path) -> None:
    suite = load_suite(example_suite_path)
    assert suite.name == "assistant"
    assert len(suite.cases) == 7
    assert suite.system_prompt


def test_defaults_are_merged_into_cases(tmp_path: Path) -> None:
    suite = load_suite(
        write(
            tmp_path,
            """
defaults: {baseline: exact, threshold: 0.5}
cases:
  - {id: a, prompt: hi}
  - {id: b, prompt: hi, baseline: off}
""",
        )
    )
    assert suite.name == "eval_suite"
    assert (suite.cases[0].baseline, suite.cases[0].threshold) == ("exact", 0.5)
    assert suite.cases[1].baseline == "off"


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("name: x", "top-level 'cases'"),
        ("cases: [{prompt: hi}]", "needs an 'id'"),
        ("cases: [{id: a, prompt: hi, checks: [{nope: 1}]}]", "unknown check"),
        ("cases: [{id: a, prompt: hi, checks: [{contains: a, regex: b}]}]", "malformed"),
        ("cases: [{id: a, prompt: hi, baseline: fuzzy}]", "invalid baseline"),
        ("cases: [{id: a, prompt: hi}, {id: a, prompt: yo}]", "duplicate"),
    ],
)
def test_invalid_suites_are_rejected(tmp_path: Path, content: str, message: str) -> None:
    with pytest.raises(SuiteError, match=message):
        load_suite(write(tmp_path, content))
