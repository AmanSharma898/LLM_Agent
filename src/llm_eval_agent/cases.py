"""Loading test cases from YAML suite files."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from llm_eval_agent.checks import CHECKS


@dataclass
class TestCase:
    """One prompt plus the expectations its answer must satisfy."""

    __test__ = False  # keep pytest from collecting this class

    id: str
    prompt: str
    checks: list[dict[str, Any]] = field(default_factory=list)
    baseline: str = "similarity"  # "exact", "similarity" or "off"
    threshold: float = 0.9
    tags: list[str] = field(default_factory=list)


@dataclass
class Suite:
    """A named collection of test cases loaded from one YAML file."""

    name: str
    cases: list[TestCase]
    path: Path | None = None
    system_prompt: str | None = None


class SuiteError(ValueError):
    """Raised when a suite file is malformed."""


def _parse_case(raw: dict[str, Any], defaults: dict[str, Any], source: str) -> TestCase:
    if "id" not in raw or "prompt" not in raw:
        raise SuiteError(f"{source}: every case needs an 'id' and a 'prompt'")
    merged = {**defaults, **raw}
    checks = merged.get("checks", [])
    for check in checks:
        if not isinstance(check, dict) or len(check) != 1:
            raise SuiteError(f"{source}: case {raw['id']!r} has a malformed check: {check!r}")
        (name,) = check
        if name not in CHECKS:
            known = ", ".join(sorted(CHECKS))
            raise SuiteError(f"{source}: unknown check {name!r} (known: {known})")
    baseline = merged.get("baseline", "similarity")
    if baseline is False:  # YAML 1.1 reads a bare `off` as boolean false
        baseline = "off"
    if baseline not in {"exact", "similarity", "off"}:
        raise SuiteError(f"{source}: case {raw['id']!r} has invalid baseline mode {baseline!r}")
    return TestCase(
        id=str(merged["id"]),
        prompt=str(merged["prompt"]),
        checks=list(checks),
        baseline=baseline,
        threshold=float(merged.get("threshold", 0.9)),
        tags=list(merged.get("tags", [])),
    )


def load_suite(path: str | Path) -> Suite:
    """Load and validate a YAML suite file."""
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict) or "cases" not in data:
        raise SuiteError(f"{path}: a suite file needs a top-level 'cases' list")
    defaults = data.get("defaults", {}) or {}
    cases = [_parse_case(raw, defaults, str(path)) for raw in data["cases"]]
    ids = [case.id for case in cases]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise SuiteError(f"{path}: duplicate case ids: {', '.join(duplicates)}")
    return Suite(
        name=str(data.get("name", path.stem)),
        cases=cases,
        path=path,
        system_prompt=data.get("system_prompt"),
    )
