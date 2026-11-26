from __future__ import annotations

from pathlib import Path

import pytest

pytest_plugins = ["pytester"]

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def example_suite_path() -> Path:
    return ROOT / "examples" / "suites" / "eval_assistant.yaml"


@pytest.fixture
def example_baselines() -> Path:
    return ROOT / "examples" / "baselines"
