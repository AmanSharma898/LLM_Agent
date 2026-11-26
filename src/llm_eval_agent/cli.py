"""Command line interface: ``llm-eval run`` and ``llm-eval record``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from llm_eval_agent import __version__
from llm_eval_agent.agents import build_agent
from llm_eval_agent.baseline import BaselineStore
from llm_eval_agent.cases import load_suite
from llm_eval_agent.report import to_console, to_json, to_markdown
from llm_eval_agent.runner import record_baselines, run_suite


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-eval", description="Non-regression testing for LLM agents."
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("suite", type=Path, help="YAML suite file")
        p.add_argument("--agent", default="demo:v1", help="e.g. demo:v1, openai:<model>")
        p.add_argument("--baselines", type=Path, default=Path("baselines"), help="baseline dir")

    run = sub.add_parser("run", help="run a suite and compare against baselines")
    common(run)
    run.add_argument("--tag", action="append", help="only run cases with this tag")
    run.add_argument("--json", type=Path, help="write a JSON report to this path")
    run.add_argument("--markdown", type=Path, help="write a Markdown report to this path")
    run.add_argument(
        "--fail-on-missing", action="store_true", help="treat missing baselines as failures"
    )

    record = sub.add_parser("record", help="record current responses as the new baseline")
    common(record)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    suite = load_suite(args.suite)
    agent = build_agent(args.agent, suite.system_prompt)
    store = BaselineStore(args.baselines)

    if args.command == "record":
        responses = record_baselines(agent, suite, store)
        print(f"Recorded {len(responses)} baselines to {store.path_for(suite.name)}")
        return 0

    result = run_suite(agent, suite, store, args.tag)
    print(to_console(result))
    for path, render in ((args.json, to_json), (args.markdown, to_markdown)):
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(render(result), encoding="utf-8")
    failed = not result.passed or (args.fail_on_missing and result.counts["missing_baseline"])
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
