"""llm-eval-agent: non-regression testing for LLM agents."""

from llm_eval_agent.agents import Agent, AgentResponse, ToolCall, build_agent
from llm_eval_agent.baseline import BaselineStore
from llm_eval_agent.cases import TestCase, load_suite
from llm_eval_agent.runner import CaseResult, SuiteResult, run_suite

__all__ = [
    "Agent",
    "AgentResponse",
    "BaselineStore",
    "CaseResult",
    "SuiteResult",
    "TestCase",
    "ToolCall",
    "build_agent",
    "load_suite",
    "run_suite",
]

__version__ = "0.1.0"
