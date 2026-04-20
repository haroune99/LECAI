from typing import TypedDict, Literal, Annotated
from langgraph.graph import add_messages
from datetime import datetime


class ToolCall(TypedDict):
    tool_name: str
    tool_input: dict
    call_id: str
    depends_on: list[str]
    status: Literal["pending", "running", "complete", "failed", "skipped"]


class ToolResultDict(TypedDict):
    call_id: str
    tool_name: str
    status: Literal["success", "error", "timeout"]
    content: dict
    latency_ms: int
    error_message: str | None
    tokens_used: int | None


class AgentState(TypedDict):
    user_query: str
    session_id: str

    plan_text: str
    plan_steps: list[dict]

    pending_tool_calls: list[ToolCall]
    completed_tool_calls: list[ToolCall]
    tool_results: list[ToolResultDict]
    iteration: int
    max_iterations: int

    reflection_text: str
    reflection_status: Literal["sufficient", "insufficient", "tool_failed", "budget_exceeded"]
    retry_count: int

    tokens_input: int
    tokens_output: int
    cost_usd: float
    budget_cap_usd: float
    budget_exceeded: bool

    final_answer: str
    sources_cited: list[str]
    run_status: Literal["running", "success", "failed", "budget_exceeded", "max_iterations"]

    reasoning_trace: list[dict]

    last_thinking: str | None
    prompt_version: str | None


def default_state() -> AgentState:
    return AgentState(
        user_query="",
        session_id="",
        plan_text="",
        plan_steps=[],
        pending_tool_calls=[],
        completed_tool_calls=[],
        tool_results=[],
        iteration=0,
        max_iterations=8,
        reflection_text="",
        reflection_status="sufficient",
        retry_count=0,
        tokens_input=0,
        tokens_output=0,
        cost_usd=0.0,
        budget_cap_usd=0.50,
        budget_exceeded=False,
        final_answer="",
        sources_cited=[],
        run_status="running",
        reasoning_trace=[],
        last_thinking=None,
        prompt_version=None,
    )
