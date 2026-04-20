import json
import uuid
import time
from src.agent.state import AgentState
from src.agent.llm import llm_node
from src.agent.prompts.planner import PLANNER_V1, PLANNER_V2
from src.tools.base import format_tool_results


TOOL_DESCRIPTIONS = {
    "trade_regulations_lookup": "trade_regulations_lookup(query_type, commodity_code, entity_name, category) — UK tariff codes, OFSI sanctions checks, HMRC regulatory requirements",
    "document_intelligence": "document_intelligence(query, filters, top_k) — RAG over LEC document corpus with hybrid search",
    "market_intelligence_search": "market_intelligence_search(query, domain_filter, recency_days) — Real-time web search via Tavily",
    "trade_calculator": "trade_calculator(operation, params) — Landed cost, currency conversion, duty calculation, ROI projection, margin analysis",
    "partnership_profiler": "partnership_profiler(entity_name, entity_type, analysis_type) — Entity profiling, strategic fit, risk assessment",
}


def planner_node(state: AgentState) -> AgentState:
    prompt_version = state.get("prompt_version", "v1")
    budget_remaining = state.get("budget_cap_usd", 0.50) - state.get("cost_usd", 0.0)

    tool_desc_lines = [f"- {name}: {desc}" for name, desc in TOOL_DESCRIPTIONS.items()]
    tool_descriptions = "\n".join(tool_desc_lines)

    planner_prompt = PLANNER_V1 if prompt_version == "v1" else PLANNER_V2

    system_prompt = planner_prompt.format(
        tool_descriptions=tool_descriptions,
        budget_remaining=budget_remaining,
        budget_cap=state.get("budget_cap_usd", 0.50),
        iteration=state.get("iteration", 0),
        max_iterations=state.get("max_iterations", 8),
        user_query=state.get("user_query", ""),
    )

    result = llm_node(state, system_prompt, "")

    plan_text = result["text"]
    input_tokens = result["input_tokens"]
    output_tokens = result["output_tokens"]
    thinking = result["thinking"]

    plan_steps = parse_plan_steps(plan_text)
    pending_calls = build_tool_calls(plan_steps)

    reasoning_trace = state.get("reasoning_trace", [])
    reasoning_trace.append({
        "step": "planner",
        "iteration": state.get("iteration", 0),
        "thinking": thinking,
        "plan_text": plan_text,
        "plan_steps": plan_steps,
        "pending_calls": len(pending_calls),
        "tokens_used": input_tokens + output_tokens,
    })

    return {
        **state,
        "plan_text": plan_text,
        "plan_steps": plan_steps,
        "pending_tool_calls": pending_calls,
        "completed_tool_calls": [],
        "tool_results": [],
        "tokens_input": state.get("tokens_input", 0) + input_tokens,
        "tokens_output": state.get("tokens_output", 0) + output_tokens,
        "reasoning_trace": reasoning_trace,
        "last_thinking": thinking,
    }


def parse_plan_steps(plan_text: str) -> list[dict]:
    steps = []
    current_step = {}
    for line in plan_text.split("\n"):
        line = line.strip()
        if line.startswith("Step ") or (line[0].isdigit() and "." in line[:4]):
            if current_step:
                steps.append(current_step)
            parts = line.split(" using ", 1)
            if len(parts) == 2:
                step_name = parts[0].split(":", 1)[1].strip() if ":" in parts[0] else parts[0]
                tool_name = parts[1].split(" — depends on:")[0].strip()
                depends = "none"
                if " — depends on: " in line:
                    depends_part = line.split(" — depends on: ", 1)[1].split(" —")[0].strip()
                    depends = depends_part
                current_step = {"step": step_name, "tool": tool_name, "depends_on": depends}
            else:
                current_step = {"step": line, "tool": "unknown", "depends_on": "none"}
        elif line.startswith("REASONING:"):
            break
    if current_step:
        steps.append(current_step)
    return steps


def build_tool_calls(plan_steps: list[dict]) -> list[dict]:
    calls = []
    for i, step in enumerate(plan_steps):
        call_id = f"call_{uuid.uuid4().hex[:8]}"
        depends = []
        if step.get("depends_on") and step["depends_on"] != "none":
            depends.append(step["depends_on"])
        calls.append({
            "tool_name": step.get("tool", "unknown"),
            "tool_input": {},
            "call_id": call_id,
            "depends_on": depends,
            "status": "pending",
        })
    return calls
