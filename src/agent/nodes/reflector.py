import json
from src.agent.state import AgentState
from src.agent.llm import llm_node
from src.agent.prompts.reflector import REFLECTOR_PROMPT
from src.tools.base import format_tool_results


def reflector_node(state: AgentState) -> AgentState:
    tool_results = state.get("tool_results", [])
    budget_remaining = state.get("budget_cap_usd", 0.50) - state.get("cost_usd", 0.0)

    formatted_results = []
    for r in tool_results:
        formatted_results.append({
            "tool_name": r.get("tool_name", ""),
            "status": r.get("status", ""),
            "latency_ms": r.get("latency_ms", 0),
            "content": r.get("content", {}),
            "error": r.get("error_message", ""),
        })

    tool_results_str = json.dumps(formatted_results, indent=2, ensure_ascii=False)

    system_prompt = REFLECTOR_PROMPT.format(
        user_query=state.get("user_query", ""),
        plan_text=state.get("plan_text", ""),
        tool_results_formatted=tool_results_str,
        budget_remaining=budget_remaining,
        iteration=state.get("iteration", 0),
        max_iterations=state.get("max_iterations", 8),
    )

    result = llm_node(state, system_prompt, "")

    reflection_text = result["text"]
    thinking = result["thinking"]
    input_tokens = result["input_tokens"]
    output_tokens = result["output_tokens"]

    reflection_status = "sufficient"
    next_action = "answer"

    for line in reflection_text.split("\n"):
        if line.startswith("STATUS:"):
            status_part = line.split("STATUS:", 1)[1].strip().split()[0]
            reflection_status = status_part
        elif line.startswith("NEXT_ACTION:"):
            next_action = line.split("NEXT_ACTION:", 1)[1].strip()

    reasoning_trace = state.get("reasoning_trace", [])
    reasoning_trace.append({
        "step": "reflector",
        "iteration": state.get("iteration", 0),
        "thinking": thinking,
        "reflection_text": reflection_text,
        "reflection_status": reflection_status,
        "next_action": next_action,
        "tokens_used": input_tokens + output_tokens,
    })

    return {
        **state,
        "reflection_text": reflection_text,
        "reflection_status": reflection_status,
        "tokens_input": state.get("tokens_input", 0) + input_tokens,
        "tokens_output": state.get("tokens_output", 0) + output_tokens,
        "reasoning_trace": reasoning_trace,
        "last_thinking": thinking,
    }
