import json
from src.agent.state import AgentState
from src.agent.llm import llm_node
from src.agent.prompts.answerer import ANSWERER_PROMPT


def answerer_node(state: AgentState) -> AgentState:
    tool_results = state.get("tool_results", [])

    formatted_results = []
    for r in tool_results:
        formatted_results.append({
            "tool_name": r.get("tool_name", ""),
            "status": r.get("status", ""),
            "latency_ms": r.get("latency_ms", 0),
            "content": r.get("content", {}),
        })

    tool_results_str = json.dumps(formatted_results, indent=2, ensure_ascii=False)

    system_prompt = ANSWERER_PROMPT.format(
        user_query=state.get("user_query", ""),
        tool_results_formatted=tool_results_str,
    )

    result = llm_node(state, system_prompt, "")

    final_answer = result["text"]
    thinking = result["thinking"]
    input_tokens = result["input_tokens"]
    output_tokens = result["output_tokens"]

    sources = [r.get("tool_name", "") for r in tool_results if r.get("status") == "success"]

    reasoning_trace = state.get("reasoning_trace", [])
    reasoning_trace.append({
        "step": "answerer",
        "iteration": state.get("iteration", 0),
        "thinking": thinking,
        "final_answer": final_answer,
        "tokens_used": input_tokens + output_tokens,
    })

    return {
        **state,
        "final_answer": final_answer,
        "sources_cited": sources,
        "run_status": "success",
        "tokens_input": state.get("tokens_input", 0) + input_tokens,
        "tokens_output": state.get("tokens_output", 0) + output_tokens,
        "reasoning_trace": reasoning_trace,
        "last_thinking": thinking,
    }
