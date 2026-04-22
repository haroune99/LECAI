from langgraph.graph import StateGraph, END
from src.agent.state import AgentState
from src.agent.nodes.planner import planner_node
from src.agent.nodes.executor import executor_node
from src.agent.nodes.reflector import reflector_node
from src.agent.nodes.answerer import answerer_node


def route_after_reflection(state: AgentState) -> str:
    if state.get("budget_exceeded", False):
        return "answerer"
    if state.get("iteration", 0) >= state.get("max_iterations", 8):
        return "answerer"
    if state.get("retry_count", 0) > 2:
        return "answerer"

    status = state.get("reflection_status", "sufficient")
    next_action = state.get("next_action", "")
    pending = state.get("pending_tool_calls", [])

    if next_action and "answer" in next_action.lower() and not pending:
        return "answerer"

    if next_action and "call:" in next_action and pending:
        return "planner"

    if status in ("insufficient", "tool_failed"):
        return "planner"

    if status == "sufficient" and not pending:
        return "answerer"

    return "answerer"


def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("planner", planner_node)
    builder.add_node("executor", executor_node)
    builder.add_node("reflector", reflector_node)
    builder.add_node("answerer", answerer_node)

    builder.set_entry_point("planner")

    builder.add_edge("planner", "executor")
    builder.add_edge("executor", "reflector")

    builder.add_conditional_edges(
        "reflector",
        route_after_reflection,
        {
            "planner": "planner",
            "executor": "executor",
            "answerer": "answerer",
        },
    )

    builder.add_edge("answerer", END)

    return builder.compile()
