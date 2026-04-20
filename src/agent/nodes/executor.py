import asyncio
import time
from src.agent.state import AgentState
from src.agent.llm import llm_node


TOOL_REGISTRY = {}


def register_tools():
    from src.tools.trade_regulations import trade_regulations_lookup
    from src.tools.document_intelligence import document_intelligence
    from src.tools.market_search import market_intelligence_search
    from src.tools.trade_calculator import trade_calculator
    from src.tools.partnership_profiler import partnership_profiler

    TOOL_REGISTRY["trade_regulations_lookup"] = trade_regulations_lookup
    TOOL_REGISTRY["document_intelligence"] = document_intelligence
    TOOL_REGISTRY["market_intelligence_search"] = market_intelligence_search
    TOOL_REGISTRY["trade_calculator"] = trade_calculator
    TOOL_REGISTRY["partnership_profiler"] = partnership_profiler


def execute_tool(call: dict) -> dict:
    register_tools()
    tool_name = call["tool_name"]
    tool_input = call.get("tool_input", {})

    if tool_name not in TOOL_REGISTRY:
        return {
            "call_id": call["call_id"],
            "tool_name": tool_name,
            "status": "error",
            "content": {},
            "latency_ms": 0,
            "error_message": f"Unknown tool: {tool_name}",
        }

    try:
        result = TOOL_REGISTRY[tool_name](**tool_input)
        return {
            "call_id": call["call_id"],
            "tool_name": tool_name,
            "status": result.status,
            "content": result.content,
            "latency_ms": result.latency_ms,
            "error_message": result.error_message,
            "tokens_used": result.tokens_used,
        }
    except Exception as e:
        return {
            "call_id": call["call_id"],
            "tool_name": tool_name,
            "status": "error",
            "content": {},
            "latency_ms": 0,
            "error_message": str(e),
        }


async def executor_node(state: AgentState) -> AgentState:
    pending = state.get("pending_tool_calls", [])
    completed = state.get("completed_tool_calls", [])
    tool_results = state.get("tool_results", [])

    if not pending:
        return state

    ready_calls = [c for c in pending if all(dep in [comp["call_id"] for comp in completed] for dep in c.get("depends_on", []))]

    if not ready_calls:
        ready_calls = pending[:1]

    results = []
    for call in ready_calls:
        result = await asyncio.get_event_loop().run_in_executor(None, execute_tool, call)
        results.append(result)

    new_completed = completed + [r for r in results]
    new_pending = [c for c in pending if c["call_id"] not in [r["call_id"] for r in results]]
    new_results = tool_results + results

    return {
        **state,
        "completed_tool_calls": new_completed,
        "pending_tool_calls": new_pending,
        "tool_results": new_results,
        "iteration": state.get("iteration", 0) + 1,
    }
