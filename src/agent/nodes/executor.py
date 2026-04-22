import asyncio
import time
import logging
from typing import Optional
from src.agent.state import AgentState
from src.agent.llm import llm_node

logging.getLogger("chromadb.segment").setLevel(logging.CRITICAL)

TOOL_REGISTRY = {}

DEP_OUTPUT_MAP = {
    ("trade_regulations_lookup", "uk_duty_rate"): "duty_rate",
    ("trade_regulations_lookup", "vat_rate"): "vat_rate",
    ("trade_calculator", "result"): "result",
    ("trade_calculator", "future_value"): "future_value",
    ("currency_convert", "result"): "result",
    ("roi_projection", "total_return_pct"): "total_return_pct",
}

PLACEHOLDER_VALUES = {None, "null", "<null>", "<from_step_1>", "<from_step_2>", "<from_step_3>", 0}


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


def _get_dep_value(dep_call_id: str, completed: list[dict]) -> dict | None:
    for comp in completed:
        if comp.get("call_id") == dep_call_id:
            return comp.get("content", {})
    return None


def _get_simple_result(content: dict) -> float | int | None:
    if not isinstance(content, dict):
        return None
    if len(content) == 1:
        val = list(content.values())[0]
        if isinstance(val, (int, float)):
            return val
    if "result" in content:
        val = content["result"]
        if isinstance(val, (int, float)):
            return val
    if "future_value" in content:
        val = content["future_value"]
        if isinstance(val, (int, float)):
            return val
    return None


def _find_injectable_value(dep_content: dict, dep_tool: str, dst_param: str) -> float | int | None:
    results = dep_content.get("results", [])
    if isinstance(results, list) and results:
        first_result = results[0]
        if isinstance(first_result, dict):
            for (map_tool, map_field), map_param in DEP_OUTPUT_MAP.items():
                if map_tool == dep_tool and map_param == dst_param and map_field in first_result:
                    return first_result[map_field]

    return _get_simple_result(dep_content)


def _inject_dep_params(call: dict, completed: list[dict]) -> dict:
    tool_input = call.get("tool_input", {})
    call_tool = call.get("tool_name", "")
    modified = False

    for dep_id in call.get("depends_on", []):
        dep_content = _get_dep_value(dep_id, completed)
        if not dep_content:
            continue

        dep_call = next((c for c in completed if c.get("call_id") == dep_id), None)
        if not dep_call:
            continue
        dep_tool = dep_call.get("tool_name", "")

        params = tool_input.get("params", {})
        if isinstance(params, dict):
            for dst_param, cur_val in list(params.items()):
                if cur_val in PLACEHOLDER_VALUES or (isinstance(cur_val, str) and "from_step" in cur_val.lower()):
                    val = _find_injectable_value(dep_content, dep_tool, dst_param)
                    if val is not None:
                        params[dst_param] = val
                        modified = True

        for key in list(tool_input.keys()):
            if key in ("operation", "query_type", "entity_name", "category", "filters", "top_k", "domain_filter", "analysis_type", "entity_type"):
                continue
            cur_val = tool_input.get(key)
            if cur_val in PLACEHOLDER_VALUES or (isinstance(cur_val, str) and "from_step" in cur_val.lower()):
                dst_param = key
                val = _find_injectable_value(dep_content, dep_tool, dst_param)
                if val is not None:
                    tool_input[key] = val
                    modified = True

    if modified:
        call = dict(call)
        call["tool_input"] = dict(tool_input)

    return call


def executor_node(state: AgentState) -> AgentState:
    pending = state.get("pending_tool_calls", [])
    completed = state.get("completed_tool_calls", [])
    tool_results = state.get("tool_results", [])

    if not pending:
        return state

    completed_ids = [comp["call_id"] for comp in completed]
    ready_calls = [c for c in pending if all(dep in completed_ids for dep in c.get("depends_on", []))]

    if not ready_calls:
        ready_calls = pending[:1]

    ready_calls = [_inject_dep_params(rc, completed) for rc in ready_calls]

    results = [execute_tool(call) for call in ready_calls]

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
