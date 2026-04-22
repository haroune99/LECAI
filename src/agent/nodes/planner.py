import json
import uuid
import time
from src.agent.state import AgentState
from src.agent.llm import llm_node
from src.agent.prompts.planner import PLANNER_V1, PLANNER_V2
from src.tools.base import format_tool_results


TOOL_DESCRIPTIONS = {
    "trade_regulations_lookup": "trade_regulations_lookup(query_type, commodity_code, entity_name, category) — UK tariff codes, OFSI sanctions checks, HMRC regulatory requirements. See MAPPINGS below for query_type values.",
    "document_intelligence": "document_intelligence(query, filters, top_k) — RAG over LEC document corpus with hybrid search",
    "market_intelligence_search": "market_intelligence_search(query, domain_filter, recency_days) — Real-time web search via Tavily",
    "trade_calculator": "trade_calculator(operation, params) — operation must be one of: landed_cost, currency_convert, duty_calculation, roi_projection, margin_analysis. params is a dict of operation-specific parameters.",
    "partnership_profiler": "partnership_profiler(entity_name, entity_type, analysis_type) — Entity profiling, strategic fit, risk assessment",
}

QUERY_TYPE_MAPPINGS = """
QUERY_TYPE MAPPINGS for trade_regulations_lookup:
  query_type="tariff"     → Lookup UK import duty by HS commodity code
                            → Use when: user asks about duty rates, tariff codes, HS codes, VAT
                            → MUST provide: commodity_code (e.g. "2203" for beer, "8517" for telecoms)
                            → SQL: SELECT * FROM tariff_codes WHERE commodity_code = "2203"

  query_type="sanctions_check" → Check if an entity is on the OFSI sanctions list
                            → Use when: user asks about sanctions, OFSI, blocked entities, due diligence
                            → MUST provide: entity_name (e.g. "Huawei", "Meituan")
                            → SQL: SELECT * FROM sanctions_entities WHERE entity_name LIKE "%Huawei%"

  query_type="regulatory_requirements" → Lookup UK import compliance requirements
                            → Use when: user asks about UKCA, FSA, food safety, compliance
                            → MUST provide: category (e.g. "beverages", "machinery", "food")
                            → SQL: SELECT * FROM regulatory_requirements WHERE applies_to = "beverages"

EXAMPLES:
  "What is the duty on beer?"           → query_type="tariff", commodity_code="2203"
  "Is Huawei sanctioned?"               → query_type="sanctions_check", entity_name="Huawei"
  "What are the FSA requirements?"      → query_type="regulatory_requirements", category="beverages"
  "What is the commodity code for beer?" → query_type="tariff", commodity_code="2203"
"""

MINIMAX_COSTS = {"MiniMax-M2.7": {"input_per_1m": 0.30, "output_per_1m": 1.20}}


def _calc_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    rates = MINIMAX_COSTS.get(model, MINIMAX_COSTS["MiniMax-M2.7"])
    return (input_tokens / 1_000_000 * rates["input_per_1m"]) + (output_tokens / 1_000_000 * rates["output_per_1m"])


def _has_placeholder_value(val) -> bool:
    if val is None or val == "null" or val == "<null>":
        return True
    if isinstance(val, str) and ("from_step" in val.lower() or "<from step" in val.lower() or "step_" in val.lower()):
        return True
    if isinstance(val, dict):
        return any(_has_placeholder_value(v) for v in val.values())
    return False


def _auto_inject_dependencies(plan_steps: list[dict]) -> list[dict]:
    for i, step in enumerate(plan_steps):
        if i == 0:
            continue
        tool_string = step.get("tool", "unknown")
        tool_kwargs = step.get("_tool_kwargs", {})
        if not tool_kwargs:
            tool_name, tool_kwargs = parse_tool_invocation(tool_string)
        has_placeholder = _has_placeholder_value(tool_kwargs)
        if has_placeholder:
            existing_dep = str(step.get("depends_on", "none")).lower()
            if "none" in existing_dep or not step.get("depends_on"):
                step["depends_on"] = f"Step {i}"
    return plan_steps


def planner_node(state: AgentState) -> AgentState:
    prompt_version = state.get("prompt_version", "v1")
    budget_remaining = state.get("budget_cap_usd", 0.50) - state.get("cost_usd", 0.0)

    tool_desc_lines = [f"- {name}: {desc}" for name, desc in TOOL_DESCRIPTIONS.items()]
    tool_descriptions = "\n".join(tool_desc_lines)

    planner_prompt = PLANNER_V1 if prompt_version == "v1" else PLANNER_V2

    system_prompt = planner_prompt.format(
        tool_descriptions=tool_descriptions,
        query_type_mappings=QUERY_TYPE_MAPPINGS.strip(),
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
    existing_completed = state.get("completed_tool_calls", [])
    pending_calls = build_tool_calls(plan_steps, existing_completed)

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
        "completed_tool_calls": existing_completed,  # PRESERVE across planner re-runs
        "tool_results": state.get("tool_results", []),  # KEEP accumulated results across iterations
        "tokens_input": state.get("tokens_input", 0) + input_tokens,
        "tokens_output": state.get("tokens_output", 0) + output_tokens,
        "cost_usd": state.get("cost_usd", 0.0) + _calc_cost("MiniMax-M2.7", input_tokens, output_tokens),
        "budget_exceeded": (state.get("cost_usd", 0.0) + _calc_cost("MiniMax-M2.7", input_tokens, output_tokens)) >= state.get("budget_cap_usd", 0.50),
        "reasoning_trace": reasoning_trace,
        "last_thinking": thinking,
    }


def parse_plan_steps(plan_text: str) -> list[dict]:
    steps = []
    current_step = {}
    import re

    def _add_step():
        nonlocal current_step
        if current_step:
            steps.append(current_step)
            current_step = {}

    tool_call_blocks = re.findall(r'\[TOOL_CALL\](.*?)\[/TOOL_CALL\]', plan_text, re.DOTALL)
    for block in tool_call_blocks:
        tool_name_match = re.search(r'tool\s*=>\s*"([^"]+)"', block)
        if tool_name_match:
            tool_name = tool_name_match.group(1).strip()
            if tool_name in TOOL_REGISTRY_PLAIN:
                _add_step()
                current_step = {"step": f"Tool call: {tool_name}", "tool": tool_name, "depends_on": "none", "_tool_kwargs": {}}

                kv_pattern = re.compile(r'--(\w+)\s*(?:"([^"]*)"|(\{[^}]*\}))')
                for match in kv_pattern.finditer(block):
                    k = match.group(1)
                    v = match.group(2) or match.group(3)
                    if v is None:
                        continue
                    if v.startswith('{'):
                        import json as _json
                        try:
                            current_step["_tool_kwargs"][k] = _json.loads(v)
                        except _json.JSONDecodeError:
                            current_step["_tool_kwargs"][k] = v
                    elif k in ("commodity_code", "entity_name", "category", "query_type", "operation", "analysis_type", "entity_type"):
                        current_step["_tool_kwargs"][k] = v
                    else:
                        try:
                            current_step["_tool_kwargs"][k] = int(v)
                        except ValueError:
                            try:
                                current_step["_tool_kwargs"][k] = float(v)
                            except ValueError:
                                current_step["_tool_kwargs"][k] = v
                continue

    invoke_blocks = re.findall(r'<invoke name="([^"]+)"[^>]*>(.*?)(?:</invoke>|</minimax:tool_call>)', plan_text, re.DOTALL)
    for tool_name, block_content in invoke_blocks:
        if tool_name in TOOL_REGISTRY_PLAIN:
            _add_step()
            current_step = {"step": f"Tool call: {tool_name}", "tool": tool_name, "depends_on": "none", "_tool_kwargs": {}}

            param_matches = re.findall(r'<parameter name="([^"]+)"[^>]*>([^<]+)</parameter>', block_content)
            for k, v in param_matches:
                if k in ("commodity_code", "entity_name", "category", "query_type", "operation", "analysis_type", "entity_type"):
                    current_step["_tool_kwargs"][k] = v.strip()
                else:
                    try:
                        current_step["_tool_kwargs"][k] = int(v.strip())
                    except ValueError:
                        try:
                            current_step["_tool_kwargs"][k] = float(v.strip())
                        except ValueError:
                            current_step["_tool_kwargs"][k] = v.strip()
            continue

    for line in plan_text.split("\n"):
        line = line.strip()
        if len(line) < 4:
            continue

        # Skip commentary lines that aren't actual plan steps
        if re.match(r"^-\s+Step\s+\d+\s+is\s+(independent|parallel)", line):
            continue

        # Primary pattern: - Step N: ... // ARGS: ... — depends on: ...
        args_match = re.match(r"^- Step \d+[:：]\s*(.+?)\s*//\s*ARGS:\s*(.+?)\s*—\s*depends on:\s*(.+?)(?:\s*—.*)?$", line)
        if args_match:
            step_name = args_match.group(1).strip()
            args_str = args_match.group(2).strip()
            depends = args_match.group(3).strip()

            tool_name, kwargs = parse_tool_invocation(args_str)

            if tool_name and tool_name in TOOL_REGISTRY_PLAIN:
                if current_step:
                    steps.append(current_step)
                current_step = {"step": step_name, "tool": tool_name, "depends_on": depends, "_tool_kwargs": kwargs}
            continue

        # Fallback: - Step N: ... using tool_name — depends on: ... (plain, no args)
        if line.startswith("- Step ") or line.startswith("— Step ") or line.startswith("– Step "):
            _add_step()
            parts = line.split(" using ", 1)
            if len(parts) == 2:
                step_name = parts[0].split(":", 1)[1].strip() if ":" in parts[0] else parts[0]
                tool_part = parts[1].split(" — depends on:")[0].strip()
                depends = "none"
                if " — depends on: " in line:
                    depends = line.split(" — depends on: ", 1)[1].split(" —")[0].strip()
                tool_name, tool_kwargs = parse_tool_invocation(tool_part)
                current_step = {"step": step_name, "tool": tool_name, "depends_on": depends}
                if tool_kwargs:
                    current_step["_tool_kwargs"] = tool_kwargs
            else:
                current_step = {"step": line, "tool": "unknown", "depends_on": "none"}
        elif line.startswith("Step ") and line[5:6].isdigit():
            _add_step()
            parts = line.split(" using ", 1)
            if len(parts) == 2:
                step_name = parts[0].split(":", 1)[1].strip() if ":" in parts[0] else parts[0]
                tool_name, tool_kwargs = parse_tool_invocation(parts[1].split(" — depends on:")[0].strip())
                depends = "none"
                if " — depends on: " in line:
                    depends = line.split(" — depends on: ", 1)[1].split(" —")[0].strip()
                current_step = {"step": step_name, "tool": tool_name, "depends_on": depends}
                if tool_kwargs:
                    current_step["_tool_kwargs"] = tool_kwargs
            else:
                current_step = {"step": line, "tool": "unknown", "depends_on": "none"}
        elif line[0].isdigit() and "." in line[:4]:
            _add_step()
            parts = line.split(" using ", 1)
            if len(parts) == 2:
                step_name = parts[0].split(":", 1)[1].strip() if ":" in parts[0] else parts[0]
                tool_name, tool_kwargs = parse_tool_invocation(parts[1].split(" — depends on:")[0].strip())
                depends = "none"
                if " — depends on: " in line:
                    depends = line.split(" — depends on: ", 1)[1].split(" —")[0].strip()
                current_step = {"step": step_name, "tool": tool_name, "depends_on": depends}
                if tool_kwargs:
                    current_step["_tool_kwargs"] = tool_kwargs
            else:
                current_step = {"step": line, "tool": "unknown", "depends_on": "none"}
        elif line.startswith("REASONING:"):
            break

        tool_call_match = re.match(r'\[TOOL_CALL\]\s*\{tool\s*=>\s*"([^"]+)"', line)
        if tool_call_match:
            tool_name = tool_call_match.group(1).strip()
            if tool_name in TOOL_REGISTRY_PLAIN:
                _add_step()
                current_step = {"step": f"Tool call: {tool_name}", "tool": tool_name, "depends_on": "none", "_tool_kwargs": {}}
            continue

        if '<invoke ' in line or 'Ray' in line or '</invoke>' in line:
            invoke_match = re.search(r'<invoke name="([^"]+)"', line)
            if invoke_match:
                tool_name = invoke_match.group(1).strip()
                if tool_name in TOOL_REGISTRY_PLAIN:
                    _add_step()
                    current_step = {"step": f"Tool call: {tool_name}", "tool": tool_name, "depends_on": "none", "_tool_kwargs": {}}
            continue

    _add_step()
    steps = [s for s in steps if s.get("tool") != "unknown"]
    return steps


TOOL_REGISTRY_PLAIN = {
    "trade_regulations_lookup",
    "document_intelligence",
    "market_intelligence_search",
    "trade_calculator",
    "partnership_profiler",
}


def parse_tool_invocation(tool_string: str) -> tuple[str, dict]:
    import re
    import json
    tool_string = tool_string.strip()
    kwargs = {}

    paren_match = re.search(r"^(.+?)\((.*)\)$", tool_string, re.DOTALL)
    if paren_match:
        tool_name = paren_match.group(1).strip()
        args_str = paren_match.group(2)
    else:
        with_match = re.match(r"^(.+?)\s+with\s+(.+)$", tool_string)
        if with_match:
            tool_name = with_match.group(1).strip()
            args_str = with_match.group(2)
        else:
            tool_name = tool_string
            args_str = ""

    args_str = args_str.strip()

    def extract_braced_value(s: str, start: int) -> tuple[str | None, int]:
        if start >= len(s) or s[start] != '{':
            return None, start
        depth = 0
        i = start
        while i < len(s):
            c = s[i]
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    return s[start:i+1], i+1
            i += 1
        return None, start

    i = 0
    while i < len(args_str):
        m = re.match(r'(\w+)\s*=\s*', args_str[i:])
        if not m:
            i += 1
            continue
        key = m.group(1)
        value_start = i + m.end()
        if value_start < len(args_str):
            if args_str[value_start] == '"':
                end = args_str.find('"', value_start + 1)
                if end != -1:
                    kwargs[key] = args_str[value_start+1:end]
                    i = end + 1
                    continue
            elif args_str[value_start] == '{':
                braced_val, next_i = extract_braced_value(args_str, value_start)
                if braced_val is not None:
                    try:
                        kwargs[key] = json.loads(braced_val)
                    except json.JSONDecodeError:
                        kwargs[key] = braced_val
                    i = next_i
                    continue
            elif args_str[value_start].isdigit():
                m2 = re.match(r'(-?\d+)', args_str[value_start:])
                if m2:
                    kwargs[key] = int(m2.group(1))
                    i = value_start + m2.end()
                    continue
        i += 1

    return tool_name, kwargs


def _normalize_kwargs(kwargs: dict) -> dict:
    def normalize_val(v):
        if isinstance(v, dict):
            return tuple(sorted((k, normalize_val(val)) for k, val in v.items()))
        if isinstance(v, (int, float)):
            return str(v)
        return v
    return tuple(sorted((k, normalize_val(val)) for k, val in kwargs.items()))


def build_tool_calls(plan_steps: list[dict], existing_completed: list | None = None) -> list[dict]:
    if existing_completed is None:
        existing_completed = []

    plan_steps = _auto_inject_dependencies(list(plan_steps))

    calls = []
    step_name_to_index = {}
    completed_call_ids = {comp["call_id"] for comp in existing_completed if comp.get("status") == "success"}
    completed_normalized = {}
    for comp in existing_completed:
        if comp.get("status") == "success":
            norm = _normalize_kwargs(comp.get("tool_input", {}))
            key = (comp.get("tool_name"), norm)
            completed_normalized[key] = comp["call_id"]

    for i, step in enumerate(plan_steps):
        tool_string = step.get("tool", "unknown")
        tool_name, tool_kwargs = parse_tool_invocation(tool_string)
        if step.get("_tool_kwargs"):
            tool_kwargs = step["_tool_kwargs"]

        norm = _normalize_kwargs(tool_kwargs)
        match_key = (tool_name, norm)
        if match_key in completed_normalized:
            call_id = completed_normalized[match_key]
        else:
            call_id = f"call_{uuid.uuid4().hex[:8]}"

        depends_indices = []
        depends_raw = step.get("depends_on", "none")
        if depends_raw and depends_raw != "none":
            import re
            indices = re.findall(r'\d+', str(depends_raw))
            depends_indices = [int(x) - 1 for x in indices]

        calls.append({
            "tool_name": tool_name,
            "tool_input": tool_kwargs,
            "call_id": call_id,
            "depends_on": depends_indices,
            "status": "pending",
        })

        step_name = step.get("step", "")
        step_name_to_index[step_name] = i

    call_id_by_index = {i: c["call_id"] for i, c in enumerate(calls)}

    for call in calls:
        resolved_deps = []
        for dep_idx in call["depends_on"]:
            dep_call_id = call_id_by_index.get(dep_idx)
            if dep_call_id:
                if dep_call_id in completed_call_ids:
                    resolved_deps.append(dep_call_id)
                elif dep_call_id in completed_normalized.values():
                    resolved_deps.append(dep_call_id)
        call["depends_on"] = resolved_deps

    return calls
