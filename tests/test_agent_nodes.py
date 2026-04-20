import pytest
from src.agent.nodes.planner import parse_plan_steps, build_tool_calls


def test_parse_plan_steps():
    plan_text = """PLAN:
- Step 1: Look up duty rate using trade_regulations_lookup — depends on: none — rationale: Get the current tariff code
- Step 2: Calculate landed cost using trade_calculator — depends on: Step 1 — rationale: Use the rate from step 1
REASONING: First get the duty rate then calculate the total cost."""

    steps = parse_plan_steps(plan_text)
    assert len(steps) == 2
    assert steps[0]["tool"] == "trade_regulations_lookup"
    assert steps[1]["tool"] == "trade_calculator"


def test_build_tool_calls():
    steps = [
        {"step": "Step 1", "tool": "trade_regulations_lookup", "depends_on": "none"},
        {"step": "Step 2", "tool": "trade_calculator", "depends_on": "Step 1"},
    ]
    calls = build_tool_calls(steps)
    assert len(calls) == 2
    assert calls[0]["tool_name"] == "trade_regulations_lookup"
    assert calls[1]["tool_name"] == "trade_calculator"
