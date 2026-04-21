#!/usr/bin/env python
"""
LEC Agent — Full Execution Tracer

Uses graph.stream() to show every node's complete state after each execution.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
os.environ["DOTENV_LOAD"] = "override"

from dotenv import load_dotenv
load_dotenv(override=True)

import argparse
import json

from src.agent.graph import build_graph
from src.agent.state import default_state


def pretty_tool_results(tool_results: list) -> str:
    if not tool_results:
        return "  (none)"
    lines = []
    for r in tool_results:
        status_icon = "✓" if r.get("status") == "success" else "✗" if r.get("status") == "error" else "?"
        lines.append(f"  {status_icon} {r.get('tool_name', '?')} [{r.get('status', '?')}]: {r.get('error_message') or 'OK'}")
        content = r.get("content", {})
        if content:
            if isinstance(content, dict):
                keys = list(content.keys())
                if "results" in content and isinstance(content["results"], list):
                    lines.append(f"      → {len(content['results'])} results")
                elif "query" in content:
                    lines.append(f"      → query: {content.get('query', '')[:60]}")
                elif "answer" in content:
                    lines.append(f"      → answer: {str(content.get('answer', ''))[:80]}")
                elif "result" in content:
                    lines.append(f"      → result: {content.get('result')}")
                elif "entity" in content:
                    lines.append(f"      → entity: {content.get('entity', {}).get('name', '?')}")
                elif "future_value" in content:
                    lines.append(f"      → future_value: {content.get('future_value')}")
    return "\n".join(lines) if lines else "  (none)"


def trace_query(user_query: str, prompt_version: str = "v1", max_iterations: int = 8) -> dict:
    graph = build_graph()
    state = default_state()
    state["user_query"] = user_query
    state["prompt_version"] = prompt_version
    state["max_iterations"] = max_iterations
    state["budget_cap_usd"] = 0.50

    print("=" * 80)
    print(f"QUERY ({prompt_version}): {user_query}")
    print("=" * 80)

    all_chunks = []
    total_tokens = {"input": 0, "output": 0}
    total_cost = 0.0

    for i, chunk in enumerate(graph.stream(state)):
        for node_name, node_state in chunk.items():
            print(f"\n{'─' * 80}")
            print(f"▶ Chunk {i:02d} | Node: {node_name.upper()}")
            print(f"{'─' * 80}")

            iteration = node_state.get("iteration", "?")
            cost = node_state.get("cost_usd", 0.0)
            print(f"  Iteration:     {iteration}")
            print(f"  Cost USD:      ${cost:.4f}")
            print(f"  Tokens in:     {node_state.get('tokens_input', 0)}")
            print(f"  Tokens out:    {node_state.get('tokens_output', 0)}")
            print(f"  Budget exceed: {node_state.get('budget_exceeded', False)}")

            # Tool calls / results
            if node_name == "planner":
                pending = node_state.get("pending_tool_calls", [])
                print(f"\n  Plan text ({len(node_state.get('plan_text', ''))} chars):")
                print(f"  {'─' * 40}")
                for line in node_state.get("plan_text", "  (empty)").split("\n"):
                    if line.strip():
                        print(f"    {line}")
                print(f"  {'─' * 40}")
                print(f"\n  Pending tool calls ({len(pending)}):")
                for call in pending:
                    print(f"    • {call.get('tool_name', '?')}({call.get('tool_input', {})})")
                    if call.get("depends_on"):
                        print(f"      depends on: {call['depends_on']}")

            elif node_name == "executor":
                pending = node_state.get("pending_tool_calls", [])
                completed = node_state.get("completed_tool_calls", [])
                results = node_state.get("tool_results", [])
                print(f"\n  Completed calls ({len(completed)}):")
                for c in completed:
                    print(f"    ✓ {c.get('tool_name', '?')} [{c.get('status', '?')}]")
                print(f"\n  Still pending ({len(pending)}):")
                for p in pending:
                    print(f"    ○ {p.get('tool_name', '?')}")
                print(f"\n  New tool results:")
                print(pretty_tool_results(results))

            elif node_name == "reflector":
                status = node_state.get("reflection_status", "?")
                reflection_text = node_state.get("reflection_text", "")
                print(f"\n  Reflection status: {status.upper()}")
                if reflection_text:
                    print(f"  Reflection text:")
                    for line in reflection_text.split("\n"):
                        if line.strip():
                            print(f"    {line.strip()}")
                retry = node_state.get("retry_count", 0)
                print(f"  Retry count: {retry}")

            elif node_name == "answerer":
                answer = node_state.get("final_answer", "")
                print(f"\n  Final answer ({len(answer)} chars):")
                print(f"  {'─' * 40}")
                print(f"  {answer[:800]}" + ("..." if len(answer) > 800 else ""))
                print(f"  {'─' * 40}")
                sources = node_state.get("sources_cited", [])
                print(f"\n  Sources cited: {sources}")

            # Reasoning trace
            reasoning = node_state.get("last_thinking", "")
            if reasoning:
                print(f"\n  MiniMax thinking ({len(reasoning)} chars):")
                print(f"  {'─' * 40}")
                print(f"  {reasoning[:600]}" + ("..." if len(reasoning) > 600 else ""))
                print(f"  {'─' * 40}")

            # Budget / cost
            budget_remaining = node_state.get("budget_cap_usd", 0.5) - cost
            print(f"\n  Budget: ${budget_remaining:.4f} remaining of ${node_state.get('budget_cap_usd', 0.5):.2f}")

            all_chunks.append((i, node_name, node_state))

    # Summary
    print(f"\n{'=' * 80}")
    print("EXECUTION SUMMARY")
    print(f"{'=' * 80}")
    print(f"  Nodes executed: {len(all_chunks)}")
    final = all_chunks[-1][2] if all_chunks else {}
    print(f"  Total cost:    ${final.get('cost_usd', 0.0):.4f}")
    print(f"  Total tokens:   in={final.get('tokens_input', 0)} out={final.get('tokens_output', 0)}")
    print(f"  Final status:   {final.get('run_status', '?')}")
    print(f"  Tool calls:     {len(final.get('completed_tool_calls', []))} completed / {len(final.get('pending_tool_calls', []))} pending")
    print(f"  Reflection:    {final.get('reflection_status', '?')}")
    print(f"  Iterations:     {final.get('iteration', '?')}")

    # Show full reasoning trace
    full_trace = final.get("reasoning_trace", [])
    if full_trace:
        print(f"\n{'=' * 80}")
        print("FULL REASONING TRACE")
        print(f"{'=' * 80}")
        for entry in full_trace:
            print(f"\n  [{entry.get('step', '?').upper()}] iteration={entry.get('iteration', '?')} tokens={entry.get('tokens_used', 0)}")
            thinking = entry.get("thinking", "")
            if thinking:
                print(f"  Thinking: {thinking[:300]}" + ("..." if len(thinking) > 300 else ""))

    return final


def main():
    parser = argparse.ArgumentParser(description="Trace LEC agent execution")
    parser.add_argument("query", nargs="?", help="Query string to trace")
    parser.add_argument("--query-id", type=int, help="Run eval query N (1-10)")
    parser.add_argument("--prompt", choices=["v1", "v2"], default="v1", help="Prompt version")
    parser.add_argument("--max-iter", type=int, default=8, help="Max iterations")
    args = parser.parse_args()

    query = None

    if args.query_id:
        queries_path = Path(__file__).parent.parent / "eval" / "queries.json"
        if queries_path.exists():
            queries = json.loads(queries_path.read_text())
            q = next((q for q in queries if q["id"] == args.query_id), None)
            if q:
                query = q["query"]
                print(f"Running eval query {args.query_id}: {query[:60]}...")
            else:
                print(f"Query {args.query_id} not found in eval/queries.json")
                sys.exit(1)
        else:
            print("eval/queries.json not found")
            sys.exit(1)
    elif args.query:
        query = args.query
    else:
        print("Usage: python trace_query.py \"Your question here\"")
        print("   or: python trace_query.py --query-id 1")
        sys.exit(1)

    trace_query(query, prompt_version=args.prompt, max_iterations=args.max_iter)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    import os
    from dotenv import load_dotenv
    load_dotenv(override=True)
    main()
