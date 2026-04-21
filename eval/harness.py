import json
import asyncio
import time
from pathlib import Path
from src.agent.graph import build_graph
from src.agent.state import default_state
from eval.judge import llm_judge_score


def run_single_query(
    query_id: int,
    query: str,
    expected: dict,
    prompt_version: str,
    graph,
) -> dict:
    start = time.time()

    state = default_state()
    state["user_query"] = query
    state["session_id"] = f"eval_{query_id}_{prompt_version}"
    state["prompt_version"] = prompt_version

    config = {"configurable": {"thread_id": f"eval_{query_id}_{prompt_version}"}}

    try:
        result = graph.invoke(state, config=config)
    except Exception as e:
        return {
            "query_id": query_id,
            "query": query,
            "prompt_version": prompt_version,
            "score": 0,
            "judge_reason": f"Agent crashed: {str(e)}",
            "final_answer": "",
            "tools_called": [],
            "tools_count": 0,
            "retries": 0,
            "iterations": 0,
            "latency_ms": (time.time() - start) * 1000,
            "tokens_input": 0,
            "tokens_output": 0,
            "cost_usd": 0.0,
            "run_status": "failed",
        }

    latency_ms = (time.time() - start) * 1000

    judge_result = llm_judge_score(
        query=query,
        expected=expected,
        actual_answer=result.get("final_answer", ""),
        sources=result.get("sources_cited", []),
    )

    return {
        "query_id": query_id,
        "query": query,
        "prompt_version": prompt_version,
        "score": judge_result["score"],
        "judge_reason": judge_result["reason"],
        "final_answer": result.get("final_answer", ""),
        "tools_called": [r.get("tool_name", "") for r in result.get("tool_results", [])],
        "tools_count": len(result.get("tool_results", [])),
        "retries": result.get("retry_count", 0),
        "iterations": result.get("iteration", 0),
        "latency_ms": latency_ms,
        "tokens_input": result.get("tokens_input", 0),
        "tokens_output": result.get("tokens_output", 0),
        "cost_usd": result.get("cost_usd", 0.0),
        "run_status": result.get("run_status", "unknown"),
    }


async def run_full_eval():
    queries = json.loads(Path("eval/queries.json").read_text())
    graph = build_graph()
    results = []

    for prompt_version in ["v1", "v2"]:
        print(f"\n=== Running eval with prompt {prompt_version} ===")
        for q in queries:
            print(f"  Query {q['id']}: {q['query'][:60]}...", end=" ", flush=True)
            result = await asyncio.to_thread(
                run_single_query,
                q["id"], q["query"], q["expected"],
                prompt_version, graph
            )
            results.append(result)
            print(f"Score: {result['score']}/2 | Latency: {result['latency_ms']:.0f}ms | Cost: ${result['cost_usd']:.5f}")

    output_path = Path("eval/results/eval_results.json")
    output_path.parent.mkdir(exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))

    print("\n" + "=" * 60)
    print_eval_summary(results)
    return results


def print_eval_summary(results: list[dict]):
    import statistics

    for version in ["v1", "v2"]:
        v_results = [r for r in results if r["prompt_version"] == version]
        scores = [r["score"] for r in v_results]
        latencies = [r["latency_ms"] for r in v_results]
        costs = [r["cost_usd"] for r in v_results]

        print(f"\n=== Prompt {version} Summary ===")
        print(f"Average score: {statistics.mean(scores):.2f}/2")
        print(f"Success rate (score=2): {sum(1 for s in scores if s == 2) / len(scores) * 100:.0f}%")
        print(f"p50 latency: {statistics.median(latencies):.0f}ms")
        if latencies:
            sorted_lat = sorted(latencies)
            p95_idx = int(len(sorted_lat) * 0.95)
            print(f"p95 latency: {sorted_lat[p95_idx]:.0f}ms")
        print(f"Total cost: ${sum(costs):.4f}")
        print(f"Avg cost/query: ${statistics.mean(costs):.5f}")
        if statistics.mean(costs) > 0:
            print(f"Projected /1000 queries: ${statistics.mean(costs) * 1000:.2f}")


if __name__ == "__main__":
    asyncio.run(run_full_eval())
