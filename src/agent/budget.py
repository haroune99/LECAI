from dataclasses import dataclass, field
from typing import Optional
import os


MINIMAX_COSTS = {
    "MiniMax-M2.7": {"input_per_1k": 0.30, "output_per_1k": 1.20},
    "MiniMax-M2.7-highspeed": {"input_per_1k": 0.60, "output_per_1k": 2.40},
}


@dataclass
class BudgetTracker:
    cap_usd: float = 0.50
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    calls: list[dict] = field(default_factory=list)

    def track(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        call_type: str,
        query_id: str,
    ) -> dict:
        cost = (input_tokens / 1000 * MINIMAX_COSTS.get(model, MINIMAX_COSTS["MiniMax-M2.7"])["input_per_1k"]) + \
               (output_tokens / 1000 * MINIMAX_COSTS.get(model, MINIMAX_COSTS["MiniMax-M2.7"])["output_per_1k"])

        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost_usd += cost

        record = {
            "call_type": call_type,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": cost,
            "cumulative_cost": self.total_cost_usd,
            "query_id": query_id,
        }
        self.calls.append(record)
        return record

    @property
    def budget_exceeded(self) -> bool:
        return self.total_cost_usd >= self.cap_usd

    @property
    def budget_remaining(self) -> float:
        return max(0, self.cap_usd - self.total_cost_usd)

    def project_cost_per_1000_queries(self) -> float:
        if not self.calls:
            return 0.0
        queries = len(set(c["query_id"] for c in self.calls))
        return (self.total_cost_usd / queries) * 1000 if queries else 0.0

    def breakdown_by_node(self) -> dict:
        from collections import defaultdict
        breakdown = defaultdict(float)
        for call in self.calls:
            breakdown[call["call_type"]] += call["cost_usd"]
        return dict(breakdown)

    def reset(self):
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost_usd = 0.0
        self.calls = []
