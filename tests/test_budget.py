import pytest
from src.agent.budget import BudgetTracker


def test_budget_tracker_basic():
    tracker = BudgetTracker(cap_usd=0.50)
    tracker.track("MiniMax-M2.7", 1000, 500, "planner", "test_query")
    assert tracker.total_cost_usd > 0
    assert tracker.budget_remaining < 0.50


def test_budget_exceeded():
    tracker = BudgetTracker(cap_usd=0.001)
    tracker.track("MiniMax-M2.7", 10000, 10000, "planner", "test_query")
    assert tracker.budget_exceeded == True
