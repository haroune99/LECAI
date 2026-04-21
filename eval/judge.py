import os
import json
from typing import Optional
from openai import OpenAI


MODEL_FOR_JUDGE = "MiniMax-M2.7"


JUDGE_PROMPT = """You are evaluating the quality of an AI trade intelligence agent's answer for London Export Corporation.

Score the answer 0, 1, or 2 based on this rubric:
2 — EXCELLENT: The answer is factually correct, directly addresses the query, uses the right information sources, cites evidence, and contains no significant omissions or hallucinations.
1 — PARTIAL: The answer is partially correct, or addresses the query but misses key information, or uses an indirect route, or contains a minor factual error that doesn't invalidate the overall answer.
0 — FAIL: The answer is wrong, fails to address the query, contains a significant hallucination, or the agent gave up without a meaningful answer.

User query: {query}

Key facts the answer should contain:
{key_facts}

Required minimum criteria: {min_score_criteria}

Agent's answer:
{actual_answer}

Respond with JSON only, no preamble:
{{"score": <0|1|2>, "reason": "<one sentence explanation>", "missing_facts": ["<fact1>", ...], "hallucinations_detected": ["<claim1>", ...]}}
"""


def llm_judge_score(query: str, expected: dict, actual_answer: str, sources: list[str]) -> dict:
    key_facts = expected.get("key_facts", [])
    min_criteria = expected.get("min_score_criteria", "")

    prompt = JUDGE_PROMPT.format(
        query=query,
        key_facts=", ".join(key_facts),
        min_score_criteria=min_criteria,
        actual_answer=actual_answer,
    )

    client = OpenAI(api_key=os.getenv("MINIMAX_API_KEY"), base_url="https://api.minimax.io/v1")

    response = client.chat.completions.create(
        model=MODEL_FOR_JUDGE,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        extra_body={"reasoning_split": True},
    )

    response_text = response.choices[0].message.content or ""

    try:
        result = json.loads(response_text)
        return {
            "score": result.get("score", 1),
            "reason": result.get("reason", ""),
            "missing_facts": result.get("missing_facts", []),
            "hallucinations_detected": result.get("hallucinations_detected", []),
        }
    except json.JSONDecodeError:
        return {
            "score": 1,
            "reason": f"Could not parse judge response: {response_text[:100]}",
            "missing_facts": [],
            "hallucinations_detected": [],
        }
