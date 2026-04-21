import os
import json
from typing import Optional
from dotenv import load_dotenv
from openai import OpenAI
from src.agent.state import AgentState

load_dotenv(override=True)


def get_llm(temperature: float = 1.0):
    api_key = os.getenv("MINIMAX_API_KEY", "")
    return OpenAI(api_key=api_key, base_url="https://api.minimax.io/v1").chat.completions


def llm_node(state: AgentState, system_prompt: str, user_prompt: str) -> dict:
    from src.agent.budget import BudgetTracker

    budget_tracker = BudgetTracker(cap_usd=state.get("budget_cap_usd", 0.50))

    client = get_llm(temperature=1.0)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if user_prompt:
        messages.append({"role": "user", "content": user_prompt})

    messages.append({"role": "user", "content": state.get("user_query", "")})

    response = client.create(
        model="MiniMax-M2.7",
        messages=messages,
        extra_body={"reasoning_split": True},
    )

    response_text = ""
    thinking = ""

    msg = response.choices[0].message
    if hasattr(msg, "content") and msg.content:
        response_text = msg.content
    if hasattr(msg, "reasoning_details") and msg.reasoning_details:
        reasoning_list = msg.reasoning_details
        if isinstance(reasoning_list, list) and len(reasoning_list) > 0:
            first = reasoning_list[0]
            if isinstance(first, dict) and "text" in first:
                thinking = first["text"]
            elif isinstance(first, str):
                thinking = first

    input_tokens = response.usage.prompt_tokens if hasattr(response, "usage") else len(str(messages))
    output_tokens = response.usage.completion_tokens if hasattr(response, "usage") else len(response_text)

    budget_tracker.track("MiniMax-M2.7", input_tokens, output_tokens, "llm_call", state.get("session_id", "default"))

    return {
        "text": response_text,
        "thinking": thinking,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }
