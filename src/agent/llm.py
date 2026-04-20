import os
import json
from typing import Optional
from langchain_openai import ChatOpenAI
from src.agent.state import AgentState


def get_llm(temperature: float = 1.0):
    return ChatOpenAI(
        model="MiniMax-M2.7",
        openai_api_key=os.getenv("MINIMAX_API_KEY"),
        openai_api_base="https://api.minimax.io/v1",
        temperature=temperature,
        model_kwargs={"reasoning_split": True},
    )


def llm_node(state: AgentState, system_prompt: str, user_prompt: str) -> dict:
    from src.agent.budget import BudgetTracker

    budget_tracker = BudgetTracker(cap_usd=state.get("budget_cap_usd", 0.50))

    llm = get_llm(temperature=1.0)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if user_prompt:
        messages.append({"role": "user", "content": user_prompt})

    messages.append({"role": "user", "content": state.get("user_query", "")})

    response = llm.invoke(messages)
    response_text = ""
    thinking = ""

    for block in response.content:
        if hasattr(block, "type"):
            if block.type == "text":
                response_text = block.text
            elif block.type == "thinking":
                thinking = block.thinking

    input_tokens = response.usage_metadata.get("input_tokens", 0) if hasattr(response, "usage_metadata") else len(str(messages))
    output_tokens = response.usage_metadata.get("output_tokens", 0) if hasattr(response, "usage_metadata") else len(response_text)

    budget_tracker.track("MiniMax-M2.7", input_tokens, output_tokens, "llm_call", state.get("session_id", "default"))

    return {
        "text": response_text,
        "thinking": thinking,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }
