import streamlit as st
import asyncio
import os
from src.agent.graph import build_graph
from src.agent.state import default_state
from src.agent.budget import BudgetTracker

st.set_page_config(
    page_title="LEC Trade Intelligence Agent",
    page_icon="🏛",
    layout="wide",
)


def check_password():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        st.title("🏛 LEC Trade Intelligence Agent")
        st.markdown("*Internal research tool — authorised users only*")
        password = st.text_input("Password", type="password")
        if st.button("Enter"):
            if password == st.secrets.get("APP_PASSWORD", "lec2026"):
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Incorrect password")
        st.stop()


check_password()


def render_reasoning_trace(trace: list[dict]):
    for entry in trace:
        with st.expander(f"🧠 {entry.get('step', 'unknown').capitalize()} (iter {entry.get('iteration', 0)})", expanded=False):
            if entry.get("thinking"):
                st.markdown(f"**Model reasoning:**")
                st.code(entry["thinking"][:500] if entry["thinking"] else "N/A", language=None)
            if entry.get("plan_text"):
                st.markdown(f"**Plan:** {entry['plan_text'][:300]}")
            if entry.get("reflection_text"):
                st.markdown(f"**Reflection:** {entry['reflection_text'][:200]}")
            if entry.get("tools_called"):
                st.markdown(f"**Tools called:** {', '.join(entry['tools_called'])}")
            if entry.get("final_answer"):
                st.markdown(f"**Final answer:** {entry['final_answer'][:200]}")


if "messages" not in st.session_state:
    st.session_state.messages = []

if "budget" not in st.session_state:
    st.session_state.budget = BudgetTracker(cap_usd=0.50)

if "total_queries" not in st.session_state:
    st.session_state.total_queries = 0

if "graph" not in st.session_state:
    st.session_state.graph = build_graph()

with st.sidebar:
    st.markdown("### 💰 Session Budget")
    budget = st.session_state.budget
    col1, col2 = st.columns(2)
    col1.metric("Used", f"${budget.total_cost_usd:.4f}")
    col2.metric("Remaining", f"${budget.budget_remaining:.4f}")
    st.progress(min(budget.total_cost_usd / budget.cap_usd, 1.0))
    st.caption(f"Tokens: {budget.total_input_tokens + budget.total_output_tokens:,} | Queries: {st.session_state.total_queries}")

    st.divider()
    st.markdown("### ⚙️ Config")
    prompt_version = st.selectbox("Planner Prompt", ["v1 (Structured)", "v2 (Loose)"], index=0)
    bm25_weight = st.slider("BM25 Weight", 0.0, 1.0, 0.3, 0.05)
    semantic_weight = st.slider("Semantic Weight", 0.0, 1.0, 0.5, 0.05)

    st.divider()
    if st.button("🗑 Clear Session"):
        st.session_state.messages = []
        st.session_state.budget = BudgetTracker(cap_usd=0.50)
        st.session_state.total_queries = 0
        st.rerun()

st.title("🏛 LEC Trade Intelligence Agent")
st.caption("Internal research assistant for London Export Corporation")

with st.expander("💡 Example queries", expanded=False):
    examples = [
        "What is the current UK import duty for Tsingtao beer, and what would be the total landed cost of 10,000 cases?",
        "Profile Meituan Dianping as a potential LEC Global Capital investment — sanctions check included",
        "Summarise LEC's history with China trade since the 1953 founding deal",
    ]
    for ex in examples:
        if st.button(ex, key=ex[:30]):
            st.session_state.pending_query = ex

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "trace" in msg:
            render_reasoning_trace(msg["trace"])

query = st.chat_input("Ask a trade intelligence question...")

if hasattr(st.session_state, "pending_query"):
    query = st.session_state.pending_query
    delattr(st.session_state, "pending_query")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.status("🔍 Researching...", expanded=True) as status:
            st.write("📋 Planning approach...")

            state = default_state()
            state["user_query"] = query
            state["session_id"] = f"streamlit_{st.session_state.total_queries}"
            state["prompt_version"] = prompt_version[0].lower()
            state["budget_cap_usd"] = 0.50

            try:
                graph = st.session_state.graph
                config = {"configurable": {"thread_id": state["session_id"]}}
                result = asyncio.run(graph.ainvoke(state, config=config))
                status.update(label="✅ Complete", state="complete")
            except Exception as e:
                result = {"final_answer": f"Error: {str(e)}", "reasoning_trace": [], "run_status": "failed"}
                status.update(label="❌ Error", state="error")

        st.markdown(result.get("final_answer", ""))

        if result.get("reasoning_trace"):
            with st.expander("🧠 Agent Reasoning", expanded=False):
                render_reasoning_trace(result["reasoning_trace"])

        st.session_state.messages.append({
            "role": "assistant",
            "content": result.get("final_answer", ""),
            "trace": result.get("reasoning_trace", []),
        })

        st.session_state.total_queries += 1
