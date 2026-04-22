import streamlit as st
import asyncio
import os
from pathlib import Path

# Set working directory to repo root so data/indexes paths resolve correctly
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(REPO_ROOT)
import sys
sys.path.insert(0, str(REPO_ROOT))

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
                st.code(entry["thinking"] if entry["thinking"] else "N/A", language=None)
            if entry.get("plan_text"):
                st.markdown(f"**Plan:** {entry['plan_text']}")
            if entry.get("reflection_text"):
                st.markdown(f"**Reflection:** {entry['reflection_text']}")
            if entry.get("tools_called"):
                st.markdown(f"**Tools called:** {', '.join(entry['tools_called'])}")
            if entry.get("final_answer"):
                st.markdown(f"**Final answer:** {entry['final_answer']}")


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
        "What is the current UK import duty for Tsingtao beer (HS code 2203), and what would be the total landed cost of 10,000 cases shipped from Qingdao to Liverpool if the FOB price is £8 per case, freight is £3,000, and insurance is 0.5%?",
        "Check if Integrity Technology Group (a Chinese robotics company) is on the UK OFSI sanctions list, and what does our partnership profiler say about their strategic fit for LEC Robotics?",
        "What do our Tsingtao Annual Report documents say about their UK and international distribution strategy, and what does their brand positioning tell us about their premium positioning?",
        "What commodity code applies to flavored waters and functional beverages (HS code 2202), and what UK import duty applies to a new Chinese herbal drink brand importing under this classification?",
        "Convert 500,000 CNY to GBP using the trade calculator's current exchange rate, and then tell me what percentage this represents of LEC Global Capital's typical minimum investment threshold of £2M.",
        "Profile Meituan Dianping as a potential LEC Global Capital investment — who are they, what sector do they operate in, and perform a sanctions check to confirm they are clean (not on UK OFSI list).",
        "What do our documents say about LEC Industries' involvement in industrial AI, automation, or infrastructure projects, and what sectors does LEC Industries currently serve?",
        "Summarise what our documents say about the history of LEC's trade relationship with China, specifically the 1953 founding deal including who founded it, how it started, and what the first trade involved.",
        "Profile Longi Green Energy as a potential LEC Global Capital renewable energy investment — who are they, are they sanctioned, and calculate what a £2,000,000 investment would grow to in CNY over 3 years at 8% annual return.",
        "Based on our documents and tariff data: what is the UK import duty for sparkling water with added flavor (HS code 2202 10 00 00) compared to still mineral water (HS code 2201), and what does our Tsingtao Annual Report say about beverage market positioning to help evaluate a new Chinese beverage brand entering the UK market?",
    ]
    for i, ex in enumerate(examples):
        if st.button(ex[:60] + "..." if len(ex) > 60 else ex, key=f"example_{i}"):
            st.session_state.pending_query = ex

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "trace" in msg:
            render_reasoning_trace(msg["trace"])

query = st.chat_input("Ask a trade intelligence question...")

if "pending_query" in st.session_state:
    query = st.session_state.pending_query
    del st.session_state["pending_query"]

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

            st.session_state.budget.total_cost_usd = result.get("cost_usd", 0.0)
            st.session_state.budget.total_input_tokens = result.get("tokens_input", 0)
            st.session_state.budget.total_output_tokens = result.get("tokens_output", 0)

            st.markdown(result.get("final_answer", ""))

            if result.get("reasoning_trace"):
                st.markdown("**🧠 Agent Reasoning**")
                for entry in result["reasoning_trace"]:
                    with st.expander(f"  {entry.get('step', '?').capitalize()} (iter {entry.get('iteration', 0)})", expanded=False):
                        if entry.get("thinking"):
                            st.markdown(f"**Model reasoning:**")
                            st.code(entry["thinking"] if entry["thinking"] else "N/A", language=None)
                        if entry.get("plan_text"):
                            st.markdown(f"**Plan:** {entry['plan_text']}")
                        if entry.get("reflection_text"):
                            st.markdown(f"**Reflection:** {entry['reflection_text']}")

            st.session_state.messages.append({
                "role": "assistant",
                "content": result.get("final_answer", ""),
                "trace": result.get("reasoning_trace", []),
            })
            st.session_state.total_queries += 1
