import json
import os
import sys
import time

import streamlit as st

# --- PATH SETUP ---
# Add the current directory to sys.path so we can import 'agents' and 'mcp_server'
sys.path.append(os.path.abspath("."))

# --- UI CONFIG (Must be first Streamlit command) ---
st.set_page_config(page_title="Agentic Fact-Checker", page_icon="🕵️‍♂️", layout="wide")

# --- IMPORT ORCHESTRATOR WITH DEBUGGING ---
try:
    from agents.graph import app as agent_app
except ImportError as e:
    # This unmasks the actual error (e.g., "No module named 'langgraph'")
    st.error(f"❌ Import Error: {e}")
    st.info(
        "Tip: Check if you have installed all dependencies: `pip install langgraph langchain-ollama mcp pandas duckduckgo-search`"
    )
    st.stop()
except Exception as e:
    st.error(f"❌ Unexpected Error: {e}")
    st.stop()

# --- STYLING ---
st.markdown(
    """
<style>
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
    }
    .verdict-box {
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        text-align: center;
    }
    .supported { background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
    .refuted { background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
    .nei { background-color: #e2e3e5; color: #383d41; border: 1px solid #d6d8db; }
</style>
""",
    unsafe_allow_html=True,
)

# --- SIDEBAR ---
with st.sidebar:
    st.title("🕵️‍♂️ Multi Agent System")
    st.markdown("### Active Agents")

    st.success("🧠 **Analyzer**\n\n*Role:* Decomposes claims\n*Model:* Qwen 2.5")
    st.info("🔍 **Searcher**\n\n*Role:* Hybrid Retrieval\n*Source:* FEVER + ISOT + Web")
    st.warning("⚖️ **Verifier**\n\n*Role:* Adjudication\n*Output:* JSON Verdict")

    st.divider()
    st.markdown("### System Metrics")
    if "latency" in st.session_state:
        st.metric("Last Latency", f"{st.session_state['latency']:.2f}s")
    if "tools" in st.session_state:
        st.metric("Tools Used", f"{st.session_state['tools']}")

# --- MAIN INTERFACE ---
st.title("📰 AI News Fact-Checker")
st.markdown(
    "Enter a news claim below. The **Multi-Agent System** will analyze, search, and verify it using trusted datasets and live web search."
)

# Input
claim = st.text_input(
    "Claim to Verify:", placeholder="e.g. Barack Obama was born in Kenya."
)

# Execution
if st.button("🔍 Verify Claim") and claim:

    # Containers for real-time updates
    status_container = st.status("🚀 Starting Multi-Agent Workflow...", expanded=True)

    start_time = time.time()

    # Initialize State
    final_state = None
    collected_evidence = []

    try:
        # Stream the LangGraph execution
        # We pass "tool_calls": 0 to initialize the counter
        for event in agent_app.stream({"claim": claim, "tool_calls": 0}):

            # 1. ANALYZER UPDATE
            if "Analyzer" in event:
                data = event["Analyzer"]
                status_container.write(
                    "🧠 **Analyzer:** Decomposed claim into search queries."
                )
                status_container.json(data.get("queries"))

            # 2. SEARCHER UPDATE
            if "Searcher" in event:
                data = event["Searcher"]
                evidence_list = data.get("evidence", [])
                evidence_count = len(evidence_list)
                collected_evidence = evidence_list  # Store for display later

                status_container.write(
                    f"🔍 **Searcher:** Found {evidence_count} pieces of evidence."
                )
                with status_container.expander("View Raw Evidence"):
                    for item in evidence_list:
                        st.caption(item)

            # 3. VERIFIER UPDATE
            if "Verifier" in event:
                final_state = event["Verifier"]
                status_container.write(
                    "⚖️ **Verifier:** Analyzing evidence and forming verdict..."
                )

        # COMPLETE
        status_container.update(
            label="✅ Verification Complete", state="complete", expanded=False
        )

        end_time = time.time()

        # Save results to session state so they persist after rerun
        st.session_state["latency"] = end_time - start_time
        st.session_state["tools"] = final_state.get("tool_calls", 0)
        st.session_state["last_verdict"] = final_state.get("verdict", "NEI")
        st.session_state["last_reason"] = final_state.get(
            "reason", "No reason provided."
        )
        st.session_state["last_evidence"] = collected_evidence
        st.session_state["last_claim"] = claim

        st.rerun()  # Rerun to update sidebar metrics immediately

    except Exception as e:
        status_container.update(label="❌ System Error", state="error")
        st.error(f"An error occurred: {e}")

# --- RESULTS DISPLAY ---
# Only display results if the claim in the input box matches the claim we just processed
if "last_verdict" in st.session_state and claim == st.session_state.get(
    "last_claim", ""
):

    verdict = st.session_state["last_verdict"]
    reason = st.session_state["last_reason"]

    # Dynamic Styling
    class_name = "nei"
    emoji = "❓"
    if verdict == "SUPPORTED":
        class_name = "supported"
        emoji = "✅"
    elif verdict == "REFUTED":
        class_name = "refuted"
        emoji = "❌"

    st.markdown(
        f"""
    <div class="verdict-box {class_name}">
        <h2>{emoji} Verdict: {verdict}</h2>
        <p>{reason}</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Evidence Section
    if "last_evidence" in st.session_state:
        with st.expander("📂 View Supporting Evidence Sources"):
            for item in st.session_state["last_evidence"]:
                st.markdown(f"- {item}")
