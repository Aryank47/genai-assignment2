import json
from typing import List, TypedDict

# Import the bridge to talk to MCP Servers
from agents.mcp_interface import MCPClient
from langgraph.graph import END, StateGraph


# --- STATE DEFINITION ---
class AgentState(TypedDict):
    claim: str
    queries: List[str]
    evidence: List[str]
    verdict: str
    reason: str
    tool_calls: int
    next_step: str  # For the Supervisor to track intent
    retry_count: int  # To prevent infinite loops


# --- SUPERVISOR NODE (THE BRAIN) ---
def supervisor_node(state: AgentState):
    print(
        "\n[Supervisor] Assessing State..."
    )

    # claim = state.get("claim")
    queries = state.get("queries", [])
    evidence = state.get("evidence", [])
    verdict = state.get("verdict")
    retries = state.get("retry_count", 0)

    # 1. If we have a verdict, we are done.
    if verdict:
        return {"next_step": "END"}

    # 2. If no queries, we MUST Analyze first.
    if not queries:
        return {"next_step": "Analyzer"}

    # 3. If we have queries but no evidence, Search.
    if queries and not evidence:
        return {"next_step": "Searcher"}

    # Check if evidence is garbage (e.g., all "No results").
    valid_evidence = [
        e for e in evidence if "No reliable" not in e and "NO_DB_MATCH" not in e
    ]

    if not valid_evidence and retries < 1:
        print("Evidence weak. Rerouting to Analyzer for better queries.")
        # Clear queries to force re-analysis
        return {
            "next_step": "Analyzer",
            "retry_count": retries + 1,
            "queries": [],
            "evidence": [],
        }

    # 5. If we have evidence (or ran out of retries), Verify.
    if not verdict:
        return {"next_step": "Verifier"}

    return {"next_step": "END"}


def node_analyzer(state: AgentState):
    print("Dispatching to MCP Analyzer...")
    try:
        raw_plan = MCPClient.call_analyzer(state["claim"])
        plan = json.loads(raw_plan)
        queries = plan.get("queries", [state["claim"]])
    except Exception as e:
        print(f"Analyzer Failed: {e}")
        queries = [state["claim"]]

    return {"queries": queries, "tool_calls": state.get("tool_calls", 0) + 1}


def node_searcher(state: AgentState):
    print("Dispatching to MCP Searcher...")
    try:
        # Pass the list of queries to the bridge
        evidence = MCPClient.call_searcher(state["queries"])
    except Exception as e:
        print(f"Searcher Failed: {e}")
        evidence = ["Error fetching evidence."]

    return {
        "evidence": evidence,
        "tool_calls": state.get("tool_calls", 0) + len(state["queries"]),
    }


def node_verifier(state: AgentState):
    print("Dispatching to MCP Verifier...")
    ev_text = "\n".join(state["evidence"])

    try:
        raw_verdict = MCPClient.call_verifier(state["claim"], ev_text)
        verdict_data = json.loads(raw_verdict)
        v = verdict_data.get("verdict", "NEI")
        r = verdict_data.get("reason", "No reason provided")
    except Exception as e:
        print(f"Verifier Failed: {e}")
        v, r = "NEI", f"System Error: {e}"

    return {
        "verdict": v,
        "reason": r,
        "tool_calls": state.get("tool_calls", 0) + 2,  # Verdict + Log
    }


# --- GRAPH CONSTRUCTION ---
workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("Supervisor", supervisor_node)
workflow.add_node("Analyzer", node_analyzer)
workflow.add_node("Searcher", node_searcher)
workflow.add_node("Verifier", node_verifier)

# set entry
workflow.set_entry_point("Supervisor")


# Add Conditional Logic (The Router)
def router(state: AgentState):
    return state["next_step"]


workflow.add_conditional_edges(
    "Supervisor",
    router,
    {
        "Analyzer": "Analyzer",
        "Searcher": "Searcher",
        "Verifier": "Verifier",
        "END": END,
    },
)

# Workers always report back to Supervisor
workflow.add_edge("Analyzer", "Supervisor")
workflow.add_edge("Searcher", "Supervisor")
workflow.add_edge("Verifier", "Supervisor")

app = workflow.compile()
