import json
from typing import List, TypedDict

from langgraph.graph import END, StateGraph
# Simulated MCP Client Imports
from mcp_server.agent_analyzer import analyze_claim_structure
from mcp_server.agent_searcher import search_databases, search_web
from mcp_server.agent_verifier import determine_verdict, log_verification


class AgentState(TypedDict):
    claim: str
    queries: List[str]
    evidence: List[str]
    verdict: str
    reason: str
    tool_calls: int


def node_analyzer(state: AgentState):
    print("\n[Orchestrator] Calling Analyzer Agent...")
    current_tools = state.get("tool_calls", 0)

    # Tool Call 1
    raw_plan = analyze_claim_structure(state["claim"])
    current_tools += 1

    try:
        plan = json.loads(raw_plan)
        queries = plan.get("queries", [state["claim"]])
    except json.JSONDecodeError:
        print("JSON Parse Error in Analyzer. Falling back to raw claim.")
        queries = [state["claim"]]

    return {"queries": queries, "tool_calls": current_tools}


def node_searcher(state: AgentState):
    print("[Orchestrator] Calling Searcher Agent...")
    evidence = []
    current_tools = state["tool_calls"]

    for q in state["queries"]:
        # Tool Call: Database
        db_res = search_databases(q)
        current_tools += 1

        if db_res != "NO_DB_MATCH":
            evidence.append(db_res)
        else:
            # Tool Call: Web Fallback
            print(f"   -> DB miss for '{q}', trying Web...")
            web_res = search_web(q)
            current_tools += 1
            evidence.append(f"[Web] {web_res}")

    return {"evidence": evidence, "tool_calls": current_tools}


def node_verifier(state: AgentState):
    print("[Orchestrator] Calling Verifier Agent...")
    ev_text = "\n".join(state["evidence"])
    current_tools = state["tool_calls"]

    # Tool Call: Reasoning
    raw_verdict = determine_verdict(state["claim"], ev_text)
    current_tools += 1

    try:
        verdict_data = json.loads(raw_verdict)
        v = verdict_data.get("verdict", "NEI")
        r = verdict_data.get("reason", "No reason provided")
    except json.JSONDecodeError:
        print("JSON Parse Error in Verifier. Defaulting to NEI.")
        v, r = "NEI", "JSON Parsing Failed"

    # Tool Call: Logging
    log_verification(state["claim"], v, r)
    current_tools += 1

    return {"verdict": v, "reason": r, "tool_calls": current_tools}


# --- GRAPH CONSTRUCTION ---
workflow = StateGraph(AgentState)
workflow.add_node("Analyzer", node_analyzer)
workflow.add_node("Searcher", node_searcher)
workflow.add_node("Verifier", node_verifier)

workflow.set_entry_point("Analyzer")
workflow.add_edge("Analyzer", "Searcher")
workflow.add_edge("Searcher", "Verifier")
workflow.add_edge("Verifier", END)

app = workflow.compile()
