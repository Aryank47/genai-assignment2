import json
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("AgentVerifier")
llm = ChatOllama(model="qwen2.5:7b", temperature=0, format="json")
LOG_FILE = Path(__file__).parent.parent / "data" / "verification_logs.jsonl"


@mcp.tool()
def determine_verdict(claim: str, evidence: str) -> str:
    """Decides SUPPORTED/REFUTED/NEI based on evidence."""
    prompt = f"""
    Claim: {claim}
    Evidence: {evidence}
    Task: strictly classify as SUPPORTED, REFUTED, or NEI (Not Enough Info).
    Provide a 'reason' based ONLY on the evidence.
    Return JSON: {{ "verdict": "...", "reason": "..." }}
    """
    try:
        res = llm.invoke([HumanMessage(content=prompt)])
        return res.content
    except Exception as e:
        return json.dumps({"verdict": "NEI", "reason": f"LLM Error: {e}"})


@mcp.tool()
def log_verification(claim: str, verdict: str, reason: str) -> str:
    """Writes the decision to a structured JSON log file."""
    record = {
        "claim": claim,
        "verdict": verdict,
        "reason": reason,
        "timestamp": "ISO_TIME_PLACEHOLDER",
    }
    try:
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(record) + "\n")
        return "Logged successfully."
    except Exception as e:
        return f"Logging failed: {e}"


if __name__ == "__main__":
    mcp.run()


