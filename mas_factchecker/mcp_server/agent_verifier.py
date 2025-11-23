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
    You are a PEDANTIC Fact-Checker. You do not accept "close enough".

    Claim: "{claim}"
    Evidence: "{evidence}"

    Task: Classify as SUPPORTED, REFUTED, or NEI.

    CRITICAL VERIFICATION STEPS:
    1. **Date Check**: Does the claim mention a specific year (e.g., 2024)? 
       - If yes, does the evidence support THAT specific year? 
       - If Evidence says 2025 but Claim says 2024 -> REFUTED.
    2. **Role Check**: Does the claim mention a specific title (e.g., President)?
       - If Evidence says Prime Minister -> REFUTED.
    3. **Event Check**: Did the event actually happen?

    VERDICT RULES:
    - **SUPPORTED**: All details (Who, What, Where, **WHEN**) match exactly.
    - **REFUTED**: The event happened, but the *details* (Year, Role, Location) in the claim are wrong.
    - **NEI**: No evidence found.

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