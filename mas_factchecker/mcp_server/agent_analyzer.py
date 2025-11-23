import json

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from mcp.server.fastmcp import FastMCP

# Initialize MCP Server
mcp = FastMCP("AgentAnalyzer")
llm = ChatOllama(model="qwen2.5:7b", temperature=0, format="json")


@mcp.tool()
def analyze_claim_structure(claim: str) -> str:
    """
    Analyzes a claim to extract core entities and intent.
    Returns JSON string with 'entities', 'search_queries', and 'complexity'.
    """
    print(f"[Analyzer] Processing: {claim}")
    prompt = f"""
    Analyze this claim: "{claim}"
    1. Extract key entities (people, orgs, events).
    2. Generate 2 distinct search queries: one for strict keyword match, one for semantic context.
    3. Rate complexity (High/Low).
    
    Return JSON: {{ "entities": [], "queries": ["q1", "q2"], "complexity": "..." }}
    """
    try:
        res = llm.invoke(
            [
                SystemMessage(content="You are a structural analyst."),
                HumanMessage(content=prompt),
            ]
        )
        return res.content
    except Exception as e:
        return json.dumps({"error": str(e), "queries": [claim]})


if __name__ == "__main__":
    mcp.run()
