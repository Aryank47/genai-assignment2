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
    
    Task:
    1. Detect and fix typos (e.g., "Shiek" -> "Sheikh", "Ghandi" -> "Gandhi").
    2. Extract key entities.
    3. Generate 2 distinct, SEARCH-ENGINE OPTIMIZED queries.
       
       CRITICAL RULES FOR QUERIES:
       - **NO QUESTIONS**: Do not start with "Who is", "Is there", "What is". Use keywords.
       - **Enforce English**: Queries must be in English.
       - **Time Anchor**: If checking current status (e.g. "is PM"), add the current year (2024 or 2025) to the query.
       - **Context**: 
         - Books/Movies -> add "novel" or "plot" or "author".
         - Politics -> add "news" or "official".
       
       Example:
       Bad: "Who is the PM of Bangladesh?"
       Good: "Bangladesh Prime Minister current 2025 news"
       
       Query 1: Specific Entity + Action keywords.
       Query 2: Broader Fact-Check keywords.

    4. Rate complexity (High/Low).
    
    Return JSON: {{ "entities": [], "queries": ["q1", "q2"], "complexity": "..." }}
    """
    try:
        res = llm.invoke(
            [
                SystemMessage(content="You are a Search Engine Optimization (SEO) expert. You create high-precision keyword queries."),
                HumanMessage(content=prompt),
            ]
        )
        return res.content
    except Exception as e:
        return json.dumps({"error": str(e), "queries": [claim]})


if __name__ == "__main__":
    mcp.run()