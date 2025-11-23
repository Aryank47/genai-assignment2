import json
from pathlib import Path

import pandas as pd
from langchain_community.tools import DuckDuckGoSearchRun
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("AgentSearcher")
search_tool = DuckDuckGoSearchRun()

# Load Data
DATA_DIR = Path(__file__).parent.parent / "data"
df_isot = pd.DataFrame()
fever_db = []

# Optimistic Loading
try:
    if (DATA_DIR / "isot_index.csv").exists():
        df_isot = pd.read_csv(DATA_DIR / "isot_index.csv")
        df_isot["title_lower"] = df_isot["title"].str.lower()

    if (DATA_DIR / "fever_index.jsonl").exists():
        with open(DATA_DIR / "fever_index.jsonl", "r") as f:
            fever_db = [json.loads(line) for line in f]
except Exception as e:
    print(f"Warning: Data load failed - {e}")


@mcp.tool()
def search_databases(query: str) -> str:
    """Searches local ISOT (Fake News) and FEVER (Fact) databases."""
    query_lower = query.lower()
    evidence = []

    # 1. ISOT Search
    if not df_isot.empty:
        matches = df_isot[
            df_isot["title_lower"].str.contains(query_lower, na=False, regex=False)
        ]
        for _, row in matches.head(2).iterrows():
            evidence.append(
                f"[ISOT DB] Headline: '{row['title']}' | Label: {row['label']}"
            )

    # 2. FEVER Search
    count = 0
    for entry in fever_db:
        if query_lower in entry["claim"].lower():
            evidence.append(
                f"[FEVER DB] Claim: '{entry['claim']}' | Label: {entry['label']}"
            )
            count += 1
            if count >= 2:
                break

    if not evidence:
        return "NO_DB_MATCH"
    return "\n".join(evidence)


@mcp.tool()
def search_web(query: str) -> str:
    """Performs a live web search using DuckDuckGo."""
    try:
        return search_tool.invoke(query)
    except:
        return "Web search failed."


if __name__ == "__main__":
    mcp.run()
