import json
import re
import warnings
from pathlib import Path

import pandas as pd
from duckduckgo_search import DDGS
from mcp.server.fastmcp import FastMCP

# Suppress library warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="duckduckgo_search")

mcp = FastMCP("AgentSearcher")

# --- DATA LOADING ---
DATA_DIR = Path(__file__).parent.parent / "data"
df_isot = pd.DataFrame()
fever_db = []

try:
    if (DATA_DIR / "isot_index.csv").exists():
        df_isot = pd.read_csv(DATA_DIR / "isot_index.csv")
        df_isot["title_lower"] = df_isot["title"].str.lower()

    if (DATA_DIR / "fever_index.jsonl").exists():
        with open(DATA_DIR / "fever_index.jsonl", "r") as f:
            fever_db = [json.loads(line) for line in f]
except Exception as e:
    print(f"Warning: Data load failed - {e}")


# --- QUALITY CONTROL UTILS ---

def _is_valid_text(text: str) -> bool:
    """
    Heuristic to reject non-English text (e.g., Chinese, French results).
    Checks if a significant portion of the text is non-ASCII or contains CJK characters.
    """
    if not text: 
        return False
        
    # 1. Reject if contains Chinese/Japanese/Korean (CJK) characters
    if re.search(r'[\u4e00-\u9fff]', text):
        return False
        
    # 2. Reject if contains Cyrillic (Russian)
    if re.search(r'[\u0400-\u04FF]', text):
        return False
        
    return True

def _is_high_quality(result: dict) -> bool:
    """Filter out known spam, gaming, and dev forums."""
    title = result.get("title", "").lower()
    snippet = result.get("body", "").lower()
    url = result.get("href", "").lower()
    
    # 1. Check Language Integrity
    if not _is_valid_text(title) or not _is_valid_text(snippet):
        return False

    # 2. Domain/Topic Blocklist
    junk_terms = [
        # Gaming / Spam
        "cheat", "aimbot", "hack", "esp", "spoofer", "wz bo6", "minecraft", "roblox",
        # Non-News / Dev Forums (Garbage for fact checking)
        "zhihu", "baidu", "csdn", "hinative", "stackexchange", "stackoverflow", "github",
        "jeuxvideo", "bilibili",
        # Paywalls / Irrelevant
        "login", "signup", "subscribe", "403 forbidden", "access denied"
    ]
    
    if any(term in url or term in title for term in junk_terms):
        return False

    return True

# --- MCP TOOLS ---

@mcp.tool()
def search_databases(query: str) -> str:
    """Searches local ISOT and FEVER databases."""
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
    """
    Performs a live NEWS search using DuckDuckGo.
    Prioritizes the 'News' tab to avoid forums and unrelated sites.
    """
    results = []
    try:
        with DDGS() as ddgs:
            # STRATEGY 1: Try News Search first (High Precision)
            # max_results=5 to allow for filtering
            news_gen = ddgs.news(query, region="us-en", safesearch="moderate", max_results=5)
            
            for r in news_gen:
                if _is_high_quality(r):
                    # News results use 'date' instead of 'href' sometimes, normalizing:
                    title = r.get('title', 'No Title')
                    snippet = r.get('body', 'No Snippet')
                    source = r.get('url', r.get('href', 'No Link'))
                    date = r.get('date', 'Unknown Date')
                    
                    results.append(f"Source: {title}\nDate: {date}\nSnippet: {snippet}\nLink: {source}")
            
            # STRATEGY 2: Fallback to Text Search if News returns nothing (e.g., obscure history)
            if not results:
                text_gen = ddgs.text(query, region="us-en", safesearch="moderate", max_results=5)
                for r in text_gen:
                    if _is_high_quality(r):
                        title = r.get('title', 'No Title')
                        snippet = r.get('body', 'No Snippet')
                        source = r.get('href', 'No Link')
                        results.append(f"Source: {title}\nSnippet: {snippet}\nLink: {source}")

            # Return top 3 filtered results
            final_results = results[:3]
            
            if not final_results:
                return "No reliable English news results found."
            
            return "\n\n".join(final_results)
        
    except Exception as e:
        print(f"Web Search Error for '{query}': {e}")
        return "Web search unavailable due to network error."


if __name__ == "__main__":
    mcp.run()
    