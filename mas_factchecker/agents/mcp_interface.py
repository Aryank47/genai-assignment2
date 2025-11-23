import asyncio
import json
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Path to current python interpreter
PYTHON_PATH = sys.executable


class MCPClient:

    @staticmethod
    async def _call_tool(script_path: str, tool_name: str, args: dict):
        server_params = StdioServerParameters(
            command=PYTHON_PATH, args=[script_path], env=None
        )

        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, arguments=args)
                if not result.content:
                    raise ValueError(f"Tool {tool_name} returned empty result")
                return result.content[0].text

    @classmethod
    def call_analyzer(cls, claim: str):
        return asyncio.run(
            cls._call_tool(
                "mcp_server/agent_analyzer.py",
                "analyze_claim_structure",
                {"claim": claim},
            )
        )

    @classmethod
    def call_searcher(cls, queries: list):
        # The searcher agent expects a single query string, but we have a list.
        # We handle the loop here or in the agent. Let's do it here for granularity.
        results = []
        for q in queries:
            # First try DB
            db_res = asyncio.run(
                cls._call_tool(
                    "mcp_server/agent_searcher.py", "search_databases", {"query": q}
                )
            )
            if db_res != "NO_DB_MATCH":
                results.append(db_res)
            else:
                # Fallback to Web
                web_res = asyncio.run(
                    cls._call_tool(
                        "mcp_server/agent_searcher.py", "search_web", {"query": q}
                    )
                )
                results.append(f"[Web] {web_res}")
        return results

    @classmethod
    def call_verifier(cls, claim: str, evidence: str):
        # 1. Get Verdict
        verdict_json = asyncio.run(
            cls._call_tool(
                "mcp_server/agent_verifier.py",
                "determine_verdict",
                {"claim": claim, "evidence": evidence},
            )
        )

        # 2. Log it (Side effect)
        try:
            v_data = json.loads(verdict_json)
            asyncio.run(
                cls._call_tool(
                    "mcp_server/agent_verifier.py",
                    "log_verification",
                    {
                        "claim": claim,
                        "verdict": v_data.get("verdict"),
                        "reason": v_data.get("reason"),
                    },
                )
            )
        except Exception as e:
            print(f"Logging Failed: {e}")

        return verdict_json
