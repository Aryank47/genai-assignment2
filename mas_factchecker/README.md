# Agentic News Fact-Checker

**A Supervisor-Orchestrated Multi-Agent System that verifies claims using MCP Microservices.**

This system implements a "Agentic" architecture where:

* **Agents are independent processes:** Each agent runs as a distinct FastMCP server.
* **Communication is strict:** The Orchestrator talks to agents only via the Model Context Protocol (MCP) over stdio pipes.
* **Control is dynamic:** A Supervisor LLM decides the workflow (loops, retries, and self-correction) rather than a hardcoded linear chain.

---

## 1. High-Level Overview

### The "Smart" Workflow

Unlike linear pipelines, this system has the capability to self-correct.

**Example Scenario:**
> **Claim:** "Nitish Kumar won the 2024 Bihar Election."

1.  **Supervisor:** Receives the claim. Routes to **Analyzer**.
2.  **Analyzer Agent:** Generates queries (e.g., "Nitish Kumar Bihar election result 2024").
3.  **Searcher Agent:** Checks ISOT/FEVER databases. If missing, checks Web News.
4.  **Self-Correction:** If search results are irrelevant (e.g., gaming cheats or low-quality blogs), the Supervisor detects "weak evidence" and sends the workflow back to the **Analyzer** to generate better queries.
5.  **Verifier Agent:** Compares specific dates (2024 vs 2025). Returns **REFUTED**.

---

## 2. Architecture

### 2.1 Supervisor-Worker Pattern

![Architecture Diagram](./agent_architecture.png)

### 2.2 Microservices (The Agents)

All agents are located in `mcp_servers/` and run as independent Python processes.

| Agent | Core Responsibility | Key Tools |
| :--- | :--- | :--- |
| **Analyzer** | **SEO & Planning**<br>Fixes typos, extracts entities, adds time-anchors (e.g., "2025"). | `analyze_claim_structure` |
| **Searcher** | **Hybrid Retrieval**<br>Checks local DBs first. Falls back to DuckDuckGo News (filtered for English/Quality). | `search_databases`<br>`search_web` |
| **Verifier** | **Pedantic Adjudication**<br>Enforces strict date/role checks (e.g., President vs PM). | `determine_verdict`<br>`log_verification` |

-----

## 3 Setup & Installation

### 3.1 Prerequisites

  * **Python 3.10+**
  * **Ollama** running `qwen2.5:7b`

### 3.2 Installation

```bash
# 1. Create Virtual Environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 2. Install Dependencies (Crucial: mcp, langgraph, duckduckgo-search)
pip install langgraph langchain-ollama langchain-community mcp pandas duckduckgo-search streamlit watchdog
```

-----

## 4 Execution Guide

### 4.1 Data Ingestion

Build the local indexes (ISOT/FEVER) before running agents.

```bash
python data/ingest_data.py
```

### 4.2 Run Evaluation (Headless)

Runs the test suite using the Supervisor architecture without a GUI.

```bash
python eval/run_eval.py
```

### 4.3 Run Interactive UI

Launches the Streamlit dashboard to visualize the Supervisor's decision-making in real-time.

```bash
python -m streamlit run app.py
```

---

## 5 Design Highlights

  * **Process Isolation:** The orchestrator (`agents/graph.py`) uses `agents/mcp_interface.py` to spawn subprocesses. It does not import agent code directly. This satisfies the strictest interpretation of "MCP Client $\rightarrow$ Server".
  * **Pedantic Verification:** The Verifier prompt is engineered to reject "close enough" answers (e.g., matching 2024 to 2025).
  * **Noise Filtering:** The Searcher actively blocks 20+ domains (gaming sites, forums, non-English content) to prevent context pollution.

-----

## 6 Limitations

  * **Latency:** Spawning subprocesses for every tool call adds overhead (\~300ms per call).
  * **Context Window:** The Supervisor passes full evidence history; extremely long investigation chains could eventually hit LLM context limits.