---
name: langgraph-local-deploy
description: >
  Step-by-step guide for building and deploying a LangGraph agent project on a local machine —
  from project scaffolding, dependency installation, and graph compilation through running the
  LangGraph dev server, LangGraph Studio, and a FastAPI/Streamlit frontend locally.
  Use this skill whenever the user wants to: run LangGraph locally, set up a LangGraph project
  from scratch, start the LangGraph dev server, connect LangGraph Studio to a local agent,
  deploy a LangGraph app on their own machine, debug a local LangGraph graph, configure
  environment variables for LangGraph, or wire a Streamlit/FastAPI UI to a local LangGraph graph.
  Trigger even when the user just says "run my agent locally", "how do I start LangGraph",
  "set up LangGraph on my laptop", or similar — this skill applies broadly to any local
  LangGraph development and deployment workflow.
---

# LangGraph — Local Build & Deploy

This skill covers the full local lifecycle:

1. [Project scaffold & dependencies](#1-project-scaffold--dependencies)
2. [Graph definition checklist](#2-graph-definition-checklist)
3. [Environment variables](#3-environment-variables)
4. [Run modes (choose one)](#4-run-modes)
   - 4a. Python script / REPL
   - 4b. LangGraph dev server (`langgraph dev`)
   - 4c. LangGraph Studio (desktop GUI)
   - 4d. FastAPI wrapper
   - 4e. Streamlit UI
5. [Checkpointer & persistence](#5-checkpointer--persistence)
6. [Common errors & fixes](#6-common-errors--fixes)

For Docker-based local deployment see → `references/docker.md`
For LangGraph Cloud / remote deployment see → `references/cloud.md`

---

## 1. Project scaffold & dependencies

### Recommended layout

```
my_agent/
├── agent/
│   ├── __init__.py
│   ├── graph.py        ← compiled graph lives here
│   ├── nodes.py
│   ├── state.py
│   └── router.py
├── memory/
│   └── store.py
├── rag/
│   └── retriever.py
├── ui/
│   └── app.py          ← Streamlit or FastAPI
├── data/               ← auto-created at runtime
├── .env
├── langgraph.json      ← required for `langgraph dev` and Studio
├── requirements.txt
└── README.md
```

### Install

```bash
# Create and activate a virtual environment (strongly recommended)
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# Core LangGraph stack
pip install langgraph langchain-core

# LangGraph CLI (needed for dev server + Studio)
pip install langgraph-cli

# LLM backend — pick one or both
pip install ollama                 # local Ollama
pip install google-genai           # Gemini

# Optional — persistence / checkpointing
pip install langgraph-checkpoint-sqlite   # SQLite (easiest local option)
pip install langgraph-checkpoint-postgres # PostgreSQL

# Optional — RAG
pip install chromadb sentence-transformers

# Optional — UI
pip install streamlit fastapi uvicorn
```

Freeze after you're happy:
```bash
pip freeze > requirements.txt
```

---

## 2. Graph definition checklist

Every LangGraph project needs these three things before it can run:

### 2a. State (`agent/state.py`)
```python
from typing import TypedDict, List, Optional

class AgentState(TypedDict, total=False):
    question: str
    session_id: str
    intent: str
    context: str
    draft_response: str
    final_response: str
    # … add your fields
```

### 2b. Nodes (`agent/nodes.py`)
Each node is a plain Python function: `(state: AgentState) -> AgentState`

```python
def my_node(state: AgentState) -> AgentState:
    # do work
    return {**state, "draft_response": "result"}
```

### 2c. Compiled graph (`agent/graph.py`)

```python
from langgraph.graph import StateGraph, START, END
from agent.state import AgentState
from agent.nodes import my_node, final_node

def build_graph():
    g = StateGraph(AgentState)
    g.add_node("my_node", my_node)
    g.add_node("final",   final_node)
    g.add_edge(START,      "my_node")
    g.add_edge("my_node",  "final")
    g.add_edge("final",    END)
    return g.compile()

graph = build_graph()          # module-level — required for langgraph.json
```

**The module-level `graph` variable is required** when using `langgraph dev` or Studio.

---

## 3. Environment variables

Create a `.env` file in the project root (never commit this):

```dotenv
# LLM backend
LLM_BACKEND=ollama              # or "gemini"
OLLAMA_MODEL=llama3.2
GEMINI_API_KEY=your_key_here

# RAG backend
RAG_BACKEND=memory              # or "chroma" / "pgvector"
DATABASE_URL=postgresql://localhost/my_agent

# LangSmith tracing (optional but highly recommended for debugging)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_PROJECT=my-agent-local

# LangGraph dev server
LANGGRAPH_HOST=127.0.0.1
LANGGRAPH_PORT=8123
```

Load in Python:
```python
from dotenv import load_dotenv
load_dotenv()   # call once at app entry point
```

Install `python-dotenv` if not already present: `pip install python-dotenv`

---

## 4. Run modes

### 4a. Python script / REPL (simplest)

```python
# run.py
from dotenv import load_dotenv
load_dotenv()

from agent.graph import graph

result = graph.invoke({
    "question": "Explain binary search",
    "session_id": "test-001",
})
print(result["final_response"])
```

```bash
python run.py
```

### 4b. LangGraph dev server (`langgraph dev`)

The dev server exposes your graph over HTTP and enables hot-reload.

**Step 1 — create `langgraph.json`** in the project root:

```json
{
  "dependencies": ["."],
  "graphs": {
    "agent": "./agent/graph.py:graph"
  },
  "env": ".env"
}
```

- `"agent"` = the name shown in Studio (can be anything)
- `"./agent/graph.py:graph"` = path to file, colon, variable name of the compiled graph

**Step 2 — start the server:**

```bash
langgraph dev
```

Or with explicit options:
```bash
langgraph dev --port 8123 --host 127.0.0.1 --reload
```

Server starts at `http://127.0.0.1:8123`. You'll see:
```
✓ LangGraph API server started
✓ Graph loaded: agent
  API: http://127.0.0.1:8123
  Docs: http://127.0.0.1:8123/docs
```

**Call via HTTP:**
```bash
curl -X POST http://127.0.0.1:8123/runs/stream \
  -H "Content-Type: application/json" \
  -d '{"assistant_id":"agent","input":{"question":"Hello"}}'
```

Or via the Python SDK:
```python
from langgraph_sdk import get_client

client = get_client(url="http://127.0.0.1:8123")
async for chunk in client.runs.stream(
    None, "agent",
    input={"question": "Hello"},
    stream_mode="values",
):
    print(chunk.data)
```

### 4c. LangGraph Studio (desktop GUI)

Studio gives you a visual graph explorer, live state inspector, and time-travel debugging.

**Prerequisites:**
- `langgraph dev` must be running (Studio connects to it)
- macOS: download [LangGraph Studio desktop app](https://studio.langchain.com)
- Windows/Linux: use the web Studio at `https://smith.langchain.com/studio` pointing at your local server URL

**Connect Studio:**
1. Open LangGraph Studio
2. Click "Open existing project"
3. Select your project folder (the one containing `langgraph.json`)
4. Studio auto-discovers the dev server and loads your graph

**What you can do in Studio:**
- Visualise the graph as a flowchart
- Send inputs and watch state flow node-by-node
- Inspect state at each checkpoint
- Time-travel: replay from any past checkpoint
- Edit state mid-run and re-run from that point

### 4d. FastAPI wrapper

Expose your graph as a REST API for integration with other services:

```python
# ui/api.py
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()

from agent.graph import graph

app = FastAPI(title="My Agent API")

class Request(BaseModel):
    question: str
    session_id: str = "default"
    language: str = "python"

class Response(BaseModel):
    answer: str
    session_id: str

@app.post("/ask", response_model=Response)
def ask(req: Request):
    result = graph.invoke({
        "question":   req.question,
        "session_id": req.session_id,
        "language":   req.language,
    })
    return Response(
        answer=result.get("final_response", ""),
        session_id=req.session_id,
    )

@app.get("/health")
def health():
    return {"status": "ok"}
```

```bash
uvicorn ui.api:app --reload --host 0.0.0.0 --port 8000
```

Interactive docs at `http://localhost:8000/docs`

### 4e. Streamlit UI

```bash
streamlit run ui/app.py
```

Opens at `http://localhost:8501`

Key Streamlit pattern for LangGraph:
```python
import streamlit as st
from agent.graph import graph

if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.chat_input("Ask your agent…")
if user_input:
    with st.spinner("Thinking…"):
        result = graph.invoke({
            "question":   user_input,
            "session_id": st.session_state.get("session_id", "default"),
        })
    response = result.get("final_response", "")
    st.session_state.history.append(("user", user_input))
    st.session_state.history.append(("assistant", response))

for role, msg in st.session_state.history:
    with st.chat_message(role):
        st.markdown(msg)
```

---

## 5. Checkpointer & persistence

Without a checkpointer, every `graph.invoke()` starts fresh. Add one to enable:
- Multi-turn conversations
- Time-travel debugging in Studio
- State persistence across restarts

### SQLite (recommended for local dev — zero setup)

```python
from langgraph.checkpoint.sqlite import SqliteSaver

with SqliteSaver.from_conn_string("data/checkpoints.db") as checkpointer:
    graph = build_graph().compile(checkpointer=checkpointer)
```

Or for long-running servers, use a context-manager-free form:

```python
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

conn = sqlite3.connect("data/checkpoints.db", check_same_thread=False)
checkpointer = SqliteSaver(conn)
graph = build_graph().compile(checkpointer=checkpointer)
```

### In-memory (dev/testing only — resets on restart)

```python
from langgraph.checkpoint.memory import MemorySaver
checkpointer = MemorySaver()
graph = build_graph().compile(checkpointer=checkpointer)
```

### PostgreSQL (production-grade local)

```bash
pip install langgraph-checkpoint-postgres psycopg[binary]
```

```python
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg import Connection

conn = Connection.connect("postgresql://user:pass@localhost/mydb", autocommit=True)
checkpointer = PostgresSaver(conn)
checkpointer.setup()   # creates tables on first run
graph = build_graph().compile(checkpointer=checkpointer)
```

### Using thread_id for multi-turn sessions

Pass a `config` dict with `thread_id` on each invoke:

```python
config = {"configurable": {"thread_id": "user-abc-session-1"}}
result = graph.invoke({"question": "Hello"}, config=config)

# Next turn — graph loads previous state automatically
result2 = graph.invoke({"question": "Follow up"}, config=config)
```

---

## 6. Common errors & fixes

| Error | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: langgraph` | Not installed in active venv | `pip install langgraph` in the correct venv |
| `No module named 'agent'` | Running from wrong directory | `cd` to project root before running |
| `langgraph: command not found` | CLI not installed | `pip install langgraph-cli` |
| `Graph not found: agent` | `langgraph.json` path wrong | Check the `graphs` key path and variable name |
| `Port 8123 already in use` | Another process has the port | `lsof -i :8123` then kill, or `--port 8124` |
| Checkpointer `thread_id` missing | No config passed to invoke | Add `config={"configurable":{"thread_id":"…"}}` |
| Studio shows empty graph | No module-level `graph` variable | Ensure `graph = build_graph()` at module level |
| `LANGCHAIN_API_KEY not set` | Tracing enabled but no key | Either set the key or `LANGCHAIN_TRACING_V2=false` |
| SQLite `check_same_thread` error | Using SQLite in FastAPI threads | Pass `check_same_thread=False` to `sqlite3.connect()` |
| Ollama `connection refused` | Ollama not running | `ollama serve` in a separate terminal |

---

## Quick-start checklist

```
□ python -m venv .venv && source .venv/bin/activate
□ pip install langgraph langgraph-cli langchain-core
□ Create agent/state.py, agent/nodes.py, agent/graph.py
□ Add module-level `graph = build_graph()` to graph.py
□ Create .env with LLM keys
□ Create langgraph.json pointing to graph.py:graph
□ langgraph dev --reload
□ Open LangGraph Studio → select project folder
□ Test with curl or the SDK
□ Add SQLite checkpointer for persistence

**Parallel multi-agent version** (the practical approximation of Group Think in LangGraph) means:

Instead of one agent thinking step-by-step, you run **several specialized agents at the same time** on the same problem. Each agent focuses on a different aspect, then a final agent combines their outputs into a single coherent response for the student.

### How it works (high-level)

```
Student problem + current progress
              │
              ▼
        Planner / Router
              │
    ┌─────────┼─────────┬─────────┬─────────┐
    ▼         ▼         ▼         ▼         ▼
 Pattern   Complexity  Edge-case  Pedagogy  Code
 Thinker    Thinker     Thinker   Thinker  Structure
    │         │         │         │         │
    └─────────┼─────────┴─────────┴─────────┘
              │
              ▼
         Synthesizer / Coach
              │
              ▼
     Progressive hint / next step
     (shown to the student)
```

### Key characteristics

- **Parallel execution**: The specialized thinkers run concurrently (LangGraph fan-out with `Send` or parallel edges). This reduces total latency compared to sequential agents.
- **Specialization**: Each agent has its own system prompt and role. Examples for a DSA Step Coach:
  - **Pattern Thinker**: Identifies the core algorithm/pattern (two pointers, sliding window, DFS, DP, etc.).
  - **Complexity Thinker**: Analyzes time/space trade-offs and optimal approaches.
  - **Edge-case Thinker**: Finds tricky inputs, boundary conditions, and potential bugs.
  - **Pedagogy Thinker**: Decides how much to reveal based on student level (beginner → advanced) and forces progressive hints only.
  - **Code Structure Thinker**: Suggests clean function signatures, helper functions, or skeleton code without giving the full solution.
- **Shared state**: All agents can read the problem, student’s previous attempts, and current conversation. After they finish, their outputs are merged.
- **Synthesizer (Coach)**: The final node that reads all the parallel thoughts and produces one student-facing message. It decides the next teaching step, what to hint, what to withhold, and the tone.

### Why this is useful for a DSA Step Coach

- Faster and richer reasoning than a single agent.
- Better teaching quality (one agent focuses purely on not spoiling the answer).
- Easier to control and debug than true token-level Group Think.
- Scales well: you can add or remove thinkers dynamically based on the problem difficulty or student level.

### Difference from original Group Think

| Feature                    | Original Group Think              | Parallel Multi-Agent (LangGraph)      |
|---------------------------|-----------------------------------|---------------------------------------|
| Collaboration granularity | Token-by-token (mid-generation)   | After full agent responses            |
| Implementation            | Model/inference modification      | Standard graph of LLM calls           |
| Latency                   | Extremely low                     | Low (but higher than pure token-level)|
| Specialization            | Same model, different threads     | Different prompts/tools per agent     |
| Ease of use               | Hard                              | Easy                                  |

In short: the **parallel multi-agent version** is “multiple expert agents thinking at the same time + one coach that combines their insights.” It is the realistic, production-friendly way to get collaborative reasoning benefits inside a LangGraph-based DSA Step Coach.
```