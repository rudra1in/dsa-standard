# DSA Coach — Standard LangGraph Project

> A unified, production-ready DSA coaching agent built by merging the best patterns from four ITER Cohort projects.

## Source Projects Merged

| Source | Key contribution |
|---|---|
| **Root CodeCurry** | Scorecard (Correctness/Efficiency/Readability/Approach), progress persistence, score history |
| **AURA++** | ReAct planner, LLM-based intent router, session memory, problem bank with topic/difficulty filtering |
| **lumina** | Multi-agent specialization (learn/practice/hint/solution/code_review), critic → retry loop, Gemini backend |
| **AgentOps** | Evaluation node, LangGraph checkpointer, annotated conversation history, thread_id |

---

## Architecture

```
START
  ↓
memory_node        ← load score history + conversation history
  ↓
retrieve_node      ← RAG (pgvector / ChromaDB / in-memory keyword)
  ↓
router_node        ← sidebar mode → LLM classification → keyword fallback
  ↓
┌─────────────────────────────────────────┐
│ learn_node       explain DSA concepts   │
│ practice_node    fetch problem from bank│
│ hint_node        staged hints (3 levels)│
│ solution_node    full solution + explain│
│ code_review_node scorecard + feedback   │
│ direct_node      greetings / off-topic  │
└─────────────────────────────────────────┘
  ↓
critic_node        ← quality-check draft_response
  ↓
retry? ──yes──→ router (with critique appended, max 2 retries)
  ↓ no
motivation_node    ← encouragement + avg score
  ↓
final_node         ← assemble final_response
  ↓
END
```

---

## Quick Start

```bash
# 1. Clone / copy this project
cd dsa_coach_standard

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# edit .env — choose LLM_BACKEND and RAG_BACKEND

# 4. Start Ollama (if using local LLM)
ollama pull llama3.2
ollama pull qwen2.5-coder:7b

# 5. Run the Streamlit UI
streamlit run ui/app.py

# Or use the agent directly in Python:
python -c "
from agents.graph import run
result = run('Explain dynamic programming with an example', mode='learn')
print(result['final_response'])
"
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `LLM_BACKEND` | `ollama` | `ollama` or `gemini` |
| `OLLAMA_MODEL` | `llama3.2` | Main chat model |
| `OLLAMA_PLANNER_MODEL` | `qwen2.5-coder:7b` | Planner / router model |
| `GEMINI_API_KEY` | — | Required when `LLM_BACKEND=gemini` |
| `RAG_BACKEND` | `memory` | `memory`, `pgvector`, or `chroma` |
| `DATABASE_URL` | — | PostgreSQL URL for pgvector |

---

## Project Structure

```
dsa_coach_standard/
├── agents/
│   ├── state.py          # Unified AgentState TypedDict
│   ├── router.py         # Intent router (mode → LLM → keyword)
│   ├── nodes.py          # All specialized agent nodes
│   └── graph.py          # LangGraph graph definition + run()
├── llm/
│   └── client.py         # Unified LLM client (Ollama + Gemini)
├── rag/
│   └── retriever.py      # RAG retrieval (pgvector / chroma / memory)
├── memory/
│   └── store.py          # Progress + conversation persistence
├── problems/
│   └── bank.py           # DSA problem bank with 20+ built-in problems
├── ui/
│   └── app.py            # Streamlit UI (chat + code editor + problem view)
├── data/                 # Auto-created: problems.json, progress.json, conversations/
├── .env.example
├── requirements.txt
└── README.md
```

---

## Intents

| Intent | Trigger | Agent node |
|---|---|---|
| `learn` | "explain X", "what is Y", concept questions | `learn_node` |
| `practice` | "give me a problem", topic/difficulty request | `practice_node` |
| `hint` | "hint", "stuck", "clue" | `hint_node` |
| `solution` | "solution", "solve", "answer" | `solution_node` |
| `code_review` | submit code, "debug", "review", "TLE" | `code_review_node` |
| `direct` | greetings, off-topic | `direct_node` |

---

## Scorecard (Code Review)

| Criterion | Max | What it measures |
|---|---|---|
| Correctness | 4 | Does it solve the problem? |
| Efficiency | 3 | Is the time/space complexity optimal? |
| Readability | 2 | Clean code, good variable names |
| Approach | 1 | Sound algorithmic strategy |
| **Total** | **10** | |
