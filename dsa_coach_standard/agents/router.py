"""
agents/router.py — Unified intent router

Strategy (in priority order):
  1. Explicit sidebar `mode` field  ← lumina
  2. LLM classification             ← AURA++
  3. Keyword fallback               ← lumina / root

Intents / agent_types:
  learn       — explain a DSA concept via RAG
  practice    — fetch a problem from the problem bank
  hint        — give a hint for the current problem
  solution    — reveal the full solution
  code_review — analyse student code and give feedback
  direct      — greeting / off-topic (no tool needed)
"""

from agents.state import AgentState
from llm.client import chat

# ----------------------------------------------------------------
# Sidebar mode → intent map  (lumina)
# ----------------------------------------------------------------
_MODE_MAP: dict[str, str] = {
    "learn": "learn",
    "learn dsa": "learn",
    "practice": "practice",
    "hint": "hint",
    "get hint": "hint",
    "solution": "solution",
    "view solution": "solution",
    "code_review": "code_review",
    "code review": "code_review",
    "debug": "code_review",
}

# ----------------------------------------------------------------
# Keyword → intent map  (lumina / root fallback)
# ----------------------------------------------------------------
_KEYWORD_INTENTS: list[tuple[str, list[str]]] = [
    ("code_review", ["review", "debug", "error", "wrong answer", "bug", "optimize",
                     "runtime error", "tle", "my code", "my solution", "why does my code",
                     "compile error", "time complexity of my code"]),
    ("hint",        ["hint", "clue", "stuck", "nudge", "give me a hint"]),
    ("solution",    ["solution", "solve", "answer", "complete code", "full solution"]),
    ("practice",    ["practice", "leetcode", "coding problem", "give me a problem",
                     "easy problem", "medium problem", "hard problem", "dp problem",
                     "array problem", "tree problem"]),
    ("learn",       ["explain", "what is", "how does", "teach me", "concept",
                     "difference between", "complexity", "algorithm"]),
    ("direct",      ["hello", "hi", "hey", "thanks", "thank you"]),
]


def _keyword_intent(question: str) -> str:
    q = question.lower()
    for intent, keywords in _KEYWORD_INTENTS:
        if any(kw in q for kw in keywords):
            return intent
    return "learn"


def _llm_intent(question: str) -> str:
    """Use LLM to classify intent when keyword matching is ambiguous."""
    prompt = f"""You are the routing system for a DSA AI Coach.

Classify the student's message into exactly ONE of these intents:

learn       — student wants to understand a DSA concept or algorithm
practice    — student wants a DSA problem to solve
hint        — student wants a hint (not the full solution)
solution    — student wants the complete solution
code_review — student submitted code for feedback / debug / analysis
direct      — greeting, small talk, or off-topic

Student message:
{question}

Return ONLY one word (the intent). No explanation."""

    raw = chat(prompt).strip().lower()
    valid = {"learn", "practice", "hint", "solution", "code_review", "direct"}
    for intent in valid:
        if intent in raw:
            return intent
    return "learn"


# ----------------------------------------------------------------
# LangGraph node
# ----------------------------------------------------------------

def router_node(state: AgentState) -> AgentState:
    """
    Determines `intent` and `agent_type` for the current request.
    Sets `route` (AURA++ compat) as well.
    """
    question = str(state.get("question", "")).strip()
    mode = str(state.get("mode", "")).lower().strip()

    # 1. Sidebar mode wins
    if mode in _MODE_MAP:
        intent = _MODE_MAP[mode]
        print(f"[ROUTER] Sidebar mode '{mode}' → intent: {intent}")
    else:
        # 2. Keyword check first (fast, free)
        kw_intent = _keyword_intent(question)
        if kw_intent != "learn" or not question:
            intent = kw_intent
            print(f"[ROUTER] Keyword match → intent: {intent}")
        else:
            # 3. LLM classification for ambiguous learn / practice split
            intent = _llm_intent(question)
            print(f"[ROUTER] LLM classified → intent: {intent}")

    # Map intent → agent_type (AgentOps compat)
    agent_type_map = {
        "learn":       "coach",
        "practice":    "coach",
        "hint":        "hint",
        "solution":    "coach",
        "code_review": "code",
        "direct":      "direct",
    }
    agent_type = agent_type_map.get(intent, "coach")

    return {
        **state,
        "intent":     intent,
        "agent_type": agent_type,
        "route":      intent.upper(),   # AURA++ compat field
    }
