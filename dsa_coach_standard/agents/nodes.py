"""
agents/nodes.py — Specialized agent nodes

Nodes included:
  retrieve_node    — RAG retrieval (optional, graceful fallback)
  learn_node       — Explains a DSA concept                  (lumina: learning_agent)
  practice_node    — Fetches a problem from the bank         (AURA++: call_problem)
  hint_node        — Gives a staged hint                     (lumina: hint_agent)
  solution_node    — Reveals the full solution
  code_review_node — Analyses student code + scorecard       (root CodeCurry + lumina)
  critic_node      — Quality-checks draft_response           (lumina: critic_agent)
  memory_node      — Loads conversation history              (AgentOps: memory_node)
  motivation_node  — Appends encouragement                   (root CodeCurry)
  direct_node      — Handles greetings / off-topic
  final_node       — Assembles final_response
"""

from agents.state import AgentState
from llm.client import chat, extract_json, clamp
from memory.store import load_progress, save_progress


# ================================================================
# RAG RETRIEVAL  (optional — skips gracefully if DB is down)
# ================================================================

def retrieve_node(state: AgentState) -> AgentState:
    """Fetch top-k relevant chunks. Skips silently if RAG is unavailable."""
    query = state.get("question", "")
    try:
        from rag.retriever import retrieve_chunks
        chunks = retrieve_chunks(query, top_k=5)
        knowledge = [c["text"] for c in chunks]
        context = "\n\n".join(knowledge)
    except Exception as exc:
        print(f"[RETRIEVE] RAG skipped: {exc}")
        knowledge = []
        context = ""
    return {**state, "retrieved_knowledge": knowledge, "context": context}


# ================================================================
# MEMORY NODE  (AgentOps pattern — loads history into state)
# ================================================================

def memory_node(state: AgentState) -> AgentState:
    """Load per-session conversation history and progress."""
    session_id = state.get("session_id", "default")
    try:
        from memory.store import load_conversation
        history = load_conversation(session_id, limit=10)
    except Exception:
        history = []
    progress = load_progress()
    return {
        **state,
        "conversation_history": history,
        "score_history":        progress.get("score_history", []),
        "avg_score":            progress.get("avg_score", 0.0),
        "iteration":            state.get("iteration", 0),
        "max_iterations":       state.get("max_iterations", 5),
        "loop_count":           state.get("loop_count", 0),
        "retry_count":          state.get("retry_count", 0),
        "max_retries":          state.get("max_retries", 2),
    }


# ================================================================
# LEARN NODE  — DSA concept explanation via RAG
# ================================================================

def learn_node(state: AgentState) -> AgentState:
    question = state.get("question", "")
    context  = state.get("context", "No relevant context retrieved.")
    history  = _format_history(state.get("conversation_history", []))

    prompt = f"""You are an expert DSA Coach explaining concepts clearly.

Conversation so far:
{history}

Student question:
{question}

Relevant DSA knowledge:
{context}

Instructions:
1. Give a clear, structured explanation.
2. Use simple language with an example if helpful.
3. Mention time/space complexity where relevant.
4. End with a question that encourages the student to think deeper.

Format your response with a brief intro, the explanation, and an example."""

    draft = chat(prompt)
    print("[LEARN] Response generated.")
    return {**state, "draft_response": draft}


# ================================================================
# PRACTICE NODE  — Fetch a problem from the problem bank
# ================================================================

def practice_node(state: AgentState) -> AgentState:
    question   = state.get("question", "").lower()
    topic      = state.get("topic")
    difficulty = state.get("difficulty")

    # Infer topic / difficulty from question if not explicitly set
    if not topic:
        topic = _infer_topic(question)
    if not difficulty:
        difficulty = _infer_difficulty(question)

    try:
        from problems.bank import get_problem
        problem = get_problem(topic=topic, difficulty=difficulty,
                              exclude_ids=_seen_ids(state))
    except Exception as exc:
        print(f"[PRACTICE] Problem bank error: {exc}")
        problem = None

    if not problem:
        draft = ("I couldn't find a matching problem right now. "
                 "Try specifying a topic (e.g. 'give me an easy arrays problem').")
        return {**state, "draft_response": draft}

    draft = (
        f"## Problem: {problem['title']}\n\n"
        f"**Topic:** {problem.get('topic', topic or 'General')} | "
        f"**Difficulty:** {problem.get('difficulty', difficulty or 'Medium')}\n\n"
        f"{problem['description']}\n\n"
        f"*Take your time. Use the Hint button if you get stuck!*"
    )
    print(f"[PRACTICE] Served problem: {problem.get('title')}")
    return {
        **state,
        "problem_id":    problem.get("id", ""),
        "problem":       problem.get("description", ""),
        "draft_response": draft,
    }


# ================================================================
# HINT NODE  — Staged hints without revealing the solution
# ================================================================

def hint_node(state: AgentState) -> AgentState:
    question     = state.get("question", "")
    problem      = state.get("problem", "")
    student_code = state.get("student_code", "")
    context      = state.get("context", "")
    loop_count   = state.get("loop_count", 0)

    # Hint level escalates if the student keeps asking
    hint_level = min(loop_count + 1, 3)
    level_desc = {1: "subtle nudge", 2: "moderate hint", 3: "stronger hint"}[hint_level]

    prompt = f"""You are the Hint Agent of a DSA Coach.

Give a {level_desc} — do NOT reveal the complete solution or full code.

Current problem:
{problem or "Not specified — infer from the student's question."}

Student question / what they're stuck on:
{question}

Student's code (if provided):
{student_code or "No code provided."}

Relevant DSA knowledge:
{context or "None."}

Instructions:
1. Focus on the student's specific confusion.
2. Do not give away the full algorithm or working code.
3. Ask the student to think about the next step.
4. Keep the hint concise and actionable.

Start your response with: ### Hint (Level {hint_level})"""

    draft = chat(prompt)
    print(f"[HINT] Level {hint_level} hint generated.")
    return {**state, "draft_response": draft}


# ================================================================
# SOLUTION NODE  — Full solution with explanation
# ================================================================

def solution_node(state: AgentState) -> AgentState:
    problem  = state.get("problem", "")
    language = state.get("language", "python")
    context  = state.get("context", "")

    prompt = f"""You are a DSA Coach revealing the optimal solution.

Problem:
{problem or "Describe the problem the student is asking about."}

Relevant knowledge:
{context or "None."}

Provide:
1. The complete {language} solution with clear comments.
2. Time and space complexity analysis.
3. Brief explanation of the algorithm chosen.
4. Common edge cases to watch for.

Format with markdown code blocks."""

    draft = chat(prompt)
    print("[SOLUTION] Full solution generated.")
    return {**state, "draft_response": draft}


# ================================================================
# CODE REVIEW NODE  — Analyse code + structured scorecard
# ================================================================

def code_review_node(state: AgentState) -> AgentState:
    """
    Merged from root CodeCurry (scorecard) + lumina code_review_agent.
    Returns both structured scorecard and narrative feedback.
    """
    student_code = state.get("student_code", "")
    problem      = state.get("problem", "")
    language     = state.get("language", "python")
    context      = state.get("context", "")

    if not student_code:
        return {
            **state,
            "draft_response": "## Code Review\n\nPlease paste your code so I can review it.",
        }

    prompt = f"""You are a DSA Code Review Agent.

Analyse the student's code and return ONLY valid JSON with this exact structure:

{{
  "intro": "1-2 sentence friendly summary",
  "analysis": "what the student did and whether the logic is sound",
  "complexity": "time and space complexity with reasoning",
  "evaluation": "edge case assessment and correctness verdict",
  "hint": "one actionable improvement hint",
  "feedback": "2-4 concrete improvement suggestions",
  "encouragement": "short motivating message",
  "scorecard": {{
    "Correctness": 0,
    "Efficiency": 0,
    "Readability": 0,
    "Approach": 0
  }}
}}

Scoring rubric:
  Correctness : 0-4  (does it solve the problem correctly?)
  Efficiency  : 0-3  (is the complexity optimal?)
  Readability : 0-2  (clean code, variable names, comments)
  Approach    : 0-1  (sound algorithm strategy)
  Total       : 0-10

Problem:
{problem or "Infer from the code."}

Language: {language}

Relevant knowledge:
{context or "None."}

Student code:
{student_code}"""

    raw  = chat(prompt, json_mode=True)
    data = extract_json(raw)

    # Parse scorecard with clamping
    sc             = data.get("scorecard", {})
    correctness    = clamp(sc.get("Correctness"), 0, 4)
    efficiency     = clamp(sc.get("Efficiency"),  0, 3)
    readability    = clamp(sc.get("Readability"), 0, 2)
    approach_score = clamp(sc.get("Approach"),    0, 1)
    total          = correctness + efficiency + readability + approach_score

    # Graceful fallback when JSON parsing fails
    if not data:
        has_code     = bool(student_code.strip())
        correctness  = readability = approach_score = efficiency = 1 if has_code else 0
        total        = correctness + efficiency + readability + approach_score
        data = {
            "intro": "Your submission is ready for review.",
            "analysis": "The AI evaluator could not produce structured output. Basic check applied.",
            "complexity": "Run with a working LLM backend for full complexity analysis.",
            "evaluation": "Code detected. Full review requires the LLM to be reachable.",
            "hint": "Check edge cases and verify expected time complexity.",
            "feedback": "Ensure the LLM backend is running for full AI evaluation.",
            "encouragement": "Keep going — every practice session counts!",
        }

    # Persist progress
    history = state.get("score_history", [])
    history.append(total)
    avg_score = sum(history) / len(history)
    save_progress(history, avg_score, state.get("problem_id", ""))

    # Build human-readable draft
    draft = (
        f"{data.get('intro', '')}\n\n"
        f"**Analysis:** {data.get('analysis', '')}\n\n"
        f"**Complexity:** {data.get('complexity', '')}\n\n"
        f"**Correctness verdict:** {data.get('evaluation', '')}\n\n"
        f"**Hint:** {data.get('hint', '')}\n\n"
        f"**Improvements:** {data.get('feedback', '')}\n\n"
        f"---\n"
        f"**Score:** {total}/10  "
        f"(C:{correctness}/4  E:{efficiency}/3  R:{readability}/2  A:{approach_score}/1)\n\n"
        f"*{data.get('encouragement', '')}*"
    )

    print(f"[CODE REVIEW] Score: {total}/10")
    return {
        **state,
        "draft_response": draft,
        "scorecard": {
            "Correctness": correctness,
            "Efficiency":  efficiency,
            "Readability": readability,
            "Approach":    approach_score,
            "Total":       total,
        },
        "score":         total,
        "score_history": history,
        "avg_score":     avg_score,
        "intro":         data.get("intro", ""),
        "analysis":      data.get("analysis", ""),
        "complexity":    data.get("complexity", ""),
        "evaluation":    data.get("evaluation", ""),
        "hint":          data.get("hint", ""),
        "feedback":      data.get("feedback", ""),
        "encouragement": data.get("encouragement", ""),
    }


# ================================================================
# CRITIC NODE  — Quality-check draft_response (lumina / AgentOps)
# ================================================================

def critic_node(state: AgentState) -> AgentState:
    """
    Reviews draft_response for quality. If poor, sets needs_retry=True
    so the graph loops back. Caps at max_retries.
    """
    draft      = state.get("draft_response", "")
    question   = state.get("question", "")
    loop_count = state.get("loop_count", 0)
    max_retries = state.get("max_retries", 2)

    if loop_count >= max_retries:
        print(f"[CRITIC] Max retries ({max_retries}) reached — accepting draft.")
        return {**state, "needs_retry": False, "evaluation": "good"}

    if not draft or len(draft.strip()) < 40:
        print("[CRITIC] Draft too short — requesting retry.")
        return {
            **state,
            "needs_retry": True,
            "evaluation":  "bad",
            "critique":    "The response is too short or empty. Provide a complete, helpful answer.",
        }

    prompt = f"""You are a strict quality reviewer for a DSA Coach.

Student question: {question}

Draft response:
{draft[:1500]}

Evaluate whether the draft:
1. Actually answers the student's question
2. Is accurate (no hallucinated algorithms or wrong complexity)
3. Is helpful and educational

Return ONLY valid JSON:
{{"verdict": "good" or "bad", "critique": "one sentence on what to improve (if bad)"}}"""

    raw  = chat(prompt, json_mode=True)
    data = extract_json(raw)

    verdict  = str(data.get("verdict", "good")).lower()
    critique = str(data.get("critique", ""))

    print(f"[CRITIC] Verdict: {verdict}")
    return {
        **state,
        "needs_retry": verdict == "bad",
        "evaluation":  verdict,
        "critique":    critique,
    }


# ================================================================
# MOTIVATION NODE  (root CodeCurry)
# ================================================================

def motivation_node(state: AgentState) -> AgentState:
    avg = state.get("avg_score", 0.0)
    existing = state.get("encouragement", "")
    if not existing:
        encouragement = f"🚀 Keep going! Your current average score is {avg:.1f}/10."
        return {**state, "encouragement": encouragement}
    return state


# ================================================================
# DIRECT NODE  — Greetings / off-topic
# ================================================================

def direct_node(state: AgentState) -> AgentState:
    return {
        **state,
        "draft_response": (
            "Hi! I'm your DSA Coach 👋 I can help you:\n\n"
            "- **Learn** a DSA concept\n"
            "- **Practice** with problems\n"
            "- **Get hints** when you're stuck\n"
            "- **Review your code** for correctness and efficiency\n\n"
            "What would you like to work on today?"
        ),
    }


# ================================================================
# FINAL NODE  — Assembles final_response
# ================================================================

def final_node(state: AgentState) -> AgentState:
    final = state.get("draft_response", "") or state.get("answer", "")
    if not final:
        final = "I wasn't able to generate a response. Please try again."
    return {**state, "final_response": final, "answer": final, "done": True}


# ================================================================
# Internal helpers
# ================================================================

def _format_history(history: list[dict]) -> str:
    if not history:
        return "No previous conversation."
    lines = []
    for msg in history[-6:]:
        role    = msg.get("role", "user")
        content = msg.get("content", "")
        lines.append(f"{role.capitalize()}: {content}")
    return "\n".join(lines)


def _infer_topic(question: str) -> str | None:
    mapping = {
        "dynamic programming": "dynamic_programming", "dp": "dynamic_programming",
        "array": "arrays", "string": "strings", "linked list": "linked_list",
        "tree": "trees", "graph": "graphs", "stack": "stacks", "queue": "queues",
        "binary search": "binary_search", "greedy": "greedy",
        "backtracking": "backtracking", "heap": "heaps", "trie": "tries",
    }
    for kw, topic in mapping.items():
        if kw in question:
            return topic
    return None


def _infer_difficulty(question: str) -> str | None:
    for d in ("easy", "medium", "hard"):
        if d in question:
            return d
    return None


def _seen_ids(state: AgentState) -> set[str]:
    history = state.get("conversation_history", [])
    return {m.get("problem_id") for m in history if m.get("problem_id")}
