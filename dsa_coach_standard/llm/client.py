"""
llm/client.py — Unified LLM client

Supports:
  - Ollama  (local, default)   ← root CodeCurry / AURA++ / lumina
  - Gemini  (cloud fallback)   ← lumina uses google-genai

Set LLM_BACKEND env var to "gemini" to switch.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama").lower()
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_PLANNER_MODEL = os.getenv("OLLAMA_PLANNER_MODEL", "qwen2.5-coder:7b")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


# ----------------------------------------------------------------
# Ollama helpers
# ----------------------------------------------------------------

def _ollama_chat(prompt: str, model: str = OLLAMA_MODEL, max_tokens: int = 1200) -> str:
    from ollama import chat
    response = chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"num_predict": max_tokens},
    )
    return response["message"]["content"]


def _ollama_json(prompt: str, model: str = OLLAMA_PLANNER_MODEL) -> str:
    """Call Ollama with JSON format enforced (for planner)."""
    import ollama
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": "You are a strict JSON responder. Return only valid JSON."},
            {"role": "user", "content": prompt},
        ],
        format="json",
        options={"temperature": 0, "num_predict": 120},
    )
    return response["message"]["content"]


# ----------------------------------------------------------------
# Gemini helper
# ----------------------------------------------------------------

def _gemini_chat(prompt: str) -> str:
    from google import genai
    client = genai.Client(api_key=GEMINI_API_KEY)
    response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
    return response.text


# ----------------------------------------------------------------
# Public API
# ----------------------------------------------------------------

def chat(prompt: str, *, json_mode: bool = False, planner: bool = False) -> str:
    """
    Send a prompt to the configured LLM backend.

    Args:
        prompt:    The full prompt string.
        json_mode: If True, instruct the model to return pure JSON.
        planner:   If True, use the smaller/faster planner model (Ollama only).

    Returns:
        Raw text response from the LLM.
    """
    try:
        if LLM_BACKEND == "gemini":
            return _gemini_chat(prompt)
        else:
            if json_mode or planner:
                model = OLLAMA_PLANNER_MODEL if planner else OLLAMA_MODEL
                return _ollama_json(prompt, model=model)
            return _ollama_chat(prompt)
    except Exception as exc:
        print(f"[LLM] Backend '{LLM_BACKEND}' error: {exc}")
        return ""


# ----------------------------------------------------------------
# JSON extraction utility (used across all nodes)
# ----------------------------------------------------------------

def extract_json(text: str) -> dict[str, Any]:
    """Robustly parse JSON from LLM output, stripping markdown fences."""
    if not text:
        return {}
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    cleaned = re.sub(r"```(?:json)?", "", text, flags=re.I).replace("```", "")
    try:
        return json.loads(cleaned.strip())
    except Exception:
        pass
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass
    return {}


def clamp(value: Any, low: int, high: int) -> int:
    """Clamp a numeric value to [low, high]."""
    try:
        return max(low, min(high, int(float(value))))
    except Exception:
        return low
