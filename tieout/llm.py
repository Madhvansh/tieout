"""One small client for the free LLM options, with a disk cache.

Providers: Gemini API free tier, Groq free tier, or a local model through
Ollama. Every call asks for JSON and is cached on disk under a hash of
(provider, model, system prompt, user prompt, schema), so an evaluation run
is reproducible and can be replayed offline with no key at all.

Select with TIEOUT_LLM, for example
    TIEOUT_LLM=gemini:gemini-3.5-flash-lite
    TIEOUT_LLM=groq:openai/gpt-oss-120b
    TIEOUT_LLM=ollama:phi4-mini
Keys are read from the environment or from a .env file in the repo root.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import time
from typing import Optional

import requests

ROOT = pathlib.Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache" / "llm"
CALLS = {"live": 0, "cached": 0, "seconds": 0.0}
REPLAY_MODEL = "gemini:gemini-3.5-flash-lite"      # the model the cached benchmark plans were written by


def load_env() -> None:
    p = ROOT / ".env"
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def default_model() -> str:
    load_env()
    if os.environ.get("TIEOUT_LLM"):
        return os.environ["TIEOUT_LLM"]
    if os.environ.get("GEMINI_API_KEY"):
        return "gemini:gemini-3.5-flash-lite"
    if os.environ.get("GROQ_API_KEY"):
        return "groq:openai/gpt-oss-120b"
    return "ollama:qwen3.5:4b"


def ask_json(system: str, user: str, schema: Optional[dict] = None, model: Optional[str] = None,
             tag: str = "", offline: bool = False) -> dict:
    """Return the model's JSON answer. Cached by content; `offline` forbids live calls."""
    # An offline replay must find the cached answers whatever keys this machine has, so it
    # always asks for the model the shipped cache was recorded with.
    load_env()
    model = model or (os.environ.get("TIEOUT_LLM") or REPLAY_MODEL if offline else default_model())
    key = hashlib.sha256(json.dumps([model, system, user, schema], sort_keys=True).encode()).hexdigest()
    path = CACHE / f"{tag + '_' if tag else ''}{key[:24]}.json"
    if path.exists():
        CALLS["cached"] += 1
        return json.loads(path.read_text(encoding="utf-8"))["answer"]
    if offline:
        raise RuntimeError("no cached answer and offline mode is on")
    provider, name = model.split(":", 1)
    t0 = time.time()
    text = {"gemini": _gemini, "groq": _groq, "ollama": _ollama}[provider](name, system, user, schema)
    CALLS["live"] += 1
    CALLS["seconds"] += time.time() - t0
    answer = _parse(text)
    CACHE.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"model": model, "system": system, "user": user, "answer": answer,
                                "seconds": round(time.time() - t0, 2)}, indent=1), encoding="utf-8")
    return answer


def _parse(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.index("{"):]
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("model reply is not JSON")
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError as e:
        raise ValueError(f"model reply is not valid JSON: {e.msg}") from None


def _gemini(name: str, system: str, user: str, schema: Optional[dict]) -> str:
    cfg = {"temperature": 0, "responseMimeType": "application/json"}
    body = {"systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}], "generationConfig": cfg}
    for attempt in range(6):
        r = requests.post(f"https://generativelanguage.googleapis.com/v1beta/models/{name}:generateContent",
                          headers={"x-goog-api-key": os.environ["GEMINI_API_KEY"]}, json=body, timeout=120)
        if r.status_code in (429, 500, 503):
            time.sleep(8 * (attempt + 1))          # free tier: wait out the per-minute limit
            continue
        r.raise_for_status()
        return r.json()["candidates"][0]["content"]["parts"][0]["text"]
    r.raise_for_status()
    return ""


def _groq(name: str, system: str, user: str, schema: Optional[dict]) -> str:
    body = {"model": name, "temperature": 0, "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    for attempt in range(6):
        r = requests.post("https://api.groq.com/openai/v1/chat/completions",
                          headers={"Authorization": f"Bearer {os.environ['GROQ_API_KEY']}"}, json=body, timeout=120)
        if r.status_code in (429, 500, 503):
            time.sleep(8 * (attempt + 1))
            continue
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    r.raise_for_status()
    return ""


def _ollama(name: str, system: str, user: str, schema: Optional[dict]) -> str:
    body = {"model": name, "stream": False, "format": schema or "json", "options": {"temperature": 0, "num_ctx": 8192},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    r = requests.post(os.environ.get("OLLAMA_HOST", "http://localhost:11434") + "/api/chat", json=body, timeout=600)
    r.raise_for_status()
    return r.json()["message"]["content"]
