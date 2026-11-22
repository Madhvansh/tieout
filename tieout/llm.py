"""One small client for the free LLM options, with a disk cache.

Providers: Gemini API free tier, Groq free tier, or a local model through
Ollama. Every call asks for JSON and is cached on disk under a hash of
(provider, model, system prompt, user prompt, schema), so an evaluation run
is reproducible and can be replayed offline with no key at all.

Select with TIEOUT_LLM, for example
    TIEOUT_LLM=gemini:gemini-3.5-flash-lite
    TIEOUT_LLM=groq:openai/gpt-oss-120b
    TIEOUT_LLM=groq:openai/gpt-oss-120b@low      (reasoning effort for models that reason)
    TIEOUT_LLM=ollama:tieout-planner
Keys are read from the environment or from a .env file in the repo root.
Token counts of every live call are kept with its cached answer, for the cost report.
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
CALLS = {"live": 0, "cached": 0, "seconds": 0.0, "tokens_in": 0, "tokens_out": 0}
CALL_LOG: list = []          # one record per call: provider, model, tag, tokens, seconds, cached
LOCAL_MODEL = "ollama:tieout-planner"              # finetune/MODEL_CARD.md: plans with no key
LOCAL_LINKER = "ollama:tieout-planner-4b"          # the fine-tuned 4B also picks lines among labelled candidates
REPLAY_MODEL = LOCAL_MODEL                         # offline replay: the models the shipped cache was recorded with
REPLAY_LINKER = LOCAL_LINKER
_USAGE: dict = {}                                  # token counts of the last live call


class DailyLimit(RuntimeError):
    """The provider's daily quota is spent. A run must stop here, not record abstentions."""


def load_env() -> None:
    p = ROOT / ".env"
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def link_model(model: Optional[str] = None, offline: bool = False) -> Optional[str]:
    """The model the linker uses: TIEOUT_LINK_LLM if set, else the local 4B linker when the planner is
    local or no key is set (so nothing needs a key), else the planner's model."""
    load_env()
    if os.environ.get("TIEOUT_LINK_LLM"):
        return os.environ["TIEOUT_LINK_LLM"]
    if offline:
        return REPLAY_LINKER
    planner = model or os.environ.get("TIEOUT_LLM") or default_model()
    return LOCAL_LINKER if planner.startswith("ollama:") else planner


def default_model() -> str:
    load_env()
    if os.environ.get("TIEOUT_LLM"):
        return os.environ["TIEOUT_LLM"]
    if os.environ.get("GEMINI_API_KEY"):
        return "gemini:gemini-3.5-flash-lite"
    if os.environ.get("GROQ_API_KEY"):
        return "groq:openai/gpt-oss-120b"
    return LOCAL_MODEL                             # no key anywhere: the fine-tuned local model, on CPU


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
        rec = json.loads(path.read_text(encoding="utf-8"))
        _log(rec.get("model", model), tag, rec.get("usage") or {}, rec.get("seconds", 0.0), cached=True)
        return rec["answer"]
    if offline:
        raise RuntimeError("no cached answer and offline mode is on")
    provider, name = model.split(":", 1)
    t0 = time.time()
    _USAGE.clear()
    text = {"gemini": _gemini, "groq": _groq, "ollama": _ollama}[provider](name, system, user, schema)
    CALLS["live"] += 1
    CALLS["seconds"] += time.time() - t0
    CALLS["tokens_in"] += _USAGE.get("in", 0)
    CALLS["tokens_out"] += _USAGE.get("out", 0)
    _log(model, tag, dict(_USAGE), round(time.time() - t0, 2), cached=False)
    answer = _parse(text)
    CACHE.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"model": model, "system": system, "user": user, "answer": answer,
                                "seconds": round(time.time() - t0, 2), "usage": dict(_USAGE)}, indent=1),
                    encoding="utf-8")
    return answer


def _log(model: str, tag: str, usage: dict, seconds: float, cached: bool) -> None:
    provider = model.split(":", 1)[0]
    CALL_LOG.append({"provider": provider, "model": model, "tag": tag, "prompt_tokens": int(usage.get("in", 0)),
                     "completion_tokens": int(usage.get("out", 0)), "seconds": float(seconds or 0.0),
                     "cached": cached})


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
        last = _last_object(text)                 # a model that reasons aloud ends with its answer
        if last is not None:
            return last
        raise ValueError(f"model reply is not valid JSON: {e.msg}") from None


def _last_object(text: str) -> Optional[dict]:
    """The last complete JSON object in a reply that holds other text or several objects."""
    decoder = json.JSONDecoder()
    found = None
    i = text.find("{")
    while i >= 0:
        try:
            obj, length = decoder.raw_decode(text[i:])
        except json.JSONDecodeError:
            i = text.find("{", i + 1)
            continue
        if isinstance(obj, dict):
            found = obj
        i = text.find("{", i + length)
    return found


def _gemini(name: str, system: str, user: str, schema: Optional[dict]) -> str:
    cfg = {"temperature": 0, "responseMimeType": "application/json"}
    body = {"systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}], "generationConfig": cfg}
    for attempt in range(6):
        r = requests.post(f"https://generativelanguage.googleapis.com/v1beta/models/{name}:generateContent",
                          headers={"x-goog-api-key": os.environ["GEMINI_API_KEY"]}, json=body, timeout=120)
        if r.status_code == 429 and "PerDay" in r.text:
            raise DailyLimit(f"gemini {name}: daily quota spent")
        if r.status_code in (429, 500, 503):
            time.sleep(8 * (attempt + 1))          # free tier: wait out the per-minute limit
            continue
        r.raise_for_status()
        meta = r.json().get("usageMetadata", {})
        _USAGE.update({"in": meta.get("promptTokenCount", 0),
                       "out": meta.get("candidatesTokenCount", 0) + meta.get("thoughtsTokenCount", 0)})
        return r.json()["candidates"][0]["content"]["parts"][0]["text"]
    r.raise_for_status()
    return ""


def _groq(name: str, system: str, user: str, schema: Optional[dict]) -> str:
    name, _, effort = name.partition("@")         # "openai/gpt-oss-120b@low" sets the reasoning effort
    body = {"model": name, "temperature": 0, "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    if effort:
        body["reasoning_effort"] = effort
    for attempt in range(8):
        try:
            r = requests.post("https://api.groq.com/openai/v1/chat/completions",
                              headers={"Authorization": f"Bearer {os.environ['GROQ_API_KEY']}"}, json=body, timeout=120)
        except (requests.ConnectionError, requests.Timeout):
            time.sleep(10 * (attempt + 1))         # a dropped connection is retried like a rate limit
            continue
        if r.status_code == 429 and "per day" in r.text:
            raise DailyLimit(f"groq {name}: daily quota spent")
        if r.status_code == 400 and "failed_generation" in r.text:
            # JSON mode rejected the reply (often a stray character); the reply itself is returned,
            # and _parse keeps it only if it holds a JSON object
            return r.json().get("error", {}).get("failed_generation") or ""
        if r.status_code in (429, 500, 503):
            wait = r.headers.get("retry-after")     # free tier: tokens per minute; the server says how long
            time.sleep(min(90.0, float(wait) + 1.0) if wait and wait.replace(".", "", 1).isdigit()
                       else 8 * (attempt + 1))
            continue
        r.raise_for_status()
        usage = r.json().get("usage", {})
        _USAGE.update({"in": usage.get("prompt_tokens", 0), "out": usage.get("completion_tokens", 0)})
        return r.json()["choices"][0]["message"]["content"]
    raise RuntimeError(f"groq {name}: no answer after 8 attempts")


def _ollama(name: str, system: str, user: str, schema: Optional[dict]) -> str:
    # CPU by default (num_gpu 0): the planner is small enough, and laptops should not need a GPU
    options = {"temperature": 0, "num_ctx": 8192, "num_gpu": int(os.environ.get("OLLAMA_NUM_GPU", "0"))}
    body = {"model": name, "stream": False, "format": schema or "json", "options": options,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    r = requests.post(os.environ.get("OLLAMA_HOST", "http://localhost:11434") + "/api/chat", json=body, timeout=600)
    r.raise_for_status()
    _USAGE.update({"in": r.json().get("prompt_eval_count", 0), "out": r.json().get("eval_count", 0)})
    return r.json()["message"]["content"]
