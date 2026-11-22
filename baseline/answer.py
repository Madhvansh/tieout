"""Conventional RAG answerer: retrieve, hand the text to the LLM, let it read and compute.

Two retrievers. v2 (default) is baseline/hybrid.py: passages, BM25 + dense, company-aware, at most
16,000 characters of filing text per prompt. v1 (--retriever v1) is baseline/index.py: whole pages,
BM25 + TF-IDF, as described below.

Choices
  k            8 pages by default.
  page budget  2,400 characters of page body per page (plus its header line), so 8 pages are
               about 19,000 characters, roughly 5,000 to 6,500 tokens of filing text. That fits a
               single Groq free-tier request (about 8,000 tokens a minute on the large models),
               Gemini's free tier easily, and the 8,192-token context tieout/llm.py gives Ollama,
               with room left for the instructions and the answer.
  trimming     a page longer than the budget keeps one contiguous window of whole lines: the
               window with the most question-word hits (distinct non-stop-word question tokens per
               line, summed). This is ordinary passage windowing; the first 2,400 characters of a
               page would often cut off the table the page was retrieved for.
  LLM          tieout.llm.ask_json only, tag "baseline", so provider, model and cache are shared
               with the graph system. One call per question, temperature 0 (set in llm.py).
               If the reply is not valid JSON the call is made once more, then the answer abstains.
"""
from __future__ import annotations

import re
import time

from baseline import index

K = 8
PAGE_BUDGET = 2400

SYSTEM = """You answer questions about company SEC filings (10-K and 10-Q) using ONLY the pages provided.
Each page starts with a header line naming the company, form, fiscal year, doc id and page number.
Read the numbers you need from the pages and do any arithmetic yourself.

Reply with one JSON object and nothing else:
{"answer": short answer as a string,
 "value": the answer as a number, a string, a list of company names, or null,
 "unit": one of "USD", "ratio", "shares", "count", "text", "company", "USD/share", "none",
 "abstain": true or false,
 "facts_used": [{"doc": doc id from the page header, "page": page number from the header,
                 "value": the number as a plain number in full units, "label": the line item}]}

Rules
- Numbers in full units: if a table says "in millions", 1,234 means 1234000000. No commas, no symbols.
- Ratios, margins, growth rates and percentages as fractions: 12.5% is 0.125, unit "ratio".
- Per-share amounts: unit "USD/share", value in dollars per share.
- Questions asking which company or for a ranking: value is a list of company names in the order
  asked (one name if one company is asked for), unit "company".
- Names, places and other words: value is a string, unit "text".
- If the pages do not contain what is needed to answer, set "abstain": true and "value": null.
  Do not guess and do not use outside knowledge.
- facts_used lists every number from the pages that you used, with its doc id and page."""

_STOP = None


def _qtokens(question: str) -> set:
    global _STOP
    if _STOP is None:
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        _STOP = set(ENGLISH_STOP_WORDS)
    return {t for t in index.tokenize(question) if t not in _STOP and len(t) > 1}


def trim(text: str, question: str, budget: int = PAGE_BUDGET) -> str:
    header, _, body = text.partition("\n")
    if len(body) <= budget:
        return text
    lines = body.split("\n")
    q = _qtokens(question)
    hits = [len(q & set(index.tokenize(ln))) for ln in lines]
    best, best_score = (0, 0), -1
    for s in range(len(lines)):
        size, score, e = 0, 0, s
        while e < len(lines) and size + len(lines[e]) + 1 <= budget:
            size += len(lines[e]) + 1
            score += hits[e]
            e += 1
        if score > best_score:
            best, best_score = (s, e), score
        if e == len(lines):
            break
    s, e = best
    if e == s:                                   # one line longer than the budget
        return header + "\n" + lines[s][:budget]
    return header + "\n" + ("[...]\n" if s else "") + "\n".join(lines[s:e]) + ("\n[...]" if e < len(lines) else "")


_NUM = re.compile(r"^\(?-?\$?\s*[\d,]*\.?\d+\)?$")
_SCALE = {"thousand": 1e3, "million": 1e6, "billion": 1e9, "trillion": 1e12}


def to_number(v):
    """Plain number from what the LLM wrote, or None. Parsing only, no guessing."""
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().lower().replace("usd", "").strip()
    mult = 1.0
    for w, m in _SCALE.items():
        if s.endswith(w):
            s, mult = s[: -len(w)].strip(), m
    if s.endswith("%"):
        s, mult = s[:-1].strip(), 0.01
    if not _NUM.match(s):
        return None
    neg = s.startswith("(") or s.startswith("-")
    x = float(s.strip("()").replace("$", "").replace(",", "").replace("-", "").strip()) * mult
    return -x if neg else x


def to_pred(reply: dict) -> dict:
    """Map the LLM's reply to the {"kind", "value"} shape bench.is_correct expects.
    Company names are left as written; run_baseline maps them to tickers for scoring."""
    v, unit = reply.get("value"), str(reply.get("unit") or "").lower()
    if reply.get("abstain") is True or v is None:
        return {"kind": "abstain", "value": None}
    if isinstance(v, list) or unit == "company":
        names = v if isinstance(v, list) else [v]
        names = [str(x.get("name", x) if isinstance(x, dict) else x).strip() for x in names]
        names = [n for n in names if n]
        return {"kind": "companies", "value": [[n, None] for n in names]} if names else {"kind": "abstain", "value": None}
    if unit not in ("text",):
        x = to_number(v)
        if x is not None:
            return {"kind": "number", "value": x}
    s = str(v).strip()
    if not s:
        return {"kind": "abstain", "value": None}
    return {"kind": "text", "value": s}


def _facts(reply: dict, allowed: set) -> list:
    out = []
    for f in reply.get("facts_used") or []:
        if not isinstance(f, dict):
            continue
        doc = str(f.get("doc", "")).strip()
        x = to_number(f.get("value"))
        if x is None:
            continue
        try:
            page = int(str(f.get("page")).strip().lower().lstrip("p").replace("page", "").strip())
        except ValueError:
            page = None
        out.append({"ticker": doc.split("_")[0], "doc": doc, "page": page, "value": x,
                    "label": str(f.get("label", ""))[:120], "in_context": (doc, page) in allowed})
    return out


PASSAGE_SYSTEM = SYSTEM.replace("using ONLY the pages provided", "using ONLY the passages provided").replace(
    "Each page starts with a header line naming the company, form, fiscal year, doc id and page number.",
    "Each passage starts with a header line naming the company, form, fiscal year, period end, doc id and page "
    "number. A passage is a piece of one page; a table cut across passages repeats its title and column headings.").replace(
    "from the pages", "from the passages").replace("the pages do not contain", "the passages do not contain")


def _call(system: str, user: str, model: str):
    """One question's LLM call(s): reply, number of calls, error, live token counts."""
    from tieout import llm
    reply, calls, error = None, 0, None
    before = (llm.CALLS["tokens_in"], llm.CALLS["tokens_out"], llm.CALLS["live"])
    for _ in range(2):
        calls += 1
        try:
            reply = llm.ask_json(system, user, model=model, tag="baseline")
            if isinstance(reply, dict):
                break
            reply, error = None, "reply is not a JSON object"
        except llm.DailyLimit:
            raise
        except Exception as e:  # noqa: BLE001 - bad JSON or provider error: retry once, then abstain
            error = f"{type(e).__name__}: {e}"[:200]
    live = llm.CALLS["live"] - before[2]
    tokens = {"live_calls": live, "tokens_in": llm.CALLS["tokens_in"] - before[0],
              "tokens_out": llm.CALLS["tokens_out"] - before[1]} if live else {"live_calls": 0}
    return reply or {"answer": "", "value": None, "abstain": True, "facts_used": []}, calls, error, tokens


def build_prompt(question: str, retriever: str = "v2", k: int = K, budget_chars: int = None, mode: str = None) -> dict:
    """Retrieve and assemble the prompt (no LLM call): system, user, retrieved, texts, filing_chars."""
    if retriever == "v1":
        hits = index.retrieve(question, k)
        context = "\n\n".join(f"=== PAGE {i + 1} ===\n" + trim(p["text"], question) for i, p in enumerate(hits))
        return {"system": SYSTEM,
                "user": f"Question: {question}\n\nPages:\n\n{context}\n\nReply with the JSON object only.",
                "retrieved": [{"doc": p["doc"], "page": p["page"], "score": p["score"]} for p in hits],
                "texts": [], "filing_chars": len(context)}
    from baseline import hybrid
    hits = hybrid.retrieve(question, budget_chars or hybrid.BUDGET_CHARS, mode=mode)
    # reading order inside each searched company: filing, page, part
    shown = sorted(hits, key=lambda h: (h["scope"], h["doc"], h["page"], h["part"]))
    context = "\n\n".join(f"=== PASSAGE {i + 1} ===\n" + h["text"] for i, h in enumerate(shown))
    return {"system": PASSAGE_SYSTEM,
            "user": f"Question: {question}\n\nPassages:\n\n{context}\n\nReply with the JSON object only.",
            "retrieved": [{"doc": h["doc"], "page": h["page"], "part": h["part"], "scope": h["scope"],
                           "rank": h["rank"]} for h in shown],
            "texts": [h["text"] for h in shown], "filing_chars": sum(len(h["text"]) for h in shown)}


def answer(question: str, k: int = K, model: str = None, retriever: str = "v2",
           budget_chars: int = None, mode: str = None) -> dict:
    """v2: passages within budget_chars (default hybrid.BUDGET_CHARS); v1: the top k pages."""
    t0 = time.time()
    b = build_prompt(question, retriever, k, budget_chars, mode)
    system, user, retrieved = b["system"], b["user"], b["retrieved"]
    t_retrieve = time.time() - t0
    reply, calls, error, tokens = _call(system, user, model)
    pred = to_pred(reply)
    allowed = {(p["doc"], p["page"]) for p in retrieved}
    return {"answer": str(reply.get("answer", "")), "pred": pred, "unit": reply.get("unit"),
            "retrieved_pages": retrieved, "retrieved_text": b["texts"],
            "facts_used": _facts(reply, allowed),
            "seconds": round(time.time() - t0, 2), "retrieval_seconds": round(t_retrieve, 2),
            "llm_calls": calls, "error": error, "prompt_chars": len(system) + len(user),
            "filing_chars": b["filing_chars"], "est_tokens_in": (len(system) + len(user)) // 4, **tokens}
