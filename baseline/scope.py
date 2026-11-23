"""Which companies does a question name? (v2 retriever: search inside each named company's filings.)

Names come from Graph.load().companies: the full name as written ("Apple Inc."), the name without
its corporate suffix ("Apple"), a run-together form ("cocacola", "mcdonalds"), and the first word
of a multi-word name when it identifies the company by itself ("Costco", "Cisco", "Exxon", "Goldman").
ALIASES adds the nicknames any reader of financial news knows (Google, Facebook, Coke, Amex, BofA).
It was written from the 40 company names, not from benchmark questions.

Matching is on a lower-cased, punctuation-free copy of the question, on word boundaries
("Apple's" matches "apple"). A ticker of two or more letters written in capitals (NVDA) also
counts. Names that are ordinary English words (Target, Visa, Meta, Ford, Chase, Oracle) only match
when capitalised in the question. Companies are returned in the order the question first names them.
"""
from __future__ import annotations

import re

SUFFIXES = {"inc", "corp", "corporation", "co", "company", "group", "holdings", "ltd", "plc", "the"}
GENERIC_FIRST = {"general", "american", "united", "advanced", "bank", "home", "johnson", "procter", "morgan",
                 "wells", "meta"}
CAPITALISED_ONLY = {"target", "visa", "meta", "ford", "chase", "oracle"}
ALIASES = {
    "GOOGL": ["google", "alphabet"], "META": ["facebook", "meta"],
    "JPM": ["jpmorgan", "jp morgan", "jpmorgan chase", "chase"], "BAC": ["bofa", "bank of america"],
    "C": ["citi", "citigroup"], "WFC": ["wells fargo"], "GS": ["goldman sachs", "goldman"],
    "KO": ["coke", "coca cola", "cocacola"], "PG": ["p and g", "procter and gamble"],
    "JNJ": ["j and j", "johnson and johnson"], "XOM": ["exxon", "exxonmobil", "exxon mobil"],
    "MA": ["master card"], "AXP": ["amex", "american express"], "HD": ["home depot"],
    "UNH": ["unitedhealth", "united health"], "WMT": ["walmart", "wal mart"], "MCD": ["mcdonalds", "mcdonald"],
    "MRK": ["merck"], "PEP": ["pepsi", "pepsico"], "GM": ["general motors"], "F": ["ford", "ford motor"],
    "AMD": ["advanced micro devices"], "CRM": ["salesforce"], "NKE": ["nike"], "AMZN": ["amazon"],
}

_PUNCT = re.compile(r"[^a-z0-9]+")
_CACHE: dict = {}


def norm(text: str) -> str:
    """Lower case, '&' as 'and', anything else that is not a letter or digit as one space."""
    return " " + _PUNCT.sub(" ", text.lower().replace("&", " and ")).strip() + " "


def _forms(ticker: str, name: str, all_names: dict) -> set:
    tokens = norm(name).split()
    while tokens and (tokens[-1] in SUFFIXES or tokens[-1] == "and"):
        tokens.pop()
    while tokens and tokens[0] in SUFFIXES:
        tokens.pop(0)
    forms = {" ".join(tokens), "".join(tokens)}
    if "com" in tokens:                                  # "Amazon.com" -> "amazon"
        forms.add(" ".join(t for t in tokens if t != "com"))
    if len(tokens) > 1 and tokens[0] not in GENERIC_FIRST and len(tokens[0]) >= 5:
        firsts = [norm(n).split()[0] for t, n in all_names.items() if t != ticker]
        if tokens[0] not in firsts:
            forms.add(tokens[0])
    forms |= {norm(a).strip() for a in ALIASES.get(ticker, [])}
    out = set()
    for f in forms:
        out.add(f)
        if f.endswith(" s"):
            out.add(f[:-2])
    return {f for f in out if f}


def _patterns(companies: dict) -> list:
    key = id(companies)
    if key in _CACHE:
        return _CACHE[key]
    names = {t: c["name"] for t, c in companies.items()}
    pats = []
    for t, name in names.items():
        forms = _forms(t, name, names)
        plain = sorted((f for f in forms if f not in CAPITALISED_ONLY), key=len, reverse=True)
        caps = sorted(f for f in forms if f in CAPITALISED_ONLY)
        rx = re.compile("|".join(r"(?<![a-z0-9])" + re.escape(f).replace(r"\ ", " ") + r"(?![a-z0-9])" for f in plain)) \
            if plain else None
        cap_rx = re.compile(r"\b(?:" + "|".join(f.capitalize() + "|" + f.upper() for f in caps) + r")\b") if caps else None
        tick_rx = re.compile(r"(?<![A-Za-z0-9])" + re.escape(t) + r"(?![A-Za-z0-9])") if len(t) >= 2 else None
        pats.append((t, rx, cap_rx, tick_rx))
    _CACHE[key] = pats
    return pats


def detect(question: str, companies: dict) -> list:
    """Tickers of the companies the question names, in order of first mention."""
    spaced = norm(question)
    hits = []
    for t, rx, cap_rx, tick_rx in _patterns(companies):
        pos = []
        if rx and (m := rx.search(spaced)):
            pos.append(m.start() / max(1, len(spaced)))
        if cap_rx and (m := cap_rx.search(question)):
            pos.append(m.start() / max(1, len(question)))
        if tick_rx and (m := tick_rx.search(question)):
            pos.append(m.start() / max(1, len(question)))
        if pos:
            hits.append((min(pos), t))
    return [t for _, t in sorted(hits)]
