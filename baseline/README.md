# Conventional RAG baseline

Page-chunk retrieval plus one LLM call that reads the pages and does the arithmetic itself.
It is the standard pipeline the graph system is compared against, on the same benchmark,
through the same `tieout.llm.ask_json` (same provider, model and disk cache; tag `baseline`).

**Index** (`index.py`). One chunk per PDF page of every readable filing (MCD_FY2025_10K skipped),
9,287 pages. Text = words grouped into lines (`tieout.tables.group_lines`), gaps > 12 pt written
as ` | `, with a header line: company, ticker, form, fiscal year, period end, doc id, page (1-based).
Retrieval = BM25 (rank_bm25 Okapi, k1 1.5, b 0.75) and TF-IDF cosine (sklearn, unigrams+bigrams,
sublinear tf, min_df 2, English stop words), fused by reciprocal rank (k 60, depth 200 each).
Tokenizer: lowercase `[a-z0-9]+`. Nothing was tuned on the benchmark. Build about 25 s, 121 MB
pickle in `baseline/cache/` (rebuilt automatically when a word cache changes).

**Answer** (`answer.py`). Top k = 8 pages; each page body trimmed to 2,400 characters (one
contiguous window of whole lines with the most question-word hits), so about 19,000 characters
of filing text per prompt: fits a Groq free-tier request and Ollama's 8,192-token context.
The LLM returns JSON (answer, value in full units or fraction, unit, abstain, facts_used) and is
told to abstain when the pages lack the answer. A bad JSON reply is retried once, then abstains.
String values are parsed ("12.5%" to 0.125, "$1.2 billion" to 1.2e9); nothing else is fixed up.

**Run**
    set PYTHONPATH=<repo root> & set PYTHONIOENCODING=utf-8 & set TIEOUT_LLM=<provider:model>
    .venv\Scripts\python.exe -m baseline.index --rebuild        # optional; first use builds it
    .venv\Scripts\python.exe -m baseline.sanity                  # retrieval ranks of 6 known pages
    .venv\Scripts\python.exe scripts\run_baseline.py [benchmark\questions.jsonl] [--k 8] [--limit N]

Writes `results/baseline.json` (rows compatible with `bench.score`) and prints two tables:
(a) fact-level, retrieved = the facts_used the LLM lists, scored by `bench.fact_match`;
(b) page-level, retrieved = the k pages; gold fact recalled if its (doc, page) is among them,
precision = share of the k pages holding a gold fact; unanswerable questions left out.
A supplementary relaxed recall counts the gold figure printed on any retrieved page of that company.
**Honest limits**
- k = 8 pages cannot cover questions needing many filings (C42 needs 39); that is the method.
- Keyword retrieval favours MD&A pages that repeat a figure in prose over the statement page
  itself; the right figure is usually in the top 3, the statement page often is not.
- facts_used is self-reported by the LLM and not checked against the page; (a) therefore counts
  only what it says it used, not all it read, which is generous on precision; (b) is the
  everything-it-read view.
- Exact-tolerance questions (tolerance 0) require the LLM to scale "in millions" itself.
