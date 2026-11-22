# Conventional RAG baseline (v2 default, v1 kept)

The standard pipeline the graph system is compared against, on the same benchmark, through the same
`tieout.llm.ask_json` (same providers, disk cache and token accounting; cache tag `baseline`) and
scored by the same `tieout.eval.bench` (`is_correct`, `fact_match`, `score`).
One LLM call per question: retrieve filing text, the LLM reads it and does the arithmetic itself.

## What v2 does (default)

| Part | File | Choice |
|---|---|---|
| Passages | `passages.py` | 28,519 passages of 1,000 to 1,500 characters (median 1,357; 78% inside that range, 6% under 500 are page ends) cut from the 9,287 readable pages on line boundaries, blocks split at vertical gaps wider than 1.5x the page median. A table's header (title, "in millions" caption, column headings) is kept with its first rows and copied onto every later piece of a cut table. Every passage starts with `Company (TICKER) \| form \| fiscal year \| period ended \| doc id \| page`. |
| Dense | `dense.py` | BAAI/bge-small-en-v1.5 (MIT, 33M parameters, 384 dimensions), the Hugging Face repo's own ONNX export (fp32, 133 MB) pinned to revision 5c38ec7 and checked by SHA-256, run with onnxruntime (6 threads) and `tokenizers`; query prefix "Represent this sentence for searching relevant passages: " as the model card says; passages have none; [CLS] vector, L2-normalised; inputs cut at 512 tokens. Downloaded once into `baseline/cache/` (gitignored). |
| Lexical | `index.py` | rank_bm25 Okapi, defaults k1 1.5 b 0.75, the v1 tokenizer, over the same text (header line + body). |
| Fusion | `hybrid.py` | reciprocal rank fusion, 1/(60+rank), each ranking cut at 200 (Cormack et al. 2009). |
| Company scope | `scope.py`, `hybrid.py` | Companies named in the question are found from `Graph.load().companies` (full name, name without suffix, run-together form, unambiguous first word, a short list of nicknames such as Google, Coke, Amex, BofA, a capitalised ticker). Each named company is searched on its own filings; the budget is shared by round robin (best passage of each company in order of mention, then the second best of each, ...). Nobody named: search everything. Detection on the 87 development questions: 81 exact, 5 name nobody (Starbucks, "owns the Gatorade brand", two "all 39 companies" questions), 1 misses one of two (a GM abbreviation was in the question; the nickname list was extended). |
| Budget | `hybrid.py` | at most 16,000 characters of filing text per prompt (passages whole, header lines included); 12 passages on average. |
| Answer | `answer.py` | system prompt as v1 (JSON: answer, value in full units or fraction, unit, abstain, facts_used with doc and page), reworded for passages. Bad JSON: one retry, then abstain. A `DailyLimit` error stops the run instead of recording abstentions. |

Per question `scripts/run_baseline.py` records: retrieved passages (doc, page, part, scope, rank), prompt
characters, filing characters, estimated tokens in (chars / 4), the provider's tokens in and out
on live calls (none for a cached reply), seconds for retrieval and in total.

`--retriever v1` keeps the earlier baseline: one chunk per page (9,287), BM25 + TF-IDF fused by RRF,
top 8 pages trimmed to 2,400 characters each (about 20,000 characters).

## Index build

First use builds everything: 21 s passages, 2 s BM25, **2,954 s dense encoding (49 min, CPU, 6 threads, other
processes were also running)**, 118 MB pickle + 28 MB of cached vectors in `baseline/cache/v2/`.
Peak working set during the build 2.1 GB. Embeddings are cached in 2,000-passage shards, so an interrupted
build resumes. Later runs load in about 10 s. Query cost: 0.1 to 0.6 s (BM25 over 28k passages dominates; dense is 1 ms per scope).

## Retrieval evaluation (no LLM)

`python -m baseline.retrieval_eval` (writes `baseline/retrieval_eval.json`). 81 answerable development
questions (round2/dev, questions, dev; unanswerable left out). A gold supporting fact is found when its
(doc, page) is among the retrieved items. Mean over questions of gold facts found / gold facts;
share of questions with every gold fact found in brackets.

Deployed prompts (v1 = top 8 pages, about 20,000 characters; v2 = what fits 16,000 characters, 11.6 passages on average):

| | all (81) | single_fact (32) | multi_hop (27) | cross_filing (22) |
|---|---|---|---|---|
| v1 pages | 0.264 (0.20) | 0.375 (0.34) | 0.259 (0.15) | 0.110 (0.05) |
| v2 hybrid, scoped | 0.418 (0.35) | 0.641 (0.62) | 0.370 (0.22) | 0.153 (0.09) |

At fixed k items (v1 pages, v2 passages) the full table is in `retrieval_eval.json`; v2 hybrid scoped at k = 8 / 12 / 16 / 24 reads 0.341 / 0.415 / 0.465 / 0.567, v1 reads 0.264 / 0.280 / 0.331 / 0.359.
Ablations (k = 12 passages, all): BM25 only 0.278; dense only **0.572**; hybrid 0.415; hybrid without company scope 0.400.

**Finding that a practitioner has to face.** On these development questions dense-only beats the
specified BM25 + dense fusion (0.572 vs 0.415 at k 12; 0.604 vs 0.465 at k 16). BM25 over whole
questions full of generic words and number-heavy tables adds noise that RRF cannot discount. The default
stays `hybrid` as specified; `--mode dense` runs the stronger variant. Choosing it is a choice made on the
aggregate dev numbers (not on any question), which is ordinary practice but is a tuning advantage in
the baseline's favour; the held-out run should state which mode it used. Cross-filing questions stay
low for every variant: they need pages from many filings and no page budget covers them.

## LLM budget plan

Measured offline for all 87 development questions (`python -m baseline.budget`, no LLM call):

| | prompt chars (mean / max) | est. tokens in, chars/4 | est. tokens in, chars/3 (number-heavy) |
|---|---|---|---|
| v2 | 17,260 / 17,934 | 4,315 | 5,753 |
| v1 | 20,357 / 21,725 | 5,089 | 6,786 |

The chars/4 and chars/3 figures are **estimates**, not tokenizer counts (no tokenizer for the Groq or Gemini models is
installed). The plumbing test measured 4,363 tokens in against a chars/4 estimate of 4,370 for gpt-oss-20b, so chars/4 is
the working figure (chars/3 is the safe upper bound); output was 651 tokens (reasoning included). One question is
therefore about **5,000 tokens** (5.0k in + 0.65k out, round up to 5.2k).

Quotas as told: Groq 200,000 tokens a day per model (gpt-oss-120b spent for today; qwen3.8-27b about 80,000 left;
gpt-oss-20b free), 8,000 tokens a minute; Gemini gemini-3.5-flash-lite about 500 calls a day (reset about 13:30 IST),
shared with Session A.

| Run | Questions | Tokens (est.) | Model and day | Pacing |
|---|---|---|---|---|
| Held-out, once, after the freeze | 30 | 30 x 5.2k = about 156k | Preferred: Gemini 3.5 flash-lite, same model as the graph system's planner: 30 calls, no token problem. Fallback: Groq gpt-oss-20b, 156k of its 200k a day, so it needs a day in which Session A leaves it at least that much; retries cost 5k each | Gemini: none needed (retry sleeps are in llm.py). Groq: `--pace 50` after each live call: one question is about 5.0k to 5.8k tokens against 8,000 a minute, so never two in the same minute; 30 questions take about 27 min |
| Development sets | 87 | about 450k | Gemini on the day after the held-out run or a quota-free day: 87 calls. On Groq it is 2.3 model-days (38 questions per 200k) | same |
| Rerun with cached replies | any | 0 | cache/llm | none |

Rule: never mix models inside one reported number. A `DailyLimit` stops the run and writes the rows so far; resume with `--ids` for the rest the next day (completed questions are served from the disk cache and cost nothing).
Both systems must be run with the same model for the comparison; the graph system's planner and linker use its own tokens, the baseline's use whole passages, which is the cost difference the report states.

## Plumbing test (6 development questions, gpt-oss-20b)

`TIEOUT_LLM=groq:openai/gpt-oss-20b`, 6 live calls (of the 10 allowed), the first two single_fact (A01, A02), first two
multi_hop (A19, A20) and first two cross_filing (B05, B07) of `benchmark/round2/dev.jsonl`, one every 65 s; output in `baseline/plumbing_v2.json`.
Rows went through `bench.is_correct`, `bench.fact_match` and `bench.score` exactly as the graph rows do.

| | n | correct | notes |
|---|---|---|---|
| single_fact | 2 | 2 | both gold pages retrieved |
| multi_hop | 2 | 2 | A20: all 5 gold pages retrieved, 1 of 5 figures listed by the model |
| cross_filing | 2 | 0 | both abstained; gold pages found 2/4 and 0/4 |

Measured on the live calls: **4,363 tokens in (mean; the chars/4 estimate said 4,370), 651 tokens out (mean, reasoning included)**,
about 5,000 per question; 1.6 to 2.6 s per call. Six questions prove the plumbing, not accuracy; the interval on 4 of 6 is [0.30, 0.90].

## Design choices a practitioner might call a tuning advantage

- Passage size, header-copy rule, 16,000-character budget, round-robin split across companies: chosen from the pipeline's goals (fit one free-tier request; keep table columns with their numbers), not from questions. The company-name list and nickname list were written from the 40 company names and then extended once after a detection check on the development questions (a GM abbreviation).
- Company-aware retrieval uses the same company list the graph has. A conventional system would need that metadata too, but it is a form of entity linking the graph also gets for free.
- Company scope is applied from a name found in the question text; no filing, fiscal year or form filter is applied, so "latest" and year traps are left to the LLM, as in a standard RAG system.
- `--mode dense` (see the finding above) is selected on development aggregates.
- The prompt asks for JSON with facts_used; this lets the comparison score evidence with `fact_match` the same way as the graph, but it also tells the model to cite documents. Not tuned.
- The 16,000-character budget is larger than v1's page budget per question in useful text per character (more distinct locations) and slightly smaller in total size.

## Run

    set PYTHONPATH=<repo root> & set PYTHONIOENCODING=utf-8
    .venv\Scripts\python.exe -m pip install -r requirements.txt          (baseline block: onnxruntime, tokenizers, psutil, rank-bm25, scikit-learn)
    .venv\Scripts\python.exe -m baseline.hybrid --rebuild                  (49 min once; model download about 134 MB; or skip, the first run builds it)
    .venv\Scripts\python.exe -m baseline.retrieval_eval                    (no LLM)
    .venv\Scripts\python.exe -m baseline.budget                            (no LLM)
    set TIEOUT_LLM=gemini:gemini-3.5-flash-lite
    .venv\Scripts\python.exe scripts\run_baseline.py <questions.jsonl> [--model provider:model] [--mode hybrid|dense] [--budget 16000] [--ids A01,A02] [--limit N] [--pace SECONDS] [--out results\baseline.json]
    .venv\Scripts\python.exe scripts\run_baseline.py <questions.jsonl> --retriever v1 --k 8      (old baseline)

Output: `results/baseline.json` (rows compatible with `bench.score`), fact-level and page-level tables, tokens per question.
Fact-level (a): retrieved = facts_used listed by the LLM, `bench.fact_match`. Page-level (b): retrieved = pages of the passages shown; recalled if (doc, page) is among them.

## Honest limits

- Cross-filing questions that need many filings (counts or rankings over all 39 companies) cannot fit any budget; that is the method's limit, and the graph's reason to exist.
- Dense retrieval at 33M parameters is the small end; a larger embedding model or a cross-encoder reranker would score higher and cost more CPU or tokens. Not tried.
- Gold page recall under-counts: the same figure often sits on another page or filing (the relaxed recall in `run_baseline.py` counts that).
- facts_used is self-reported by the LLM and not verified against the passages.
- Exact-tolerance questions need the LLM to scale "in millions" itself; no post-hoc fixing beyond parsing "12.5%" and "$1.2 billion".
- Company detection misses descriptions that name a company indirectly ("the company that owns Gatorade"); then the search is global.
- Token counts in the budget plan are estimates until the live counts in `results/baseline.json` replace them.
- MCD_FY2025_10K has no readable text layer and is skipped by both retrievers.
