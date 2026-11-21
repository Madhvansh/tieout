# TieOut

A knowledge graph built from SEC filing PDFs in which every number is tied out before it is served.

Auditors "tie out" a figure by tracing it to its source and checking that it agrees with everything it should agree with. That is what this graph does to each number it holds.

## The idea

A 10-K PDF is a spreadsheet with the formulas erased. The numbers still over-determine each other: a subtotal equals a signed sum of the rows above it in every period column at once, total assets equals total liabilities and equity, net income reappears on the cash-flow statement, and last year's column reappears in last year's filing.

TieOut recovers those formulas from the arithmetic alone (no LLM, no XBRL) and keeps them as the graph. The structure then does three jobs:

1. **Join.** "Net sales" at Apple and "Total revenues" at Walmart resolve to one `Revenue` concept node, and fiscal periods are nodes on each company's own calendar. A cross-company question is a traversal through shared nodes.
2. **Verify.** Every recovered formula is a check. A number inside a passing check is verified; one inside a failing check is quarantined and the system abstains.
3. **Derive.** Margins, growth, ratios and a quarter that is printed nowhere (full year minus nine months) are computed by walking formula edges. The answer is a proof tree whose leaves are cells on PDF pages.

On top sits a **living layer**: meaning that arithmetic cannot supply (which printed label is which concept; where a segment's revenue is printed) is learned when a question first needs it. An LLM proposes a row from labels only; a gate made of structure, not another LLM, admits it or holds it for a person.

The language model only turns a question into a typed plan. It never sees a number and never writes one.

## Results

30 held-out questions, written from the PDFs alone and frozen in git before the system was run on them. The system itself was frozen (tag `frozen-before-benchmark`) and run once.

| Question type | Correct | Precision | Recall |
|---|---|---|---|
| Single-fact | 9 / 10 | 1.00 | 0.90 |
| Multi-hop | 10 / 10 | 1.00 | 1.00 |
| Cross-filing | 5 / 10 | 0.90 | 0.54 |
| **Overall** | **24 / 30 (80%)** | **0.97** | **0.81** |

Precision and recall are macro averages over questions; pooled over facts they are 0.98 and 0.85. The 95% interval on 24/30 is 63% to 91%: thirty questions is a small sample.

| Comparison on the same 30 questions | Correct |
|---|---|
| TieOut, frozen first run | 24 |
| Chunk retrieval baseline (BM25 + TF-IDF, 8 pages, same LLM) | 10 |
| TieOut with hand-written plans (written blind, no gold) | 27 |
| Shared concept layer off | 10 |
| Derivations off | 18 |
| Living layer off | 20 |
| Verified-only answering | 24 |
| TieOut v1.1, after fixing four causes the first run exposed (not held-out) | 27 |

TieOut minus baseline: +14 questions, paired bootstrap 95% interval +9 to +19 of 30.

Beyond the benchmark:

- **Graph.** 55 filings (40 latest 10-Ks, 10 prior-year 10-Ks, 5 third-quarter 10-Qs). 12,808 numbers from the primary statements, 93% inside a passing check; 32 figures that a later filing restated; about 44,000 more rows from other tables, parsed and checked the same way and left unbound until a question asks for them.
- **Fault injection** on 161 parsed statements: the checks catch 99% of sign flips, 99.8% of values moved to the wrong period column and 100% of single-cell scale errors, but only 27% of two neighbouring rows trading values and 2.5% of a whole table read at the wrong scale.
- **Independent audit** against the SEC's XBRL data, which the pipeline never reads: 88% of extracted numbers confirmed by value and period. Concept bindings agreed 94.6% on the first audit; it exposed three real errors, and after fixing them 98.6% agree.
- **One filing failed completely.** The McDonald's PDF embeds fonts with no Unicode map, so its text layer is glyph codes. The build detects this and extracts nothing rather than garbage.

The two-page write-up with failures and next steps is in `report/results.pdf`.

## How it works

| Step | What happens | Code |
|---|---|---|
| 1. Words | Each PDF is read once into word boxes (pdfplumber, MIT licence) | `tieout/pdfio.py` |
| 2. Tables | Columns from the right edges of numbers, periods from header dates, scale from the caption | `tieout/tables.py`, `statements.py`, `alltables.py` |
| 3. Formulas | A row is a subtotal only if a signed run of rows above it sums to it in every column | `tieout/calc.py` |
| 4. Graph | Readings, checks (SUM, TIE, RATIO, CARRY), concepts, fiscal periods, facts, text facts | `tieout/graph.py`, `periods.py`, `textfacts.py` |
| 5. Semantics | Seed concepts as data; a label match is a proposal that structure must confirm | `semantics/seed_concepts.yaml`, `tieout/concepts.py` |
| 6. Living layer | Bindings learned from questions through the structural gate; held ones go to a review queue | `tieout/living.py`, `items.py`, `rawlayer.py` |
| 7. Plan | The LLM writes a typed plan from the vocabulary, never from data | `tieout/planner.py`, `plan.py` |
| 8. Execute | Deterministic traversal; answer with proof leaves, or an abstention that names the missing fact | `tieout/execute.py` |

## Run it

Python 3.12. No GPU. The built graph and the LLM cache are in the repo, so the first three commands need no PDFs and no API key.

```
pip install -r requirements.txt
pytest -q                                      # formula recovery and fiscal calendars, no PDFs needed
python -m tieout stats                         # what the shipped graph holds
python scripts/run_benchmark.py --offline      # replays the benchmark from cached plans
```

To ask your own question, set a free key (`GEMINI_API_KEY` in the environment or in `.env`) or use a local model:

```
python -m tieout ask "What was Amazon's gross margin in 2025?"
TIEOUT_LLM=ollama:qwen3.5:4b python -m tieout ask "Who audits the bank with the most deposits?"
```

To rebuild everything from the PDFs (about 260 MB, fetched from each company's own investor-relations site; URLs and sha256 in `data/manifest.csv`):

```
python scripts/fetch_corpus.py
python -m tieout build            # words, tables, formulas, graph: about ten minutes the first time
python scripts/build_raw.py       # every other table, kept unbound
python scripts/teach.py           # teaching run for the living layer (LLM)
```

Other experiments: `scripts/run_baseline.py`, `run_faults.py`, `run_noise.py`, `run_audit.py` (needs `SEC_USER_AGENT`). Ablations are flags of `run_benchmark.py` (`--ablate concepts|derivations|living`, `--strict`, `--oracle benchmark/oracle_plans.json`).

To reproduce the frozen 24/30 exactly, check out the tag `frozen-before-benchmark` and run `python scripts/run_benchmark.py --offline`. The current head is v1.1.

## The benchmark and how it was prepared

- **Written from the PDFs only.** 48 candidate questions (16 per type) were drafted by an AI agent that could read the PDFs and the manifest and nothing else: no code, no graph, no system output. It recorded each gold answer with its document, PDF page, printed value and derivation, and recomputed every derived answer in Python.
- **Frozen before any run.** The candidates were committed before the system had been run on any of them (commit `879976a`).
- **Split.** A seeded random split, stratified by type, gave the 30-question benchmark (`benchmark/questions.jsonl`) and 18 development questions (`benchmark/dev.jsonl`). Only the development questions were used while debugging.
- **Checked by hand.** I checked every gold answer of the 30 against its PDF page. [[TO CONFIRM BEFORE SUBMISSION: result of my check, and any corrections]]
- **Traps on purpose.** Fiscal years that do not match calendar years (12 questions), restated figures (3), scale in thousands versus millions (2), segment versus consolidated (2), and two unanswerable questions where the right answer is to abstain.
- **Scoring rules fixed in advance** (`tieout/eval/bench.py`). Retrieved facts are every fact the executor read, including every company scanned for a ranking. Gold facts are (document, page, printed value) and do not refer to graph ids. Abstaining is correct only on an unanswerable question.

## What failed

Six of 30 in the frozen run. Three were the planner and three were the graph:

- **Planner refused two answerable questions** because they also asked "why" a figure changed. The graph holds facts, not reasons, but the factual half was answerable.
- **Planner chose the wrong step as the answer** on a "how many, and which" question: it returned the six right companies where the gold was the count.
- **A partial ranking was returned as if complete.** Citigroup's deposits line was never bound, so a ranking of four named banks came back with three. This is the most serious failure: a confident wrong answer.
- **A headcount was missing** (General Motors prints hourly and salaried counts, no total), and the plan language could not rank by an attribute.
- **Share classes were ambiguous.** Asked for Meta's Class A shares, the text-fact layer returned the Class B count.

Structural limits that the experiments measured are in the results PDF: the checks' two blind spots, the coverage cost of verified-only answering on a damaged graph, and one page clipped in the source PDF.

## How I used AI

The brief says to use AI tools freely and say how. Here is exactly how.

**What the AI did.** I built this with Claude Code (Anthropic's coding agent) working as my engineering team. It wrote all of the code in this repository. It ran the background research, laid out the design options, red-teamed the chosen design before any code was written, ran the experiments, and drafted this README and the results PDF. Its sub-agents drafted the benchmark candidates (restricted to the PDFs) and the blind hand-written plans.

**What I did.** I made the decisions and I own the result.

1. **The design.** I was offered three: the recovered-formula graph, a lighter version without the check layer, and a graph that learns from questions behind a gate. I asked for the first and the third fused into one system and told it not to drop anything for being hard.
2. **The corpus.** Forty latest 10-Ks across five sectors with banks included, prior-year 10-Ks for ten companies so that restatements show up, and a few 10-Qs so that an unprinted quarter can be derived.
3. **The fine-tuned planner.** Both the literature survey and the red team advised against fine-tuning a small local planner. I overruled them. [[TO CONFIRM: the measured result, reported as it came out]]
4. **The headline number.** [[TO CONFIRM: my ruling]] The frozen first-run 24/30 is the headline; the 27/30 after fixes is shown beside it and labelled as not held-out.
5. **The gold answers.** [[TO CONFIRM]] I checked each of the 30 against its PDF page.
6. **The learned bindings.** [[TO CONFIRM]] I reviewed the 81 bindings the gate admitted and the 5 it held, and my rejections are in `semantics/review_decisions.yaml`.
7. **The name.**

**LLMs inside the system.** Only free options, as the brief asks: Gemini 3.5 Flash-Lite on the free tier writes plans and proposes bindings; a local Ollama model is the no-key path. Every call is cached in `cache/llm`. No number in the graph or in an answer ever passes through a language model.

**What I would say if asked "did you write this code?"** No. I directed it, I chose between the alternatives, I set what counted as evidence, and I can walk through every module and every number in this README.

## Repository map

```
tieout/        the package (see "How it works")
semantics/     seed concepts and metrics as data; learned bindings; review queue and sheet
benchmark/     candidates, the 30 questions, development questions, blind oracle plans, check sheets
baseline/      the chunk-retrieval baseline
finetune/      training-set generator and Kaggle scripts for the local planner
scripts/       build, teach, evaluate, audit
results/       every result file the numbers above come from
graph/         the built graph and the raw layer
data/          manifest (URLs, sha256), extracted text facts
tests/         unit tests that need no PDFs
report/        the two-page results PDF
```
