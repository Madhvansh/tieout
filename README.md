# TieOut

Questions about SEC filings, answered from a knowledge graph of the filings' own arithmetic. Every answer is read from graph nodes and comes with its proof (the filing, the page and the printed figure behind each number) and a tier that says how far to trust it. 55 filings of 40 companies. The reported run uses one free Gemini key and a laptop CPU.

**Results PDF (2 pages): [`report/results.pdf`](report/results.pdf).** Optional research note: [`report/note.pdf`](report/note.pdf). Where the code is: [repository map](#repository-map).

## Results: 30 held-out questions, run once

Written by blind authors and committed before any run (`benchmark/round2/questions.jsonl`; how: [below](#how-the-benchmark-was-made)), run once on the frozen code (tag `frozen-v2`), reported as it came. **Precision**: of the facts an answer rests on (its proof leaves: the printed figures it was computed from), the share that are gold supporting facts. **Recall**: of the gold supporting facts, the share the answer retrieved. Both are means over questions, computed by `tieout/eval/bench.py`. No model judges an answer.

<!-- gen:heldout (written by scripts/readme_numbers.py from results/; never edit by hand) -->
| Question type | Questions | Accuracy | Precision | Recall |
|---|---|---|---|---|
| Single fact | 10 | 8/10 (80%) | 80% | 67% |
| Multi-hop | 10 | 9/10 (90%) | 91% | 98% |
| Cross-filing comparison | 10 | 8/10 (80%) | 75% | 48% |
| **Overall** | 30 | 25/30 (83%) | 82% | 72% |

- On the same 30 questions: the retrieval baseline (passages, the same free model reads) 15/30; TieOut's plans alone, without the text reader, 19/30 (`results/heldout_v2.json`, `results/heldout_baseline_hybrid.json`). The baseline reads 16,000 characters of passages; with the 48,000 the text reader gets inside TieOut it scores 22/30 (`results/heldout_baseline_hybrid_b48.json`).
- One free key was the limit of this run: the stronger arbiter models answered on 2 of 30 questions before their free daily quota was spent, and 5 answers were then served on one planner's reading, FLAGGED. The same frozen system on a key without that daily limit, a side number and not the headline: 27/30 (`results/heldout_abl_none.json`).
- Right by tier: VERIFIED 4/4, WITNESSED 3/4, CORROBORATED 9/9, FLAGGED 8/11, ABSTAIN 1/2. Wrong while VERIFIED: 0.
- Under the strict scorer (the rules of 21 November, before six were relaxed after misses were read): 24/30. 95% interval of the accuracy: 66% to 93%.
- `python scripts/check_brief.py results/heldout_v2.json`: 28 answers served, 80 proof leaves, 80 of them graph nodes.
- Paid side experiment, is the model the bottleneck? (Claude Sonnet 5.5 plans and picks, gemini-3.1-pro arbitrates, labelled and never mixed into the headline): 26/30 (`results/heldout_v2_scaled.json`).
<!-- /gen:heldout -->

## Run it

Python 3.12, no GPU. Replaying the reported runs needs no key: the built graph and every model call they made are in the repository. `--offline` replays only the model calls; linking, the executor and the scoring run again. The `ask` line prints the answer shown in the next section. The last command replays the held-out run as it happened and prints 25/30: the arbiter calls the free key refused are not in the cache. The side runs (the reference without that limit, the ablations, the paid run) are kept as result files; their extra model calls are not shipped.

```
pip install -r requirements.txt
python scripts/check_brief.py results/heldout_v2.json    # every proof leaf of every served answer is a graph node
python scripts/report_numbers.py                          # the numbers of this README and the PDF, from results/
python -m tieout ask --offline "What were Apple's net sales in the fourth quarter of fiscal 2026?"
TIEOUT_LINK_LLM=gemini:gemini-3.5-flash-lite python scripts/run_benchmark.py --bench benchmark/round2/questions.jsonl \
  --model gemini:gemini-3.5-flash-lite --checker ollama:tieout-planner-4b-v3 --pick-checker gemini:gemini-3.1-flash-lite \
  --arbiter gemini:gemini-3.6-flash,gemini:gemini-3.5-flash,gemini:gemini-3-flash-preview \
  --passage gemini:gemini-3.5-flash-lite --offline --name replay     # the held-out run, replayed from the cache
```

Without `--offline` the last command answers new questions: it needs a free `GEMINI_API_KEY` in `.env` and the fine-tuned planner in Ollama (`finetune/README.md`). Rebuild the graph from the PDFs with `python scripts/fetch_corpus.py`, `python -m tieout build`, `python scripts/build_raw.py`, `python scripts/add_headings.py`, `python scripts/ledger.py`.

## Repository map

| Path | What is in it |
|---|---|
| `tieout/` | The system. Building the graph: `pdfio.py`, `tables.py`, `statements.py`, `rawlayer.py`, `textfacts.py`, `graph.py`. Answering: `planner.py` and `plan.py` (a model writes a typed plan), `link.py` (plan lines to graph nodes), `execute.py` (runs the plan on the graph, returns the answer and its proof), `readers.py` and `passage.py` (the second planner, the text reader as witness and pointer, the judge), `witness.py` (tiers). |
| `tieout/eval/` | Scoring: `bench.py` (accuracy, precision, recall), `brief.py` (is every proof leaf a graph node), fault and noise tests. |
| `scripts/` | Entry points: `run_benchmark.py`, `run_baseline.py`, `check_brief.py`, `report_numbers.py`, `clean_clone_check.py`, `check_leakage.py`, and the graph build steps. |
| `benchmark/` | The questions with gold answers and supporting facts. `round2/questions.jsonl` is the held-out set; `round3/` to `round11/` are the development rounds. |
| `baseline/` | The retrieval baseline (passages read by the same free model). |
| `graph/`, `data/` | The built graph and the filings' extracted pages. |
| `results/` | One JSON file per reported run; `first_runs/` holds each development round's first run. |
| `cache/` | Every model call of the headline run, so it replays with no key. |
| `finetune/` | Training data, notebooks and model card of the fine-tuned planner. |
| `report/` | The results PDF, the research note and their LaTeX sources. |
| `tests/` | 280 unit tests (`python -m pytest -q`). |
| `DECISIONS.md` | The design decisions D1 to D15 and why each was made. |

## One answer, end to end

"What were Apple's net sales in the fourth quarter of fiscal 2026?" No filing prints that number. A language model writes the plan and never sees a figure; the executor runs it on the graph:

```
PLAN:   {"steps": [{"id": "s1", "op": "get", "company": "AAPL", "concept": "Revenue", "period": {"fy": 2026, "q": 4}}], "answer": "s1"}
ANSWER: $114.477 billion
TRUST:  VERIFIED
  source: AAPL_FY2026_10K page 35 | Total net sales | printed 478,834 | verified
  source: AAPL_FY2026_Q3_10Q page 4 | Total net sales | printed 364,357 | verified
  note: AAPL Revenue for the fourth quarter is not printed; derived as the full year minus the nine months
```

The graph holds the equation "year = nine months + fourth quarter" across the 10-K and the third-quarter 10-Q. The fourth quarter is its only unknown, so the executor solves for it. Both printed figures sit inside checks that hold on their own pages, so the answer is VERIFIED. The two sources are its proof leaves: the facts scored for precision and recall.

## The idea: a filing is a spreadsheet whose formulas were erased

The printed numbers of a 10-K over-determine each other. TieOut recovers the formulas from the numbers alone, with no model and no XBRL, and keeps them as the graph. The structure decides what a model would otherwise guess:

- which row is a subtotal: a signed run of the rows above it sums to it in every period column;
- which column is the total: it sums the segment columns beside it;
- which two figures are one fact: the same value printed in two tables ties them;
- the scale: net income over shares must give the printed earnings per share, which pins thousands against millions;
- the period: last year's column must equal last year's filing.

The 55 filings hold 2,800 such checks (`results/graph_stats.json`). They verify a figure, solve for one printed nowhere (above), name a row by its place in the arithmetic, and explain a figure that changed between two filings (a restatement).

**What is in the graph** (`tieout/graph.py`). Nodes: filings and their tables; readings (one printed number in one cell, with its page); facts (company, concept, period) that readings report; bindings ("this printed row means this concept"); checks that tie readings together; and, outside the three statements, the rows and cells of every other table and the figures of numeric sentences (52,761 rows, 192,730 cells, 15,124 sentence records). Edges: a reading sits in a table of a filing, reports a fact, and takes part in checks. Concepts and periods are shared across companies: they are the join points of a cross-filing question.

![Merck's balance sheet, 10-K page 80, drawn by python -m tieout show: green brackets are recovered formulas, the answer cell is outlined in blue](report/fig_page.png)

## How a question is answered

1. **Plan.** A language model compiles the question into a typed plan (`benchmark/plan_language.txt`: get, calc, compare, rank, filter, series, members).
2. **Link.** A line outside the statements (a segment, a count, a sentence figure) is chosen among candidate rows shown with their labels, headings and columns, and no figures.
3. **Execute.** A deterministic executor runs the plan on the graph and returns the answer with its proof leaves, or abstains and names the missing fact.
4. **Four eyes.** No single model's choice is served as fact. Two planners read each question: a free hosted model (`gemini-3.5-flash-lite`) and a Qwen3-4B we fine-tuned on Kaggle, on the laptop CPU. Their plans are compared on what they compute; a stronger free model arbitrates only when they differ. Two pickers choose each line and are compared on the figure each pick yields. They are not fully independent: both pickers are Gemini models, and the 4B learned from paraphrases the hosted model wrote.
5. **A text reader: a witness and a pointer, never a source.** The baseline of this repository (passages of the filings, BM25 and `bge-small` vectors, the same free model reads them) is used inside TieOut in two ways only. As a witness it corroborates or disputes: the same figure makes a FLAGGED answer CORROBORATED; a different figure against a VERIFIED or WITNESSED answer sends the question to the arbiter. As a pointer it names a place (filing, page, figure); the node at the place it names is linked to the plan like any picker's choice, and the executor answers from the graph. When it disputes a FLAGGED answer, its own answer is first recomputed from graph nodes (the node it names, a planner's plan on the rows it cited, or its own arithmetic written as one expression and evaluated by code on those nodes); then the arbiter judges which of the two derivations answers the question as asked. The judge picks a derivation and never writes a figure. Nothing the reader writes is served, and `scripts/check_brief.py` fails a run if any proof leaf of any served answer is not a node of the graph.

<!-- gen:devline -->
On the four latest development rounds (142 questions, `results/ff_r7.json` to `ff_r10.json`), the plans alone were right on 114 and the whole system on 128. `check_brief` passes on 4 of the 4 files (round 10: 31 answers served, 77 proof leaves, 77 of them graph nodes).
<!-- /gen:devline -->

| Tier | What the code promises (`tieout/execute.py` `trust`, `tieout/readers.py`) |
|---|---|
| VERIFIED | Every number is a statement line bound by structure (no model chose the row), inside a formula that holds or solved from such numbers; both planners agree. |
| WITNESSED | The same, but some lines were chosen by models, each choice by two models that agreed. |
| CORROBORATED | An answer with a named weakness that the text reader repeats, figure for figure. |
| FLAGGED | Answered, each weakness named. Every answer on a line the reader pointed at, or that the judge chose, is here. |
| ABSTAIN | Not answered. Says what is missing. |

## How the benchmark was made

- **Corpus.** 55 filings of 40 companies across sectors, banks included (`data/manifest.csv`): the latest 10-K of each, the prior-year 10-K of ten (so restatements show) and five 10-Qs (so a quarter printed nowhere can be derived), fetched from SEC EDGAR by `scripts/fetch_corpus.py`.
- **Questions.** Three blind authors, language-model agents that could read only the text of the filings and never saw the system, wrote 69 candidates, each with a golden answer, its document and page, and the supporting facts as (document, page, printed value). Author A's questions were read by the developer while the system was still changing, so all of A is development. The 30 held-out questions come only from authors B and C, drawn by a seeded script (`scripts/split_round2.py`): 10 single-fact, 10 multi-hop, 10 cross-filing. The split was committed (`1d7346f`) before any run; the other 39 are development.
- **Gold, verified blind.** Another agent re-derived all 30 answers from the pages without seeing the authors' gold. A script compared the two: 28 of 30 agree. A third agent judged the other 2 to be differences of form, and the author's gold stands (`benchmark/round2/adjudication.json`). No one checked the gold by hand: every check is an automated pass (ruling D8).
- **Sealed until the freeze.** Nobody who changed the system read the held-out questions. Every later development round passed a leak gate (`scripts/check_leakage.py`) that drops near copies of held-out questions without anyone reading them.
- **LLM use.** Authors, checker and adjudicator are language models. Answers are scored by a script against keys the system did not write.

## Development: we measured our own overfitting

The first development set flattered the system, so the work ran in rounds. Blind authors (a new persona each round: analyst, auditor, credit analyst, journalist, test engineer) write a fresh round; the leak gate runs; the system answers it once, and that first run is recorded before anyone reads an answer. Only then are the misses fixed, by class, each fix with a test, and the round becomes development material. First runs stay well below the same rounds on the current code. Part of that is adaptation; part may be that fresh questions are harder (Recht et al. 2019 saw a similar drop on fresh ImageNet test sets with no reuse at all). Six scoring rules were relaxed on 22 November after misses were read, then tightened against a red team's counter-examples, so numbers are also given under the strict scorer of 21 November.

<!-- gen:rounds -->
| Fresh round | First run, final scorer | First run, strict scorer | Current code |
|---|---|---|---|
| 3 | 25/36 | 25/36 | 36/36 |
| 4 | 23/34 | 22/34 | 35/35 |
| 5 | 20/36 | 20/36 | 35/36 |
| 6 | 17/27 | 17/27 | 30/35 |
| 7 | 23/36 | 20/36 | 35/36 |
| 8 | 29/35 | 27/35 | 34/35 |
| 9 | 26/36 | 21/36 | 31/36 |
| 10 | 26/35 | 21/35 | 28/35 |
| 11 | 29/36 | 26/36 | 31/36 |
| **All** | 218/311 (70%) | 199/311 (64%) | 295/320 (92%) |

Files: `results/pooled_first_runs.json` (first runs rescored under both scorers), `results/first_runs/`, `results/ff_r<N>.json` (current code). Round 10's row is the first run of the corrected design, made before anyone read the round's answers; a version withdrawn the same night (it served answers the text reader wrote) had run it once before. Round 11's row is its first run, on the code before the judge between two derivations was merged; run once more with the judge, still before anyone had read an answer, it scored 30/36 (`results/first_runs/round11_judge.json`).
<!-- /gen:rounds -->

## Where it fails, by class

The five held-out misses (`results/heldout_v2.json`):

- **The free key ran out of arbiter calls** (2). The stronger free models answered three calls and then refused for the day. From the third question on, a disagreement between the two planners was served as one planner's reading, FLAGGED. American Express's Delta share of Card Member loans and the restated Sam's Club grocery sales were lost this way: with the arbiter available both are right (`results/heldout_abl_none.json`, a side run on a key without that limit).
- **A printed figure no node holds** (1). GM's share of U.S. vehicle sales, 17.2 percent on 10-K page 6: the planners said no filing gives a market share, the text reader named the page, but the table on that page was never read into the graph. There was no node to answer from, so the system abstained.
- **An unanswerable question answered** (1). Microsoft prints no dollar figure for Azure. One planner linked "Azure" to another, larger line and its comparison was served, FLAGGED, although the text reader said the figure is not disclosed.
- **A reading both planners shared** (1). "Which of NVIDIA's five end markets grew fastest, and how much revenue did it bring in": both planners returned another market's revenue. It was served WITNESSED, the one wrong answer above FLAGGED in this run.

The fresh rounds add two more classes (round 10: `results/first_runs/round10_pointer.json`). **A scale read wrong outside the money tables**: Exxon's refining capacity carries its caption's thousands and Chevron's does not, so their ratio comes out 1,895 instead of 1.79. **Look-alike lines**: Walmart's square feet per unit, a sentence on page 40 against the Item 2 table on page 29.

## The role's four research directions, measured on this graph

The job description names four directions for the internship. Each has a counterpart that can be measured here. The numbers, their files and their limits are in a two-page research note, [`report/note.pdf`](report/note.pdf).

- **Graph quality against correctness.** With 5 percent of the statement cells damaged, 342 of 4,158 facts are silently wrong when the checks are off and 64 when they are on; the checks withhold 1,030 (`results/damage_curve.json`, five seeds, no model). On rounds 7 to 10 no wrong answer was VERIFIED at 0, 10, 25 or 50 percent damage (`results/jd_dose.json`). Of 579 figures the text reader cited, 546 are nodes of the graph (`results/jd_resolve_rate.json`).
- **Pruning.** 1,649 of 13,384 statement cells can be recomputed by a recovered formula. Dropping the 213 facts the executor can solve for changes no verdict on 142 questions (`results/jd_prune.json`).
- **Access control with provenance.** An answer is served only if the asker may read every proof leaf. Withholding the most cited company leaves 123 of 131 served answers, and none of them rests on a withheld leaf (`results/jd_access.json`).
- **Shipping.** Every model call of a reported run is cached, so it replays with no key; `scripts/check_brief.py` and `scripts/clean_clone_check.py` guard each build.

## What did not work

- **Serving an answer the text reader wrote.** An earlier version filled abstentions with the reader's own answer: 27 of 35 on round 10's first run (`results/first_runs/round10.json`). It broke the brief's rule that answers come from the graph, so it was rebuilt to point only: 26 on the same round (`results/first_runs/round10_pointer.json`).
- **A stronger model alone.** On round 7, `gemini-3.1-pro` planning alone on the graph scored 22 of 36 and flash-lite alone 19 (`results/v_r7_pro.json`, `results/v_r7_fl.json`, code of 22 November).
- **The link store on new questions.** Round 8 scored 30 of 35 with the links learned on rounds 1 to 7 and 30 of 35 without them (`results/store_round8.json`, `results/nolinks_round8.json`): stored links help a question asked again, not a new one.
- **A model judge to admit a binding** (97.0 percent precise, but it threw away 34 of 66 right proposals: `results/judge_vs_gate.json`), and **relaxing a scoring rule after reading a miss** without a counter-example test (hence the strict scorer).
- **Measured on the last night and not adopted.** The text reader with twice the context: right alone on 91 of 142 against 89, with 13 answers better and 11 worse (`results/p_r7_b96.json` to `p_r10_b96.json`). The arbiter planning a FLAGGED answer that two planners share: no answer changed on 284 questions (`results/d1_r3.json` to `d1_r10.json`). The 8B planner as second reader: 91 of 107 against 95 for the 4B (`results/a8b_r7.json` to `a8b_r9.json`).

**The fine-tuned planners.** Qwen3 models fine-tuned on Kaggle GPUs (LoRA; plans in the plan language for templated questions, paraphrased by the hosted model, leak-checked against the held-out questions), published with cards: [4B v3](https://huggingface.co/Madhvansh/tieout-planner-qwen3-4b-v3-gguf) (the second planner), [4B v2](https://huggingface.co/Madhvansh/tieout-planner-qwen3-4b-v2-gguf), [1.7B v3.1](https://huggingface.co/Madhvansh/tieout-planner-qwen3-1.7b-v3.1-gguf), [1.7B v2](https://huggingface.co/Madhvansh/tieout-planner-qwen3-1.7b-v2-gguf), [0.6B v2](https://huggingface.co/Madhvansh/tieout-planner-qwen3-0.6b-v2-gguf). Planning alone on rounds 1 to 5 and 6 to 8 (`results/v32_table_all.json`): hosted flash-lite 175 of 194 and 75 of 106, 4B v3.2 153 and 52, 8B v3.2 160 and 57.

## What I would do next

- **Units and scales of tables that are not money.** Rows of counts and areas are sometimes read in the money scale: Exxon's refining capacity keeps its caption's thousands and Chevron's does not. A caption reader for such tables, checked by ratios that must be plausible across companies.
- **The tables the ledger lists as unread.** Two pages of GM's 10-K (market share, dealerships) cost three questions in two fresh rounds.
- **The judge on more disputes.** It was right on all 13 disputes of the development rounds and gained one answer on a fresh round. That is too few to trust beyond FLAGGED.
- **Periods as intervals**, so a twelve-month period that is not the fiscal year, or a quarter restated in a later table, is read. **Page-image rows**: a vision model laid out 13 pages and arithmetic accepted 318 of 445 of its rows, none in 100 shuffled controls (`report/vision_rows.md`); left out because arithmetic does not prove a label.

## How I used AI

The brief says to use AI tools freely and to say how. Claude Code (several sessions of Claude Opus 5.5 and Claude Fable 5.2, with sub-agents) wrote all of the code, ran every experiment and drafted this README and the PDF. Blind agents wrote and re-derived the benchmark; red-team agents attacked each change before it was kept. I set the direction and made the rulings, recorded as they were made in `DECISIONS.md`:

- **The design (D1, D15)**: the most ambitious design, nothing dropped for being hard, general rather than tuned to a few companies. **The corpus (D2)**: forty latest 10-Ks with the banks, prior-year 10-Ks for restatements, 10-Qs for derived quarters. **The target (D11)**: accuracy on questions nobody has seen, hence the fresh rounds.
- **Fine-tuning (D3) and research before training (D13)**, both against the AI's plan: it advised against fine-tuning, then wanted GPU runs at once. I ruled for a small local planner, with success criteria written before any run.
- **Stop and revamp (D9, D10)**: I stopped the first version and ordered a red team of every claim and a revamp, not a patch. On the last night I stopped a version that served answers the text reader had written, because the brief says answers come from the graph, and had it rebuilt so the reader can only point at a graph node. **Resources and honesty (D5 to D8, D12, D14)**: the keys and accounts, one free key for the reported run, no checks by hand, a public repository, a learning layer that needs a second witness and can be withdrawn.

Layout: `tieout/` the package (build, links, plans, executor, text reader, `eval/`), `scripts/` build, ledger, evaluation and report scripts, `benchmark/` every round's questions (`round2/questions.jsonl` is the held-out file), `results/` every result file quoted here, `graph/` the built graph, `cache/` every model call the reported runs used, `finetune/` the planners' training, `baseline/` the retrieval pipeline, `report/` the PDF.
