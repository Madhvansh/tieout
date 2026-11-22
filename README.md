# TieOut

Questions about SEC filings, answered from a knowledge graph of the filings' own arithmetic. Every answer is read from graph nodes and comes with its proof (the filing, the page and the printed figure behind each number) and a tier that says how far to trust it. 55 filings of 40 companies. The reported run uses one free Gemini key and a laptop CPU.

## Results: 30 held-out questions, run once

Written by blind authors and committed before any run (`benchmark/round2/questions.jsonl`; how: [below](#how-the-benchmark-was-made)), run once on the frozen code (tag `frozen-v2`), reported as it came. **Precision**: of the facts an answer rests on (its proof leaves), the share that are gold supporting facts. **Recall**: of the gold supporting facts, the share the answer retrieved. Both are means over questions, computed by `tieout/eval/bench.py`. No model judges an answer.

<!-- gen:heldout (written by scripts/readme_numbers.py from results/; never edit by hand) -->
| Question type | Questions | Accuracy | Precision | Recall |
|---|---|---|---|---|
| Single fact | [[...]] | [[...]] | [[...]] | [[...]] |
| Multi-hop | [[...]] | [[...]] | [[...]] | [[...]] |
| Cross-filing comparison | [[...]] | [[...]] | [[...]] | [[...]] |
| **Overall** | [[...]] | [[...]] | [[...]] | [[...]] |

- On the same 30 questions: the retrieval baseline (passages, the same free model reads) [[...]]; TieOut's plans alone, without the text reader, [[...]] (`results/heldout_v2.json`, `results/heldout_baseline_hybrid.json`).
- Right by tier: [[...]]. Under the strict scorer of 21 November: [[...]].
- `python scripts/check_brief.py results/heldout_v2.json`: [[...]].
- Paid side experiment, is the model the bottleneck? (Claude Sonnet 5.5 plans and picks, gemini-3.1-pro arbitrates, labelled and never mixed into the headline): [[...]].
<!-- /gen:heldout -->

## Run it

Python 3.12, no GPU, no key: the built graph and every model call of the reported runs are in the repository.

```
pip install -r requirements.txt
python scripts/check_brief.py results/heldout_v2.json    # every proof leaf of every served answer is a graph node
python scripts/report_numbers.py                          # the numbers of this README and the PDF, from results/
TIEOUT_LLM=gemini:gemini-3.5-flash-lite python -m tieout ask --offline "What were Apple's net sales in the fourth quarter of fiscal 2026?"
TIEOUT_LINK_LLM=gemini:gemini-3.5-flash-lite python scripts/run_benchmark.py --bench benchmark/round2/questions.jsonl \
  --model gemini:gemini-3.5-flash-lite --checker ollama:tieout-planner-4b-v3 --pick-checker gemini:gemini-3.1-flash-lite \
  --arbiter gemini:gemini-3.6-flash,gemini:gemini-3.5-flash,gemini:gemini-3-flash-preview \
  --passage gemini:gemini-3.5-flash-lite --offline --name replay     # the held-out run, replayed from the cache
```

Without `--offline` the last command answers new questions: it needs a free `GEMINI_API_KEY` in `.env` and the fine-tuned planner in Ollama (`finetune/README.md`). Rebuild the graph from the PDFs with `python scripts/fetch_corpus.py`, `python -m tieout build`, `python scripts/build_raw.py`, `python scripts/add_headings.py`, `python scripts/ledger.py`.

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

The 55 filings hold 2,800 such checks (`results/graph_stats.json`). They verify a figure, solve for one printed nowhere (above), name a row by its place in the arithmetic, and explain a figure that changed between two filings (a restatement). Every other table row and every numeric sentence is a node too, with value, unit, period and page (52,761 rows, 192,730 cells, 15,124 sentence records), linked to a question when a question needs one.

![Merck's balance sheet, 10-K page 80, drawn by python -m tieout show: green brackets are recovered formulas, the answer cell is outlined in blue](report/fig_page.png)

## How a question is answered

1. **Plan.** A language model compiles the question into a typed plan (`benchmark/plan_language.txt`: get, calc, compare, rank, filter, series, members).
2. **Link.** A line outside the statements (a segment, a count, a sentence figure) is chosen among candidate rows shown with their labels, headings and columns, and no figures.
3. **Execute.** A deterministic executor runs the plan on the graph and returns the answer with its proof leaves, or abstains and names the missing fact.
4. **Four eyes.** No single model's choice is served as fact. Two planners read each question: a free hosted model (`gemini-3.5-flash-lite`) and a Qwen3-4B we fine-tuned on Kaggle, on the laptop CPU. Their plans are compared on what they compute; a stronger free model arbitrates only when they differ. Two pickers choose each line and are compared on the figure each pick yields. They are not fully independent: both pickers are Gemini models, and the 4B learned from paraphrases the hosted model wrote.
5. **A text reader: a witness and a pointer, never a source.** The baseline of this repository (passages of the filings, BM25 and `bge-small` vectors, the same free model reads them) is used inside TieOut in two ways only. As a witness it corroborates or disputes: the same figure makes a FLAGGED answer CORROBORATED; a different figure against a VERIFIED or WITNESSED answer sends the question to the arbiter. As a pointer it names a place (filing, page, figure); the node at the place it names is linked to the plan like any picker's choice, and the executor answers from the graph. Nothing the reader writes is served, and `scripts/check_brief.py` fails a run if any proof leaf of any served answer is not a node of the graph.

<!-- gen:devline -->
On the four latest development rounds (142 questions, `results/h3_r7.json` to `h3_r10.json`), the plans alone were right on 114 and the whole system on 122. `check_brief` passes on 4 of the 4 files (round 10: 31 answers served, 76 proof leaves, 76 of them graph nodes).
<!-- /gen:devline -->

| Tier | What the code promises (`tieout/execute.py` `trust`, `tieout/readers.py`) |
|---|---|
| VERIFIED | Every number is a statement line bound by structure (no model chose the row), inside a formula that holds or solved from such numbers; both planners agree. |
| WITNESSED | The same, but some lines were chosen by models, each choice by two models that agreed. |
| CORROBORATED | An answer with a named weakness that the text reader repeats, figure for figure. |
| FLAGGED | Answered, each weakness named. Every answer on a line the reader pointed at is here. |
| ABSTAIN | Not answered. Says what is missing. |

## How the benchmark was made

- **Corpus.** 55 filings of 40 companies across sectors, banks included (`data/manifest.csv`): the latest 10-K of each, the prior-year 10-K of ten (so restatements show) and five 10-Qs (so a quarter printed nowhere can be derived), fetched from SEC EDGAR by `scripts/fetch_corpus.py`.
- **Questions.** Three blind authors, language-model agents that could read only the text of the filings and never saw the system, wrote 69 candidates, each with a golden answer, its document and page, and the supporting facts as (document, page, printed value). Author A's questions were read by the developer while the system was still changing, so all of A is development. The 30 held-out questions come only from authors B and C, drawn by a seeded script (`scripts/split_round2.py`): 10 single-fact, 10 multi-hop, 10 cross-filing. The split was committed (`1d7346f`) before any run; the other 39 are development.
- **Gold, verified blind.** Another agent re-derived all 30 answers from the pages without seeing the authors' gold. A script compared the two: 28 of 30 agree. A third agent judged the other 2 to be differences of form, and the author's gold stands (`benchmark/round2/adjudication.json`).
- **Sealed until the freeze.** Nobody who changed the system read the held-out questions. Every later development round passed a leak gate (`scripts/check_leakage.py`) that drops near copies of held-out questions without anyone reading them.
- **LLM use.** Authors, checker and adjudicator are language models. Answers are scored by a script against keys the system did not write.

## Development: we measured our own overfitting

The first development set flattered the system, so the work ran in rounds. Blind authors (a new persona each round: analyst, auditor, credit analyst, journalist, test engineer) write a fresh round; the leak gate runs; the system answers it once, and that first run is recorded before anyone reads an answer. Only then are the misses fixed, by class, each fix with a test, and the round becomes development material. First runs stay well below the same rounds on the current code. Part of that is adaptation; part may be that fresh questions are harder (Recht et al. 2019 saw a similar drop on fresh ImageNet test sets with no reuse at all). Six scoring rules were relaxed on 22 November after misses were read, then tightened against a red team's counter-examples, so numbers are also given under the strict scorer of 21 November.

<!-- gen:rounds -->
| Fresh round | First run, final scorer | First run, strict scorer | Current code |
|---|---|---|---|
| 3 | 25/36 | 25/36 | 34/36 |
| 4 | 23/34 | 22/34 | 32/35 |
| 5 | 20/36 | 20/36 | 34/36 |
| 6 | 17/27 | 17/27 | 29/35 |
| 7 | 23/36 | 20/36 | 32/36 |
| 8 | 29/35 | 27/35 | 32/35 |
| 9 | 26/36 | 21/36 | 31/36 |
| 10 | 26/35 | 21/35 | 27/35 |
| **All** | 189/275 (69%) | 173/275 (63%) | 251/284 (88%) |

Files: `results/pooled_first_runs.json` (first runs rescored under both scorers), `results/first_runs/`, `results/h3_r<N>.json` (current code). Round 10's row is the first run of the corrected design, made before anyone read the round's answers; a version withdrawn the same night (it served answers the text reader wrote) had run it once before.
<!-- /gen:rounds -->

## Where it fails, by class

[[HELDOUT-FAILURES: the held-out misses by class, read after the run]] On round 10's first run (`results/first_runs/round10_pointer.json`) the nine misses fall into five classes. **The planners decline what a filing prints** (NVIDIA's employee turnover rate, 3.7 percent on 10-K page 99). **A plan stops at a line it cannot place** (Coca-Cola's fall in headcount). **A look-alike line** (Walmart's square feet per unit: a sentence on page 40 against the Item 2 table on page 29). **A line the build did not read, or read in the wrong scale** (GM's 6,276 dealerships on 10-K page 7 are held by no node; Exxon's refining capacity carries its caption's thousands and Chevron's does not, so their ratio comes out 1,895 instead of 1.79). **Readings** (a balance grown over one quarter read three ways by three planners; a ranking read off a chart).

## The JD's four directions, measured on this graph

[[JD: at most eight lines from research/20_lcg_counterparts.md, each with its file]]

## What did not work

- **Serving an answer the text reader wrote.** An earlier version filled abstentions with the reader's own answer: 27 of 35 on round 10's first run (`results/first_runs/round10.json`). It broke the brief's rule that answers come from the graph, so it was rebuilt to point only: 26 on the same round (`results/first_runs/round10_pointer.json`).
- **A stronger model alone.** On round 7, `gemini-3.1-pro` planning alone on the graph scored 22 of 36 and flash-lite alone 19 (`results/v_r7_pro.json`, `results/v_r7_fl.json`, code of 22 November).
- **The link store on new questions.** Round 8 scored 30 of 35 with the links learned on rounds 1 to 7 and 30 of 35 without them (`results/store_round8.json`, `results/nolinks_round8.json`): stored links help a question asked again, not a new one.
- **A model judge to admit a binding** (97.0 percent precise, but it threw away 34 of 66 right proposals: `results/judge_vs_gate.json`), and **relaxing a scoring rule after reading a miss** without a counter-example test (hence the strict scorer).

**The fine-tuned planners.** Qwen3 models fine-tuned on Kaggle GPUs (LoRA; plans in the plan language for templated questions, paraphrased by the hosted model, leak-checked against the held-out questions), published with cards: [4B v3](https://huggingface.co/Madhvansh/tieout-planner-qwen3-4b-v3-gguf) (the second planner), [4B v2](https://huggingface.co/Madhvansh/tieout-planner-qwen3-4b-v2-gguf), [1.7B v3.1](https://huggingface.co/Madhvansh/tieout-planner-qwen3-1.7b-v3.1-gguf), [1.7B v2](https://huggingface.co/Madhvansh/tieout-planner-qwen3-1.7b-v2-gguf), [0.6B v2](https://huggingface.co/Madhvansh/tieout-planner-qwen3-0.6b-v2-gguf). Planning alone on rounds 1 to 5 and 6 to 8 (`results/v32_table_all.json`): hosted flash-lite 175 of 194 and 75 of 106, 4B v3.2 153 and 52, 8B v3.2 160 and 57. [[8B: seconds per question on the laptop CPU, and its card]]

## What I would do next

- **A judge between two sets of lines.** When the linker's rows and the rows the reader points at disagree, show both sets to the arbiter with their headings and no figures; the executor computes on the set it picks.
- **Periods as intervals**, so a twelve-month period that is not the fiscal year, or a quarter restated in a later table, is read. **Page-image rows**: a vision model laid out 13 pages and arithmetic accepted 318 of 445 of its rows, none in 100 shuffled controls (`vision_lab`); left out because arithmetic does not prove a label.

## How I used AI

The brief says to use AI tools freely and to say how. Claude Code (several sessions of Claude Opus 5.5 and Claude Fable 5.2, with sub-agents) wrote all of the code, ran every experiment and drafted this README and the PDF. Blind agents wrote and re-derived the benchmark; red-team agents attacked each change before it was kept. I set the direction and made the rulings, recorded as they were made in `DECISIONS.md`:

- **The design (D1, D15)**: the most ambitious design, nothing dropped for being hard, general rather than tuned to a few companies. **The corpus (D2)**: forty latest 10-Ks with the banks, prior-year 10-Ks for restatements, 10-Qs for derived quarters. **The target (D11)**: accuracy on questions nobody has seen, hence the fresh rounds.
- **Fine-tuning (D3) and research before training (D13)**, both against the AI's plan: it advised against fine-tuning, then wanted GPU runs at once. I ruled for a small local planner, with success criteria written before any run.
- **Stop and revamp (D9, D10)**: I stopped the first version and ordered a red team of every claim and a revamp, not a patch. On the last night I stopped a version that served answers the text reader had written, because the brief says answers come from the graph, and had it rebuilt so the reader can only point at a graph node. **Resources and honesty (D5 to D8, D12, D14)**: the keys and accounts, one free key for the reported run, no checks by hand, a public repository, a learning layer that needs a second witness and can be withdrawn.

Layout: `tieout/` the package (build, links, plans, executor, text reader, `eval/`), `scripts/` build, ledger, evaluation and report scripts, `benchmark/` every round's questions (`round2/questions.jsonl` is the held-out file), `results/` every result file quoted here, `graph/` the built graph, `cache/` every model call the reported runs used, `finetune/` the planners' training, `baseline/` the retrieval pipeline, `report/` the PDF.
