# TieOut

Questions about SEC filings, answered from a graph, with the page and the printed number behind every answer.

A 10-K is a spreadsheet whose formulas were erased. TieOut puts the formulas back, found from the arithmetic alone (no model, no XBRL), and keeps them as the graph. The structure then does the work: it checks every number, solves for numbers printed nowhere, names rows that have no label, explains why a figure changed between two filings, and reads a PDF whose fonts have no Unicode map. A language model only turns a question into a typed plan, and picks a printed row by its wording with every number masked. It never sees or writes a number.

Every answer carries a tier: VERIFIED (every source cell sits inside a passing check), FLAGGED (answered, with a named weakness) or ABSTAIN (says what is missing).

![A statement page with its recovered formulas](report/fig_page.png)

*Merck's balance sheet (10-K page 80), drawn by `python -m tieout show`. Green brackets are recovered formulas. The two rows with no label are total assets. The answer cell is outlined in blue.*

## Results

Every score is computed by script against answer keys the system did not write: the SEC's own data, or gold answers written by blind authors and re-derived by other blind agents. No language model judges any answer.

**Held-out benchmark.** 30 questions written by authors who never saw the system, committed before any run, run once on the frozen code (tag `frozen-v2`). Planner for this run: [PRIMARY PLANNER: name]. Files: `results/heldout_<planner>.json` [HELD-OUT: files]. Precision and recall are macro averages over questions, as the brief defines them.

| Type | Questions | Accuracy | Precision | Recall |
|---|---|---|---|---|
| Single-fact | 10 | [HELD-OUT] | [HELD-OUT] | [HELD-OUT] |
| Multi-hop | 10 | [HELD-OUT] | [HELD-OUT] | [HELD-OUT] |
| Cross-filing | 10 | [HELD-OUT] | [HELD-OUT] | [HELD-OUT] |
| **Overall** | 30 | [HELD-OUT] | [HELD-OUT] | [HELD-OUT] |
| Overall without B26 (see the benchmark section) | 29 | [HELD-OUT] | [HELD-OUT] | [HELD-OUT] |

Accuracy by tier (VERIFIED, FLAGGED, ABSTAIN) and the count of silent wrong answers (wrong and served as VERIFIED): [HELD-OUT: trust table].

**Round-1 history.** The first version of this project scored 24 of 30 on its own held-out set (`results/benchmark_tieout.json`, tag `frozen-before-benchmark`: single-fact 9/10, multi-hop 10/10, cross-filing 5/10). Those 30 questions are now development material, so that number is history, not the headline.

**Development.** Questions the system was built against: round 1 (30 and 18) and round 2 development (39). Not held-out. Runs made on the graph before its last rebuild at 07:50 on 22 Nov. The linker was the fine-tuned 4B model in every row except round-1 hand-written plans (Groq gpt-oss-120b) [CHECK AFTER FREEZE: reruns on the final code].

| Planner | Round-2 dev (39) | Round 1 (30) | Round-1 dev (18) | Files |
|---|---|---|---|---|
| Fine-tuned Qwen3 0.6B, local | 21 | 18 | 11 | `results/a_t_ft06_*.json` |
| Fine-tuned Qwen3 1.7B, local | 23 | 21 | 15 | `results/a_t_ft17_*.json` |
| Fine-tuned Qwen3 4B, local | 23 | 22 | 16 | `results/a_t_ft4b_*.json` |
| Gemini 3.5 Flash-Lite, free tier | 28 | not run | not run | `results/a_t_gemini_r2dev.json` |
| Hand-written plans, no planner | 32 | 28 | not run | `results/a_r2dev_oracle_link4b.json`, `results/v2_wip_round1_oracle.json` |

The gap between hand-written plans and the best planner is the cost of the language model. The remaining misses with hand-written plans are the graph's.

**Baseline.** Passage retrieval over the same filings with a small embedding model, then a free LLM reads the passages and does the arithmetic itself. Held-out: [HELD-OUT: baseline accuracy, precision, recall, `results/heldout_baseline_dense.json`; reader model `gemini-3.5-flash-lite`]. On development questions, before any model is called, the gold page is among the top 12 passages for 57.2% of gold facts with dense retrieval and 41.5% with dense and keyword fused (`baseline/retrieval_eval.json`, 81 questions). Dense was chosen on those development numbers, which favours the baseline. The first version's page-level baseline scored 10 of 30 on round 1 (`results/baseline.json`).

**What the graph holds** (`results/graph_stats.json`): 55 filings (40 latest 10-Ks, 10 prior-year 10-Ks, 5 third-quarter 10-Qs) from 40 companies. 13,314 statement readings, 12,613 of them (94.7%) inside a passing check. 2,708 checks: 991 SUM, 358 TIE, 89 RATIO, 1,270 CARRY. 4,158 facts under 47 concepts, 1,774 active bindings. 750 text facts. 51,438 rows and 188,073 cells from every other table. 14,759 sentences holding 46,775 numbers.

### Claims and how to re-run them

Results keyed by outside data first, then the benchmark.

| Claim | Number | File | Re-run |
|---|---|---|---|
| Recovered formulas agree with the SEC's calculation linkbase | 416 of 418 declared relations recovered (314 under the strict mapping). Same arithmetic as declared: IS 97.0%, BS 99.0%, CF 93.7% | `results/lab_linkbase.json` | `python scripts/linkbase_check.py analyze` |
| Scale test keyed by the SEC's data, no model | 8,667 questions: 7,833 correct, 709 abstained, 125 silent wrong (1.4%) | `results/scale_test.json` | `python scripts/run_scale_test.py` |
| Concept bindings against the SEC's data | 2,874 of 2,912 agree (98.7%); 10,714 of 12,160 readings confirmed by value and period (88.1%) | `results/xbrl_audit.json` | `python scripts/run_audit.py` |
| Other tables against the SEC's data | 18,723 of 91,954 cells confirmed; the rest could not be confirmed, which is not a count of errors. 0 tables at the wrong scale, 22 of 11,499 columns with the wrong period | `results/xbrl_audit_raw.json` | `python scripts/audit_raw.py` |
| Development facts are nodes on their own page | 247 of 247 (196 statement, 42 other table, 9 sentence) | `results/addressability.json` | `python scripts/addressability.py --same-page --out results/addressability.json` |
| Fault injection, final graph (164 tables, about 1,300 faults of each kind) | Share caught by the checks, then with two outside channels added (a figure in a sentence pins a table's scale; last year's column in the earlier filing pins row order): whole table at the wrong scale 1.8% to 59.1%, two rows swapped 25.0% to 50.8%, digit error 84.9% to 88.0%. Unchanged: sign flip 99.5%, value in the wrong column 99.8%, one cell at the wrong scale 100%, dropped row 77.7%. Intervals in the file | `results/faults.json` | `python scripts/run_faults.py` |
| Fault injection, first session's own study | Whole table at the wrong scale 2.5%, two rows swapped 26.7% (sign flip 99.1%, wrong column 99.8%, cell scale 100%, digit error 87.0%, dropped row 78.9%) | `results/faults_v1.json` | not repeatable (first graph) |
| Damage curve: checks off against on | A share of statement cells is corrupted, the graph rebuilt, every one of the 4,158 facts graded against the clean graph (mean of 5 seeds). At 5% corrupted: 342.2 silent wrong facts with checks off, 64.4 with checks on. Cost: with checks on, 1,030.0 facts are withheld (990.8 of them right); at 0% corruption the checks already withhold 116 right facts | `results/damage_curve.json` | `python scripts/run_damage.py` |
| Held-out benchmark | [HELD-OUT] | `results/heldout_*.json` | `python scripts/run_benchmark.py --bench benchmark/round2/questions.jsonl --model <MODEL> --name heldout_<short>` |
| Held-out baseline | [HELD-OUT] | `results/heldout_baseline_dense.json` | `python scripts/run_baseline.py benchmark/round2/questions.jsonl --model gemini:gemini-3.5-flash-lite --mode dense --out results/heldout_baseline_dense.json` |
| Ablations, learned-store transfer, cost per question | [HELD-OUT] | `results/heldout_*.json` | `--ablate concepts\|solve\|link\|checks\|store` on the same command |

## What the structure does

**1. Verify.** Every recovered formula is a check, and a number inside a passing check is verified. In J&J's cash-flow statement (10-K page 53) the fiscal 2023 financing column does not add up to its printed total of (15,825): a stray 47 with no label had been read into it. The formula fails, the nine cells above that total are marked failed, and the other two columns of the table stay verified. Measured: 94.7% of readings are verified; the checks catch 99.5% of injected sign flips (`results/faults.json`). Their blind spots are measured too (swapped rows, a whole table at the wrong scale), and listed under what failed.

**2. Solve.** A number printed nowhere is answered from the one equation where it is the only unknown. Google's fourth-quarter 2025 revenue appears in no filing. It is 402,836 (10-K page 50) minus 289,007 (third-quarter 10-Q, page 7) = $113.829 billion, with both sources verified. Measured: of 156 scale-test questions that need a solved value, 148 are correct, 8 abstain, none are wrong (`results/scale_test.json`).

**3. Name.** Rows are bound to concepts by their place in the arithmetic, and a label only confirms. Merck prints its total assets on page 80 with no label at all. The row is bound because its 136,866 equals the sum of the lines above it and equals the total liabilities and equity printed below. Measured: 2,874 of 2,912 concept bindings agree with the SEC's data (`results/xbrl_audit.json`).

**4. Explain.** A figure that changed between two filings is diagnosed. Apple's FY2025 10-K (page 34) prints other non-current assets of 83,727. The FY2026 10-K (page 37) prints 72,634 for the same date. The difference is 11,093, which is a new line, intangible assets, net. Total non-current assets (211,284) and total assets (359,241) did not move, so it is a reclassification, and the filing's own sentence (page 40) is quoted. Measured: 18 events at 6 of the 10 companies with two annual reports: 6 reclassifications, 5 restatements, 7 presentation changes (`results/restatements.json`).

**5. Read.** The McDonald's PDF has fonts with no Unicode map, so its text layer is glyph codes. They are decoded as a cipher: glyph widths against standard font metrics, then vocabulary, with the statement arithmetic as the certificate. Of 281,068 glyph occurrences, 117,558 are settled by width and order, 163,440 by vocabulary, and 70 stay undecoded (`data/decoded_fonts.json`). After decoding, 70 of 71 balance-sheet readings, 93 of 96 cash-flow readings and 57 of 73 income-statement readings are verified (`results/graph_stats.json`).

## How it works

| Step | What happens | Code |
|---|---|---|
| 1. Words | Each PDF is read once into word boxes. Fonts with no Unicode map are decoded | `tieout/pdfio.py`, `glyphs.py` |
| 2. Tables | Columns from the right edges of numbers, periods from header dates, scale from the caption. Statements that run over two pages are joined | `tieout/tables.py`, `statements.py`, `alltables.py` |
| 3. Formulas | A row is a subtotal only if a signed run of rows above it sums to it in every period column | `tieout/calc.py` |
| 4. Graph | Readings, checks (SUM, TIE, RATIO, CARRY), fiscal periods, facts, text facts | `tieout/graph.py`, `periods.py`, `textfacts.py` |
| 5. Every other table and sentence | Rows, cells and numeric sentences kept as nodes with value, unit, period and page | `tieout/rawlayer.py`, `mentions.py` |
| 6. Concepts | 47 seed concepts as data. A label is a proposal that structure must confirm. Absence ("this company does not report this") is stored as a fact | `semantics/seed_concepts.yaml`, `tieout/concepts.py`, `living.py` |
| 7. Links | A question phrase is tied once to a printed row or sentence. The model sees labels with numbers masked, structure vouches for the number, the link is stored with the question that taught it | `tieout/link.py` |
| 8. Changes | Reclassification, restatement or presentation change between two filings, with the filing's sentence | `tieout/restate.py` |
| 9. Plan | A model writes a typed plan from the plan language, never from data | `tieout/planner.py`, `plan.py`, `benchmark/plan_language.txt` |
| 10. Execute | Deterministic traversal: get, calc, rank, filter, aggregate, series, members. Answer with proof leaves and a tier, or an abstention that names the missing fact | `tieout/execute.py`, `ask.py` |
| 11. Evaluate | Scoring rules fixed before any run, fault injection, SEC-keyed tests | `tieout/eval/` |

## Run it

Python 3.12, no GPU. The built graph and the model-call cache are in the repo.

```
pip install -r requirements.txt       # also installs tieout itself
pytest -q                             # 109 passed on 22 Nov
python -m tieout stats                # what the shipped graph holds
```

Draw a statement page with its recovered formulas (needs the PDF from `scripts/fetch_corpus.py`, no model):

```
python -m tieout show --doc MRK_FY2025_10K --page 80 --out page.png
```

Replay the benchmark from cached model calls, no key:

```
python scripts/run_benchmark.py --offline
python -m tieout ask --offline "What was Apple's gross margin in fiscal 2025?"
```

[CHECK AFTER FREEZE: both offline commands replay only what the cache holds. On 22 Nov at 11:50 they answered 2 of 30 round-1 questions and abstained on the rest, because the cached plans were written for an older prompt. The other session is fixing this.]

Ask your own question, with no key (the fine-tuned local model, served by Ollama on the CPU) or with a free key:

```
set TIEOUT_LLM=ollama:tieout-planner
python -m tieout ask "How many Walmart Supercenters were operating in the U.S. at the end of fiscal 2025?"
set GEMINI_API_KEY=...
set TIEOUT_LLM=gemini:gemini-3.5-flash-lite
python -m tieout ask "What was Amazon's gross margin in 2025?"
```

[CHECK AFTER FREEZE: model download link on Hugging Face, `finetune/README.md`, [PENDING: link].]

Rebuild from the PDFs (about 260 MB from each company's own site; URLs and sha256 in `data/manifest.csv`) [CHECK AFTER FREEZE: not re-run in the write-up pass]:

```
python scripts/fetch_corpus.py
python -m tieout build
python scripts/build_raw.py
python scripts/teach.py              # coverage pass: bind what the statements print, record what a company lacks
```

`python scripts/clean_clone_check.py` clones the public repository into a fresh folder, installs it and runs the commands above. Its last report is `results/clean_clone_check.json` [CHECK AFTER FREEZE: it was not passing on 22 Nov at 02:18].

## The benchmark and how it was prepared

**Held-out set (round 2).**

- **Written blind.** Three AI agents, each allowed to read only the PDFs and the manifest (no code, no graph, no output), wrote 69 candidate questions. Each gold answer has its document, page, printed value and derivation. Author A wrote 18 single-fact and 9 multi-hop questions, B 9 multi-hop and 18 cross-filing, C 10 single-fact and 5 multi-hop.
- **Split by a seeded script** (`scripts/split_round2.py`). The developer sessions had read author A's questions, so all of A is development material. The 30 held-out questions come only from B and C: 10 single-fact (all of C's), 10 multi-hop (C's 5 and 5 of B's), 10 cross-filing (10 of B's). The other 39 are the development set, `benchmark/round2/dev.jsonl`.
- **Committed before any run** (commit `1d7346f`, 21 Nov 21:12).
- **Checked by independent agents, not by me.** A separate blind agent re-derived all 30 gold answers from the PDFs without seeing the gold. A script compared them: 28 agree. For the other 2, a third agent read the pages and found differences of form only, so the author's gold stands (`benchmark/round2/adjudication.json`).
- **Never read by whoever changed the system.** Checked by script: `scripts/check_leakage.py` looks for every held-out question and answer, and close paraphrases, in every prompt, example pool, training file, learned store, cached call, oracle plan and result (`results/leakage_check.json`). It flags two questions. **B26 is a near paraphrase (overlap 0.87) of a development question written independently by another blind author.** That is why the table above also reports the held-out score without B26. H04's gold answer quotes a sentence that the filing itself contains, so the sentence is in the mentions file, which is built from the filings.
- **Frozen, then run once.** Tag `frozen-v2`, then one run per planner, result files committed untouched. Anything fixed afterwards is reported separately as "after the run" and is not held-out.
- **Scoring rules fixed before any run** (`tieout/eval/bench.py`). Retrieved facts are every leaf the executor read, including every company scanned for a ranking. A gold fact is (document, page, printed value). A fact matches when company and value agree. Abstaining is correct only on an unanswerable question.

**Round 1** (`benchmark/questions.jsonl`, 30, and `benchmark/dev.jsonl`, 18). 48 candidates written by one agent allowed to read only the PDFs, committed before any run (`879976a`), split by a seeded script. The frozen first run scored 24 of 30. A later blind re-derivation (kept outside the repo, `bench2/round1_recheck.json`) agreed with the gold in substance for 30 of 30; five questions have a second defensible reading. I did not check any gold answer by hand. All checks were automated, by independent agents.

## What failed

[HELD-OUT: the failed held-out questions, grouped by stage, with the printed numbers.]

Failure classes already measured on development material:

- **Rows bound to the wrong concept are the silent-wrong class.** The scale test has 125 silent wrong answers out of 8,667 (28 interest expense, 11 effective tax rate, 9 total equity), and 46 of them equal a second defensible definition of the concept. Example: Apple's long-term debt for FY2026, key 68,208, was read as 9,906 from a line labelled "Term debt" on page 37, served as verified. A binding audit that lets the statements' own arithmetic veto such bindings was committed on 22 Nov at 11:50 (`5acb63c`) [CHECK AFTER FREEZE: scale test rerun, `results/scale_test.json`].
- **A planner slip served as VERIFIED.** Coca-Cola's third-quarter operating income: gold 3,982 million, answered 11.921 billion, the nine-month figure (Gemini planner on development, `results/a_t_gemini_r2dev.json`). The graph was right about the number it read; the plan asked for the wrong period.
- **Wording the linker could not bridge.** "Net new money" for BlackRock, where the filing says "total net inflows": no link, abstained. A ranking of four banks by CET1 ratio abstained because one bank had no fact (same file).
- **A wrong row served as VERIFIED.** Morgan Stanley's compensation ratio came out as 3.51% against a gold of 41.4%, served as VERIFIED with hand-written plans (`results/a_r2dev_oracle_link4b.json`). A rounded sentence figure was served where the table prints the exact one (AMD Data Center: $16.600 billion against $16,635 million, FLAGGED).
- **Checks are blind to symmetric errors.** Two neighbouring rows trading values keep every sum intact: 25.0% are caught by the checks alone, 50.8% once last year's column from the earlier filing is added. A whole table at the wrong scale is caught only when another number pins it: 1.8% by the checks alone, 59.1% once a figure stated in a sentence is added (`results/faults.json`). The first session's own study had reported 26.7% and 2.5% (`results/faults_v1.json`). Half the swapped rows and four tables in ten at the wrong scale still go unseen.
- **Parser errors that checks do see.** The J&J stray 47 above. 22 of 11,499 columns in other tables carry the wrong period (`results/xbrl_audit_raw.json`).
- **The checks withhold right answers.** With 5% of statement cells corrupted, checks off serve 3,777 facts correctly and 342.2 silently wrong (mean of 5 seeds, 4,158 facts). Checks on serve 2,786 correctly and 64.4 silently wrong, flag 277.8 and withhold 1,030.0, of which 990.8 were right: one bad cell unverifies its whole formula. Even with no corruption, 116 right facts are withheld because no formula covers them (`results/damage_curve.json`). The first version at 10% corruption showed the same trade on the benchmark: 20.3 of 30 correct with 5.0 wrong when serving everything, 9.7 correct with 3.0 wrong when verified-only (`results/noise_frozen_v1.json`).
- **Thirty questions is a small sample.** The intervals are wide. The scale test (8,667 keyed questions, no model) measures the graph; the 30 questions measure the whole system.

## How I used AI

The brief says to use AI tools freely and to say how. I treated it as running a small research team: I set the goal and the rules, made the design and evaluation decisions, and used Claude Code (Anthropic's coding agent) as the engineering team that built, measured and wrote.

**What the AI did.** It wrote all of the code in this repository. It surveyed prior work, laid out design options with their risks, red-teamed the design before and after it was built, ran every experiment, and drafted this README and the results PDF. Separate sub-agents, each allowed to read only the filings, wrote the benchmark questions and their gold answers. Other sub-agents, blind to those answers, re-derived every gold answer from the PDFs. A disagreement went to a third agent that read the pages and ruled. The work ran as three parallel Claude Code sessions (question side, graph side, review). Each stopped at usage limits and resumed, coordinated through a shared log file.

**What I decided.** These are the calls that shaped the project, in order:

1. **The design.** I was offered three: a graph of formulas recovered from the arithmetic, a lighter version without the checks, and a graph that learns from questions behind an admission gate. I asked for the first and the third fused into one system, and ruled that nothing be dropped for being hard.
2. **The corpus.** Forty latest annual reports across five sectors with the big banks included, the previous annual report for ten companies so that restatements show up, and five quarterly reports so that a quarter printed nowhere can be derived.
3. **A fine-tuned local planner.** The literature survey and the red team both advised against it. I overruled them, because a system that needs no API key is one anyone can run. Result on development: the fine-tuned 4B model answered 23 of 39 round-2 development questions against 28 for Gemini 3.5 Flash-Lite. [HELD-OUT: result on the held-out questions.]
4. **A second, independent red team.** After the first version scored 24 of 30 on its held-out questions, I did not accept its own account of itself. I ordered a fresh review that treated every claim of the first build as something to test. It found, among other things, statements cut at a page break, net income bound to the wrong line at three companies, a filing dismissed as unreadable that could be decoded, and gold answers that had never been checked. This version is the result.
5. **The evaluation rules.** A new held-out set written by authors who never saw the system, committed to git before any run, checked by blind re-derivation, and run once after the code was frozen. The headline number is that run, whatever it is. The first version's result is shown beside it as history. Anything fixed after the run is reported separately and labelled.
6. **No checks by hand.** I ruled that checks be done by independent automated passes rather than by me, and that this README say so instead of claiming hand checks.
7. **Compute.** All training ran on cloud GPUs (Google Colab and Kaggle), none on my laptop. I set up the accounts and the API keys (Gemini, Groq and Hugging Face free tiers).
8. **How the work was organised.** I split the revamp into parallel sessions with separate ownership of files and a shared coordination log, so that they could work through the night without overwriting each other.
9. **The name.** TieOut, after the auditor's habit of tracing every figure to its source and to everything it must agree with.

**Language models inside the system.** Only free options, as the brief asks. [PRIMARY PLANNER: name] writes plans. A model picks a printed row from its wording, with every number masked, when a question names a line that has no link yet. The fine-tuned local model does both with no key (`finetune/README.md`, model card `finetune/MODEL_CARD.md`; trained on data the graph wrote from its own vocabulary: exact plan match on 600 held-out synthetic phrasings was 73.8% for 0.6B, 83.5% for 1.7B and 88.2% for 4B). Every call is cached in `cache/llm`. No number in the graph or in an answer ever passes through a language model.

**If asked "did you write this code?"** No. I directed it. I chose between the alternatives, set what counted as evidence, decided what was reported and how, and I can walk through every module and every number in this README.

## Repository map

```
tieout/        the package (see "How it works")
semantics/     seed concepts and metrics as data; learned links, absence facts, review queue
benchmark/     round-1 questions, round-2 development and held-out questions, plan language, hand-written plans
baseline/      the passage-retrieval baseline
finetune/      training-data generator, Kaggle and Colab scripts, model card
scripts/       build, evaluate, audit, scale test, figures
results/       every result file the numbers above come from
graph/         the built graph, the raw layer, sentence mentions
data/          manifest (URLs, sha256), text facts, decoded fonts
cache/         every model call, so results replay offline
tests/         unit tests
report/        the two-page results PDF and its figure
```
