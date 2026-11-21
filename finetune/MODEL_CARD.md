---
license: apache-2.0
base_model: Qwen/Qwen3-1.7B
tags: [text-to-query, semantic-parsing, finance, sec-filings, knowledge-graph, gguf, ollama]
---

# tieout-planner

A small model that turns a question about US public companies into a query plan for the TieOut
knowledge graph (github.com/Madhvansh/tieout). It never states a figure: every number in an answer
comes from the graph, which recovers the formulas of SEC filings and checks each value against them.
The model's only job is to translate words into the graph's vocabulary, so the whole system runs on
a laptop CPU with no API key.

## What it does

Input: a question. Output: a JSON plan in TieOut's plan language (`benchmark/plan_language.txt`):
reads of shared concepts, metrics and line items, arithmetic, growth, rankings, filters, counts,
aggregates, series, breakdowns into segments, restatement checks, or "cannot" for what filings do
not print. Example:

    Q: Which of Microsoft's segments grew revenue fastest in fiscal 2025, and how much did it bring in?
    A: {"steps":[{"id":"s1","op":"members","company":"MSFT","of":"revenue by segment","period":{"fy":2025}},
                 {"id":"s2","op":"rank","over":"$s1","by":{"growth":{"value":true}},"period":{"fy":2025},"order":"desc","k":1},
                 {"id":"s3","op":"get","company":"$s2","period":{"fy":2025}}],"answer":"s3"}

## Training data: written by the graph itself

9,000 (question, plan) pairs sampled from the graph's own vocabulary: its 40 companies and their
common names, 47 shared concepts, 17 metrics, text attributes, and the segment, product and region
labels that the filings print and the graph's recovered formulas tie to revenue or operating income.
A plan is sampled first, so it is valid by construction, and then put into words with a frame and
ways of naming the company, the measure and the period. Every target plan passes the project's
validator (`tieout.plan.validate`). No benchmark question, no held-out question and no answer value
is in the data; the validation set (600 pairs) uses frames and phrasings that never appear in
training. Script: `finetune/make_data.py`; data: `finetune/data_v2/`.

## Method

LoRA (rank 16, alpha 32, all attention and MLP projections), one epoch, learning rate 2e-4, cosine
schedule, loss on the plan tokens only, the shared system prompt run once per step as a cached prefix.
Plain transformers and peft, merged, converted to GGUF with llama.cpp and quantized to Q4_K_M.
Script: `finetune/train_kaggle.py` (Kaggle T4) and `finetune/train_colab.ipynb` (Colab L4).

## Measured

Exact plan match on the 600 held-out phrasings (synthetic):

| base model | training time | exact match | valid JSON | GGUF |
|---|---|---|---|---|
| Qwen3-0.6B | 10 min, T4 | 73.8% | 99.8% | 397 MB |
| Qwen3-1.7B | 15 min, T4 | 83.5% | 100% | 1.1 GB |
| Qwen3-4B-Instruct-2507 | 29 min, L4 | 88.2% | 100% | 2.5 GB |

End to end on TieOut's 87 development questions (written by people reading the PDFs, not from the
training frames), with the graph and the local linker: [filled from results/a_t_*.json].

## Limits

It was trained on phrasings produced from frames, so long or unusual analyst questions are harder for
it than for large hosted models (see the development table). It knows only the 40 companies and the
vocabulary of this graph; for another corpus, regenerate the data with `make_data.py` from that graph.

## Use

    ollama create tieout-planner -f Modelfile      # Modelfile in this repository, next to the GGUF
    TIEOUT_LLM=ollama:tieout-planner TIEOUT_LINK_LLM=ollama:tieout-planner python -m tieout ask "..."

Runs on CPU (`num_gpu 0`): about 4 seconds a question for the 1.7B model on a laptop.
