# The local planner: no API key anywhere

TieOut needs a language model for two narrow jobs: writing a plan from a question (the planner) and,
the first time a question names a line outside the three statements, picking that line among
labelled candidates (the linker). Neither job ever sees a number. A small model fine-tuned on data
the graph writes itself does both on a laptop CPU.

## Use it

    # one-time: get the GGUF and its Modelfile (see the model card for the download), then
    ollama create tieout-planner -f Modelfile
    # answer questions with no key at all
    set TIEOUT_LLM=ollama:tieout-planner
    set TIEOUT_LINK_LLM=ollama:tieout-planner-4b
    python -m tieout ask "Which of Salesforce's service offerings grew fastest in fiscal 2026?"

With no key in `.env` and no `TIEOUT_LLM`, `tieout/llm.py` already defaults to `ollama:tieout-planner`.
Ollama runs it on the CPU (`num_gpu 0` is set by `tieout/llm.py`).

## Make it again

    python finetune/make_data.py                 # 9,000 training and 600 validation pairs, every plan validated
    python finetune/push_kaggle.py --data-dir data_v2 --system system_prompt.v2.txt --out out/v2_q17 ^
        --dataset-slug tieout-planner-v2-data --kernel-slug tieout-planner-v2-q17 --set EPOCHS=1
    # larger base model on Colab: open finetune/train_colab.ipynb from GitHub and run all
    python finetune/plan_with_hf.py MODEL_DIR plans.json   # on the training GPU: plans for the dev questions

Files: `make_data.py` (data from the graph's vocabulary), `train_kaggle.py` (LoRA training, merge,
GGUF export; runs on Kaggle, Colab or any CUDA machine), `push_kaggle.py` (private Kaggle dataset and
kernel, waits, downloads), `train_colab.ipynb`, `colab_fetch.py` (brings a 2.5 GB GGUF back in parts),
`plan_with_hf.py`, `publish.py` (Hugging Face or a GitHub release), `MODEL_CARD.md`.

## What was measured

Exact plan match on 600 held-out synthetic phrasings: Qwen3-0.6B 73.8%, Qwen3-1.7B 83.5%, Qwen3-4B 88.2%.
On the 87 development questions written by people: see `results/a_planner_table.json`.
What did not help: JSON-schema constrained decoding. Forced into a generic grammar of the plan language,
the 1.7B model emitted keys in the schema's order instead of its trained order and filled optional
fields it never uses, and both test plans got worse; the deterministic plan normalizer
(`tieout/plan.py`) handles its small slips instead.
