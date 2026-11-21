"""Write the fine-tuned planner's plans for the development questions, on a GPU, with transformers.

Used on Colab, where the merged model already sits on disk after training: the plans (a small
JSON file) come back instead of the 2.5 GB model, and `scripts/run_benchmark.py --oracle <file>`
then runs them through the graph on the laptop. Reads the development sets only.

Usage: python finetune/plan_with_hf.py MODEL_DIR OUT.json
"""
import json
import pathlib
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEV_SETS = ["benchmark/questions.jsonl", "benchmark/dev.jsonl", "benchmark/round2/dev.jsonl"]   # never the held-out set
PROMPT_FMT = "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"


def main() -> None:
    model_dir, out = sys.argv[1], sys.argv[2]
    system = (ROOT / "finetune" / "system_prompt.v2.txt").read_text(encoding="utf-8").strip()
    rows = [json.loads(line) for f in DEV_SETS for line in (ROOT / f).read_text(encoding="utf-8").splitlines() if line.strip()]
    tok = AutoTokenizer.from_pretrained(model_dir)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16).cuda().eval()
    end = tok.convert_tokens_to_ids("<|im_end|>")
    plans, t0 = {}, time.time()
    for i in range(0, len(rows), 16):
        chunk = rows[i:i + 16]
        batch = tok([PROMPT_FMT.format(system=system, question=r["question"]) for r in chunk], return_tensors="pt",
                    padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            gen = model.generate(**batch, max_new_tokens=384, do_sample=False, eos_token_id=end, pad_token_id=tok.pad_token_id)
        for r, g in zip(chunk, gen):
            text = tok.decode(g[batch["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            try:
                plans[r["id"]] = json.loads(text)
            except json.JSONDecodeError:
                plans[r["id"]] = {"steps": [], "answer": None, "cannot": "planner reply is not JSON"}
    pathlib.Path(out).write_text(json.dumps(plans, separators=(",", ":")), encoding="utf-8")
    print(f"{len(plans)} plans in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
