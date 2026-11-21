"""Runs ON Kaggle (T4). LoRA fine-tune a small instruct LLM: question -> compact JSON plan.

Plain transformers + peft with a hand-written training loop (no Trainer, no Unsloth) so that
nothing depends on fast-moving library APIs. Steps: load -> LoRA -> train -> eval on val
(exact match + JSON validity) -> save adapter -> merge -> GGUF f16 -> q4_k_m -> Modelfile.
Outputs in /kaggle/working: metrics.json, train_log.txt, planner-q4_k_m.gguf, Modelfile, adapter/.
"""
# ============================== HYPERPARAMETERS ==============================
BASE_MODEL = "Qwen/Qwen3-1.7B"      # fallback: "Qwen/Qwen2.5-1.5B-Instruct"
EPOCHS = 3
LR = 2e-4
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
BATCH_SIZE = 8          # sequences per micro-batch
GRAD_ACCUM = 2          # effective batch = BATCH_SIZE * GRAD_ACCUM
WARMUP_FRAC = 0.05
WEIGHT_DECAY = 0.0
MAX_LEN = 2048          # prompt + plan tokens; longer rows are dropped (and counted)
NUM_CTX = 4096          # Ollama context window written into the Modelfile
GRAD_CHECKPOINT = True
EVAL_BATCH = 16
EVAL_MAX_NEW_TOKENS = 384
EVAL_LIMIT = 0          # 0 = whole val set
GGUF_QUANT = "Q4_K_M"   # llama-quantize type; falls back to q8_0 from the converter if the build fails
MODEL_NAME = "planner"  # output file stem
SEED = 435
# ============================================================================

import glob
import json
import math
import os
import random
import subprocess
import sys
import time

T0 = time.time()
WORK = "/kaggle/working"
SCRATCH = "/tmp/ft"
os.makedirs(SCRATCH, exist_ok=True)
LOG = open(os.path.join(WORK, "train_log.txt"), "w", buffering=1)
METRICS = {"base_model": BASE_MODEL, "hparams": {k: v for k, v in globals().items() if k.isupper() and k not in ("WORK", "SCRATCH", "LOG", "T0")}}


def log(*a):
    msg = f"[{time.time() - T0:7.1f}s] " + " ".join(str(x) for x in a)
    print(msg, flush=True)
    LOG.write(msg + "\n")


def sh(cmd, cwd=None, check=True):
    log("$", cmd)
    p = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    tail = (p.stdout + p.stderr)[-3000:]
    if tail.strip():
        LOG.write(tail + "\n")
    if check and p.returncode != 0:
        print(tail, flush=True)
        raise RuntimeError(f"command failed ({p.returncode}): {cmd}")
    return p


def save_metrics():
    METRICS["elapsed_min_total"] = round((time.time() - T0) / 60, 2)
    with open(os.path.join(WORK, "metrics.json"), "w") as fh:
        json.dump(METRICS, fh, indent=2)


# ---------------------------------------------------------------- data + env
def find(name):
    hits = sorted(glob.glob(f"/kaggle/input/**/{name}", recursive=True))
    if not hits:
        raise FileNotFoundError(f"{name} not found under /kaggle/input")
    return hits[0]


# Optional overrides.json in the dataset (written by push_kaggle.py --set KEY=VALUE).
_ov = glob.glob("/kaggle/input/**/overrides.json", recursive=True)
if _ov:
    for _k, _v in json.load(open(_ov[0])).items():
        if _k not in METRICS["hparams"]:
            raise KeyError(f"unknown hyperparameter override {_k}")
        globals()[_k] = _v; METRICS["hparams"][_k] = _v
    log("overrides", json.load(open(_ov[0])))

try:
    import peft  # noqa: F401
except ImportError:
    sh(f"{sys.executable} -m pip install -q peft")
import torch
import transformers
import peft
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model

random.seed(SEED); torch.manual_seed(SEED)
gpus = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
cc = torch.cuda.get_device_capability(0) if gpus else (0, 0)
DTYPE = torch.bfloat16 if cc[0] >= 8 else torch.float16   # T4 is sm_75 -> fp16
METRICS["env"] = {"python": sys.version.split()[0], "torch": torch.__version__, "transformers": transformers.__version__,
                  "peft": peft.__version__, "gpus": gpus, "dtype": str(DTYPE)}
log("env", METRICS["env"])

SYSTEM = open(find("system_prompt.txt"), encoding="utf-8").read().strip()
train_rows = [json.loads(l) for l in open(find("train.jsonl"), encoding="utf-8") if l.strip()]
val_rows = [json.loads(l) for l in open(find("val.jsonl"), encoding="utf-8") if l.strip()]
if EVAL_LIMIT:
    val_rows = val_rows[:EVAL_LIMIT]
log(f"train={len(train_rows)} val={len(val_rows)}")

# The chat template, written out by hand so the Ollama TEMPLATE below is byte-identical.
PROMPT_FMT = "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
END = "<|im_end|>"


def plan_text(plan):
    return json.dumps(plan, ensure_ascii=False, separators=(",", ":"))


tok = AutoTokenizer.from_pretrained(BASE_MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
END_ID = tok.convert_tokens_to_ids(END)


def encode(row):
    p = tok(PROMPT_FMT.format(system=SYSTEM, question=row["question"]), add_special_tokens=False)["input_ids"]
    t = tok(plan_text(row["plan"]) + END, add_special_tokens=False)["input_ids"]
    return {"input_ids": p + t, "labels": [-100] * len(p) + t}   # loss on the plan only


enc = [encode(r) for r in train_rows]
dropped = sum(len(e["input_ids"]) > MAX_LEN for e in enc)
enc = [e for e in enc if len(e["input_ids"]) <= MAX_LEN]
lens = [len(e["input_ids"]) for e in enc]
plan_lens = [sum(x != -100 for x in e["labels"]) for e in enc]
METRICS["data"] = {"train_rows": len(enc), "dropped_over_max_len": dropped, "val_rows": len(val_rows),
                   "mean_seq_tokens": round(sum(lens) / len(lens), 1), "max_seq_tokens": max(lens),
                   "max_plan_tokens": max(plan_lens), "train_tokens_per_epoch": sum(lens)}
log("data", METRICS["data"])


def batches(epoch):
    """Length-bucketed shuffled batches: less padding, same randomness."""
    r = random.Random(SEED + epoch)
    idx = list(range(len(enc))); r.shuffle(idx)
    mega = BATCH_SIZE * 32
    out = []
    for i in range(0, len(idx), mega):
        chunk = sorted(idx[i:i + mega], key=lambda j: lens[j])
        out += [chunk[k:k + BATCH_SIZE] for k in range(0, len(chunk), BATCH_SIZE)]
    r.shuffle(out)
    return out


def collate(ids):
    m = max(lens[j] for j in ids)
    x = torch.full((len(ids), m), tok.pad_token_id); y = torch.full((len(ids), m), -100); a = torch.zeros((len(ids), m), dtype=torch.long)
    for b, j in enumerate(ids):
        n = lens[j]
        x[b, :n] = torch.tensor(enc[j]["input_ids"]); y[b, :n] = torch.tensor(enc[j]["labels"]); a[b, :n] = 1
    return x.cuda(), y.cuda(), a.cuda()


# ---------------------------------------------------------------- model + LoRA
t = time.time()
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=DTYPE, attn_implementation="sdpa").cuda()
METRICS["load_min"] = round((time.time() - t) / 60, 2)
if GRAD_CHECKPOINT:
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
model.config.use_cache = False
model = get_peft_model(model, LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
                                         target_modules=TARGET_MODULES, task_type="CAUSAL_LM"))
for p in model.parameters():          # LoRA weights in fp32 so the fp16 GradScaler can unscale them
    if p.requires_grad:
        p.data = p.data.float()
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
log(f"trainable params {trainable / 1e6:.2f}M")

# ---------------------------------------------------------------- train
steps_per_epoch = math.ceil(len(batches(0)) / GRAD_ACCUM)
total_steps = steps_per_epoch * EPOCHS
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=WEIGHT_DECAY)
sched = get_cosine_schedule_with_warmup(opt, max(1, int(WARMUP_FRAC * total_steps)), total_steps)
try:
    scaler = torch.amp.GradScaler("cuda", enabled=(DTYPE == torch.float16))
except (AttributeError, TypeError):
    scaler = torch.cuda.amp.GradScaler(enabled=(DTYPE == torch.float16))
model.train()
t_train = time.time(); seen_tokens = 0; step = 0; losses = []
for ep in range(EPOCHS):
    bl = batches(ep)
    for i, ids in enumerate(bl):
        x, y, a = collate(ids)
        with torch.autocast("cuda", dtype=DTYPE):
            loss = model(input_ids=x, attention_mask=a, labels=y).loss / GRAD_ACCUM
        scaler.scale(loss).backward()
        seen_tokens += int(a.sum())
        if (i + 1) % GRAD_ACCUM == 0 or i + 1 == len(bl):
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step(); step += 1
            losses.append(loss.item() * GRAD_ACCUM)
            if step % 10 == 0 or step == total_steps:
                el = time.time() - t_train
                log(f"ep {ep + 1}/{EPOCHS} step {step}/{total_steps} loss {sum(losses[-10:]) / len(losses[-10:]):.4f} "
                    f"tok/s {seen_tokens / el:.0f} elapsed {el / 60:.1f}m")
train_s = time.time() - t_train
METRICS["train"] = {"minutes": round(train_s / 60, 2), "steps": step, "tokens_seen": seen_tokens,
                    "tokens_per_sec": round(seen_tokens / train_s), "final_loss": round(sum(losses[-10:]) / len(losses[-10:]), 4),
                    "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
# Projection for the real run: 10,000 pairs with the same mean length and epochs.
METRICS["train"]["projected_minutes_10k_pairs_same_epochs"] = round(train_s / 60 * 10000 / len(enc), 1)
log("train", METRICS["train"]); save_metrics()

model.save_pretrained(os.path.join(WORK, "adapter"))
tok.save_pretrained(os.path.join(WORK, "adapter"))

# ---------------------------------------------------------------- eval on val
model.eval()
if GRAD_CHECKPOINT:
    model.gradient_checkpointing_disable()
model.config.use_cache = True
tok.padding_side = "left"
t_eval = time.time(); preds = []
for i in range(0, len(val_rows), EVAL_BATCH):
    chunk = val_rows[i:i + EVAL_BATCH]
    b = tok([PROMPT_FMT.format(system=SYSTEM, question=r["question"]) for r in chunk], return_tensors="pt",
            padding=True, add_special_tokens=False).to("cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=DTYPE):
        out = model.generate(**b, max_new_tokens=EVAL_MAX_NEW_TOKENS, do_sample=False, eos_token_id=END_ID,
                             pad_token_id=tok.pad_token_id)
    preds += [tok.decode(o[b["input_ids"].shape[1]:], skip_special_tokens=True).strip() for o in out]
valid = exact = 0; by_op = {}; fails = []
for r, p in zip(val_rows, preds):
    try:
        obj = json.loads(p); ok_json = True
    except Exception:
        obj = None; ok_json = False
    hit = ok_json and obj == r["plan"]
    valid += ok_json; exact += hit
    key = "+".join(s["op"] for s in r["plan"]["steps"])
    d = by_op.setdefault(key, [0, 0]); d[0] += hit; d[1] += 1
    if not hit and len(fails) < 10:
        fails.append({"question": r["question"], "gold": plan_text(r["plan"]), "pred": p})
n = len(val_rows)
METRICS["eval_hf"] = {"n": n, "json_valid": round(valid / n, 4), "exact_match": round(exact / n, 4),
                      "by_shape": {k: f"{v[0]}/{v[1]}" for k, v in sorted(by_op.items())},
                      "seconds": round(time.time() - t_eval, 1), "sample_failures": fails}
log("eval", {k: v for k, v in METRICS["eval_hf"].items() if k != "sample_failures"}); save_metrics()

# ---------------------------------------------------------------- merge
t = time.time()
merged_dir = os.path.join(SCRATCH, "merged")
merged = model.merge_and_unload()
merged = merged.to(torch.float16)
merged.save_pretrained(merged_dir, safe_serialization=True)
tok.padding_side = "right"
tok.save_pretrained(merged_dir)
del model, merged; torch.cuda.empty_cache()
METRICS["merge_min"] = round((time.time() - t) / 60, 2)
log("merged ->", merged_dir)

# ---------------------------------------------------------------- GGUF export
t = time.time()
lc = os.path.join(SCRATCH, "llama.cpp")
sh(f"git clone --depth 1 https://github.com/ggml-org/llama.cpp {lc}")
METRICS["llama_cpp_commit"] = sh("git rev-parse --short HEAD", cwd=lc).stdout.strip()
sh(f"{sys.executable} -m pip install -q {lc}/gguf-py sentencepiece protobuf")
f16 = os.path.join(SCRATCH, f"{MODEL_NAME}-f16.gguf")
sh(f"{sys.executable} {lc}/convert_hf_to_gguf.py {merged_dir} --outtype f16 --outfile {f16}")
gguf_name = f"{MODEL_NAME}-{GGUF_QUANT.lower()}.gguf"
final = os.path.join(WORK, gguf_name)
try:
    if sh("which cmake", check=False).returncode != 0:
        sh(f"{sys.executable} -m pip install -q cmake")
    sh(f"cmake -S {lc} -B {lc}/build -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DBUILD_SHARED_LIBS=OFF -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release")
    sh(f"cmake --build {lc}/build --target llama-quantize -j {os.cpu_count()}")
    qbin = glob.glob(f"{lc}/build/**/llama-quantize", recursive=True)[0]
    sh(f"{qbin} {f16} {final} {GGUF_QUANT}")
    METRICS["gguf_quant"] = GGUF_QUANT
except Exception as e:  # fallback: pure-python q8_0, quantize locally with `ollama create -q q4_K_M`
    log("llama-quantize route failed, falling back to q8_0:", e)
    gguf_name = f"{MODEL_NAME}-q8_0.gguf"; final = os.path.join(WORK, gguf_name)
    sh(f"{sys.executable} {lc}/convert_hf_to_gguf.py {merged_dir} --outtype q8_0 --outfile {final}")
    METRICS["gguf_quant"] = "Q8_0 (fallback)"
METRICS["gguf_file"] = gguf_name
METRICS["gguf_mb"] = round(os.path.getsize(final) / 1e6, 1)
METRICS["gguf_min"] = round((time.time() - t) / 60, 2)
log("gguf", gguf_name, METRICS["gguf_mb"], "MB")

# ---------------------------------------------------------------- Ollama Modelfile
# TEMPLATE renders exactly PROMPT_FMT; the model then writes plan JSON and <|im_end|>.
modelfile = f'''FROM ./{gguf_name}
TEMPLATE """<|im_start|>system
{{{{ .System }}}}<|im_end|>
<|im_start|>user
{{{{ .Prompt }}}}<|im_end|>
<|im_start|>assistant
"""
SYSTEM """{SYSTEM}"""
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"
PARAMETER temperature 0
PARAMETER num_ctx {NUM_CTX}
PARAMETER num_predict {EVAL_MAX_NEW_TOKENS}
'''
with open(os.path.join(WORK, "Modelfile"), "w", encoding="utf-8") as fh:
    fh.write(modelfile)
with open(os.path.join(WORK, "val_predictions.jsonl"), "w", encoding="utf-8") as fh:
    for r, p in zip(val_rows, preds):
        fh.write(json.dumps({"question": r["question"], "gold": plan_text(r["plan"]), "pred": p}) + "\n")
save_metrics()
log("DONE", json.dumps({k: METRICS[k] for k in ("train", "gguf_file", "gguf_mb", "elapsed_min_total")}, default=str))
