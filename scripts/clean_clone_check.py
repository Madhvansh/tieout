"""Clean-clone check: does the public repository run for a stranger who has only the clone?

  python scripts/clean_clone_check.py                      # clone, install, test, replay, compare
  python scripts/clean_clone_check.py --dest D:\\somewhere  # use another clone folder
  python scripts/clean_clone_check.py --no-compare         # skip the same replays in this working tree

What it does, in plain words:
  1. Clones the public repository (branch v2) into a fresh folder and records the commit.
  2. Makes a new virtual environment there and runs `pip install -r requirements.txt`.
  3. Runs `python -m pytest -q` in the clone.
  4. Runs what a reader would run, with no API key and no network for the model:
       - `python -m tieout stats`
       - `python -m tieout ask "<first round-1 question>"` from the cached plan,
       - `scripts/run_benchmark.py --offline` on round 1 and on round-2 development, once with
         the cached planner plans and once with the hand-written plans (`--oracle`).
     Each replay is then run again in THIS working tree, and the two accuracies are compared.
     A replay passes only if it finishes, makes no live LLM call, and scores the same in both.
  5. Checks that no tracked file is larger than 50 MB and that no secret is tracked
     (a .env or key file, or a key-shaped string).

It makes no LLM call, uses no GPU and no process pool. It never touches the held-out round-2
files: it replays only benchmark/questions.jsonl and benchmark/round2/dev.jsonl, and the secret
scan skips benchmark/round2/. It deletes the clone folder only if a previous run of this
script made it (marker file inside .git); any other existing folder stops the run.

Output: one line per step ("PASS"/"FAIL", seconds) and results/clean_clone_check.json.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
URL = "https://github.com/Madhvansh/tieout"
MARKER = "clean_clone_check_marker"
REPLAY_MODEL = "gemini:gemini-3.5-flash-lite"
MAX_BYTES = 50 * 1024 * 1024
SECRET_NAMES = re.compile(r"(^|/)(\.env|id_rsa[^/]*|credentials[^/]*\.json|[^/]*\.(pem|key|p12|pfx))$")
SECRET_TEXT = [("google key", re.compile(r"AIza[0-9A-Za-z_\-]{35}")),
               ("groq key", re.compile(r"gsk_[A-Za-z0-9]{20,}")),
               ("hugging face token", re.compile(r"hf_[A-Za-z0-9]{30,}")),
               ("openai style key", re.compile(r"sk-[A-Za-z0-9]{32,}")),
               ("github token", re.compile(r"gh[pousr]_[A-Za-z0-9]{36,}")),
               ("private key block", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"))]

STEPS: list[dict] = []
REPLAYS: list[dict] = []
HELDOUT = False                                  # set by --with-heldout


def clean_env(extra: dict | None = None) -> dict:
    env = {k: v for k, v in os.environ.items()
           if not (k.startswith("TIEOUT_") or k in {"GEMINI_API_KEY", "GROQ_API_KEY", "HF_TOKEN", "PYTHONPATH"})}
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env.update(extra or {})
    return env


def run(cmd: list[str], cwd: Path, env: dict | None = None, timeout: int = 3600) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, cwd=str(cwd), env=env or clean_env(), capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=timeout)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired as e:
        return 124, f"timeout after {timeout}s\n{(e.stdout or '')}"


def step(name: str, fn) -> bool:
    t0 = time.time()
    try:
        ok, detail = fn()
    except Exception as e:                       # a crash in a step is a failure of that step
        ok, detail = False, f"{type(e).__name__}: {e}"
    sec = round(time.time() - t0, 1)
    STEPS.append({"step": name, "ok": bool(ok), "seconds": sec, "detail": detail})
    print(f"[{'PASS' if ok else 'FAIL'}] {name} ({sec} s)  {str(detail).splitlines()[0][:150] if detail else ''}", flush=True)
    return bool(ok)


def tail(text: str, n: int = 12) -> str:
    return "\n".join(text.strip().splitlines()[-n:])


def _rm_readonly(func, path, _exc):
    os.chmod(path, stat.S_IWRITE)
    func(path)


def venv_python(dest: Path) -> Path:
    return dest / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def prepare_dest(dest: Path) -> None:
    if dest.exists():
        if (dest / ".git" / MARKER).exists():
            shutil.rmtree(dest, onerror=_rm_readonly)
        else:
            raise RuntimeError(f"{dest} exists and was not made by this script; remove it or pass --dest")


def do_clone(dest: Path, branch: str, url: str):
    def f():
        prepare_dest(dest)
        rc, out = run(["git", "clone", "-q", "-b", branch, url, str(dest)], ROOT)
        if rc:
            return False, tail(out)
        (dest / ".git" / MARKER).write_text("made by scripts/clean_clone_check.py\n")
        rc, sha = run(["git", "rev-parse", "HEAD"], dest)
        n = len(run(["git", "ls-files"], dest)[1].split())
        return True, f"commit {sha.strip()} ({n} tracked files)"
    return f


def do_venv(dest: Path, py: str):
    def f():
        rc, out = run([py, "-m", "venv", str(dest / ".venv")], dest)
        if rc:
            return False, "venv: " + tail(out)
        rc, out = run([str(venv_python(dest)), "-m", "pip", "install", "-q", "--disable-pip-version-check",
                       "-r", "requirements.txt"], dest)
        return rc == 0, "pip install -r requirements.txt: " + (tail(out, 6) or "ok")
    return f


def do_pytest(dest: Path):
    def f():
        rc, out = run([str(venv_python(dest)), "-m", "pytest", "-q", "-p", "no:cacheprovider"], dest)
        return rc == 0, tail(out, 6)
    return f


def do_stats(dest: Path):
    def f():
        rc, out = run([str(venv_python(dest)), "-m", "tieout", "stats"], dest)
        try:
            d = json.loads(out[out.index("{"):])
        except Exception:
            return False, tail(out)
        return rc == 0, "stats keys: " + ", ".join(list(d)[:6])
    return f


def do_ask(dest: Path):
    def f():
        first = (dest / "benchmark" / "questions.jsonl").read_text(encoding="utf-8").splitlines()[0]
        q = json.loads(first)["question"]
        # `ask` has no --offline flag; naming the replay model makes it find the cached plan
        rc, out = run([str(venv_python(dest)), "-m", "tieout", "ask", q], dest,
                      env=clean_env({"TIEOUT_LLM": REPLAY_MODEL}))
        line = next((l for l in out.splitlines() if l.startswith("ANSWER:")), tail(out, 6))
        ok = rc == 0 and "ANSWER:" in out and "planner failed" not in line   # a cache miss shows as an abstention
        return ok, f"Q: {q[:70]} | {line[:120]}"
    return f


def read_result(path: Path) -> dict:
    d = json.loads(path.read_text(encoding="utf-8"))
    rows = d.get("rows", [])
    return {"correct": sum(1 for r in rows if r.get("correct")), "n": len(rows), "complete": d.get("complete"),
            "live": (d.get("llm") or {}).get("live"), "cached": (d.get("llm") or {}).get("cached"),
            "ok_ids": sorted(r["id"] for r in rows if r.get("correct"))}


# The declared configuration (README, "Four eyes, four tiers"): two readers, two pickers, an arbiter chain.
# Offline, every one of these calls comes from cache/llm, so no key and no local model server is needed.
DECLARED = ["--model", "gemini:gemini-3.5-flash-lite", "--checker", "ollama:tieout-planner-4b-v3",
            "--arbiter", "gemini:gemini-3.6-flash,gemini:gemini-3.5-flash,gemini:gemini-3-flash-preview",
            "--pick-checker", "gemini:gemini-3.1-flash-lite"]
DECLARED_ENV = {"TIEOUT_LINK_LLM": "gemini:gemini-3.5-flash-lite"}


def replay(cwd: Path, py: str, bench: str, oracle: str | None, out_json: Path) -> tuple[dict | None, str, bool]:
    """Run scripts/run_benchmark.py --offline in cwd and read its results file from out_json.
    Without hand-written plans (`oracle`) the run is the declared configuration, replayed from the
    call cache. Third value: True if the plain command failed with "No module named 'tieout'" and
    only worked (or was measured) after PYTHONPATH was set to the repository root."""
    results_dir = cwd / "results"
    results_dir.mkdir(exist_ok=True)
    name = os.path.relpath(out_json.with_suffix(""), results_dir).replace("\\", "/")
    cmd = [py, "scripts/run_benchmark.py", "--offline", "--bench", str(cwd / bench), "--name", name]
    env = clean_env(DECLARED_ENV)
    if oracle:
        if not (cwd / oracle).exists():
            return None, f"{oracle} is missing", False
        cmd += ["--oracle", oracle]
    else:
        cmd += DECLARED
    rc, out = run(cmd, cwd, env=env)
    needs_path = False
    if rc and "No module named 'tieout'" in out:
        needs_path = True
        rc, out = run(cmd, cwd, env=clean_env({**DECLARED_ENV, "PYTHONPATH": str(cwd)}))
    if rc or not out_json.exists():
        return None, tail(out, 8), needs_path
    return read_result(out_json), "", needs_path


def do_replays(dest: Path, compare: bool, wt_python: str):
    cases = [("round 8 (the last fresh round), declared configuration", "benchmark/round8/dev.jsonl", None),
             ("round-2 development, declared configuration", "benchmark/round2/dev.jsonl", None),
             ("round 1, declared configuration", "benchmark/questions.jsonl", None)]
    if HELDOUT:                                       # only after the freeze and the one held-out run
        cases.insert(0, ("held-out round 2, declared configuration", "benchmark/round2/questions.jsonl", None))
    tmp = Path(tempfile.mkdtemp(prefix="cleanclone_wt_"))
    try:
        for i, (label, bench, oracle) in enumerate(cases):
            def f(label=label, bench=bench, oracle=oracle, i=i):
                clone_res, err, needs_path = replay(dest, str(venv_python(dest)), bench, oracle, dest / "results" / f"cc_replay_{i}.json")
                rec = {"label": label, "bench": bench, "oracle": oracle, "clone": clone_res, "clone_error": err,
                       "needs_pythonpath_in_clone": needs_path}
                if compare:
                    wt_res, werr, _ = replay(ROOT, wt_python, bench, oracle, tmp / f"wt_{i}.json")
                    rec["worktree"], rec["worktree_error"] = wt_res, werr
                REPLAYS.append(rec)
                if clone_res is None:
                    return False, f"{label}: clone crashed or missing input: {err.splitlines()[-1] if err else ''}"
                acc = f"{clone_res['correct']}/{clone_res['n']}"
                notes = []
                if needs_path:
                    notes.append("README command fails: No module named 'tieout' (scripts/ is on sys.path, not the repo root)")
                if clone_res["live"]:
                    notes.append(f"{clone_res['live']} live LLM calls")
                if clone_res["complete"] is False:
                    notes.append("incomplete")
                if compare:
                    w = rec["worktree"]
                    if w is None:
                        notes.append(f"worktree run failed: {rec['worktree_error'][-80:]}")
                    else:
                        acc += f" (working tree {w['correct']}/{w['n']})"
                        if w["ok_ids"] != clone_res["ok_ids"]:
                            diff = sorted(set(w["ok_ids"]) ^ set(clone_res["ok_ids"]))
                            notes.append("differs on " + ",".join(diff[:12]))
                return not notes, f"{label}: clone {acc}" + ("; " + "; ".join(notes) if notes else "")
            step(f"replay: {label}", f)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def do_hygiene(dest: Path):
    def f():
        rc, out = run(["git", "ls-files", "-z"], dest)
        files = [x for x in out.split("\0") if x]
        big = [(x, (dest / x).stat().st_size) for x in files if (dest / x).is_file() and (dest / x).stat().st_size > MAX_BYTES]
        names = [x for x in files if SECRET_NAMES.search(x) and not x.endswith(".example")]
        hits = []
        for x in files:
            if x.startswith("benchmark/round2/"):
                continue
            p = dest / x
            try:
                if not p.is_file() or p.stat().st_size > 5 * 1024 * 1024:
                    continue
                text = p.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for kind, rx in SECRET_TEXT:
                if rx.search(text):
                    hits.append(f"{x}: {kind}")      # file and kind only, never the value
        bad = []
        if big:
            bad.append("over 50 MB: " + ", ".join(f"{n} ({s // 2**20} MB)" for n, s in big))
        if names:
            bad.append("secret-like names: " + ", ".join(names))
        if hits:
            bad.append("secret-like text: " + ", ".join(hits[:10]))
        return not bad, "; ".join(bad) or f"{len(files)} tracked files, largest {max((dest / x).stat().st_size for x in files if (dest / x).is_file()) // 1024} KB, no secrets"
    return f


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dest", default=str(Path(os.path.expanduser("~")) / "Downloads" / "tieout_clean_check"))
    ap.add_argument("--branch", default="v2")
    ap.add_argument("--url", default=URL)
    ap.add_argument("--wt-python", default=None,
                    help="python with the requirements installed, for the working-tree replays (default: .venv here, else this python)")
    ap.add_argument("--no-compare", action="store_true", help="do not rerun the replays in this working tree")
    ap.add_argument("--with-heldout", action="store_true",
                    help="also replay the held-out round (only after the freeze and its one run; the replay is offline)")
    a = ap.parse_args()
    global HELDOUT
    HELDOUT = a.with_heldout
    dest = Path(a.dest).resolve()
    wt_python = a.wt_python or (str(venv_python(ROOT)) if venv_python(ROOT).exists() else sys.executable)
    t0 = time.time()

    dirty = run(["git", "status", "--porcelain", "--untracked-files=no"], ROOT)[1].strip().splitlines()
    head = run(["git", "rev-parse", "HEAD"], ROOT)[1].strip()
    print(f"working tree {ROOT}\n  HEAD {head}, {len(dirty)} modified tracked files (a replay here uses them; the clone does not)")
    if not step("1 clone", do_clone(dest, a.branch, a.url)):
        return finish(a, dest, head, dirty, t0)
    if step("2 venv + pip install", do_venv(dest, sys.executable)):
        step("3 pytest -q", do_pytest(dest))
        step("4a python -m tieout stats", do_stats(dest))
        step("4b python -m tieout ask (cached plan)", do_ask(dest))
        do_replays(dest, not a.no_compare, wt_python)
    step("5 size and secrets", do_hygiene(dest))
    return finish(a, dest, head, dirty, t0)


def finish(a, dest, head, dirty, t0) -> int:
    commit = next((s["detail"].split()[1] for s in STEPS if s["step"] == "1 clone" and s["ok"]), None)
    out = {"command": "python scripts/clean_clone_check.py", "url": a.url, "branch": a.branch, "dest": str(dest),
           "clone_commit": commit, "worktree_head": head, "worktree_modified_tracked_files": dirty,
           "seconds": round(time.time() - t0, 1), "all_passed": all(s["ok"] for s in STEPS),
           "steps": STEPS, "replays": [{k: v for k, v in r.items() if k not in ("clone_error", "worktree_error")}
                                       | {"clone_error": r["clone_error"][-400:], "worktree_error": r.get("worktree_error", "")[-400:]}
                                       for r in REPLAYS]}
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "clean_clone_check.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\n{'ALL PASSED' if out['all_passed'] else 'FAILURES'}: {sum(s['ok'] for s in STEPS)}/{len(STEPS)} steps, {out['seconds']} s; "
          "wrote results/clean_clone_check.json")
    return 0 if out["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
