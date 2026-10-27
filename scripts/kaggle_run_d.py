"""Kaggle driver for Run D (2x T4): clone the repository, install vLLM, run stage 60 for one model on both GPUs.

Pushed as a Kaggle script kernel, one kernel per model so the two models run in parallel. RUN below is
rewritten per launch. The repository is cloned to /tmp so the saved output (/kaggle/working) holds only
the result files: traces, rollouts, per-shard logs and run_d.log (versions, timings, token counts).
Rollout rows are appended per chunk, so a run cut short by its time limit keeps what finished.
"""
import os
import subprocess
import time

RUN = {"model": "Qwen/Qwen3-1.7B", "skip": 20, "n": 480, "m": 8, "max_gen": 6144, "limit_s": 23400,
       "branch": "story-and-benchmarks"}

START = time.time()
REPO = "/tmp/repo"
OUT = "/kaggle/working/run_d"


def sh(cmd: str, check: bool = True) -> int:
    print(f"\n$ {cmd}", flush=True)
    r = subprocess.run(cmd, shell=True)
    if check and r.returncode:
        raise SystemExit(f"failed ({r.returncode}): {cmd}")
    return r.returncode


def log(msg: str) -> None:
    line = f"[{time.strftime('%F %T')} +{(time.time() - START) / 60:.1f} min] {msg}"
    print(line, flush=True)
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/run_d.log", "a") as fh:
        fh.write(line + "\n")


sh(f"rm -rf {REPO} && git clone -q --depth 1 -b {RUN['branch']} https://github.com/HarshDubey23/early-failure-detection-cot {REPO}")
sh("pip install -q vllm 'math-verify[antlr4_13_2]' 2>&1 | tail -n 3")
sh("pip uninstall -y -q torchaudio", check=False)
sh("python -c \"import vllm, torch, transformers; print('vllm', vllm.__version__, 'torch', torch.__version__, "
   "'transformers', transformers.__version__, 'gpus', torch.cuda.device_count())\" | tee -a /tmp/versions.txt")
log(f"RUN {RUN} | {open('/tmp/versions.txt').read().strip()}")

model, name = RUN["model"], RUN["model"].split("/")[-1]
left = int(RUN["limit_s"] - (time.time() - START) - 300)
args = f"--model {model} --skip-problems {RUN['skip']} --n-problems {RUN['n']} --m {RUN['m']} --max-gen {RUN['max_gen']} --out {OUT}"
procs = [subprocess.Popen(f"cd {REPO} && CUDA_VISIBLE_DEVICES={g} timeout {left} python scripts/60_reasoning_rollouts.py "
                          f"{args} --shard {g}/2 > {OUT}/{name}.s{g}.out 2>&1", shell=True) for g in (0, 1)]
while any(p.poll() is None for p in procs):
    time.sleep(1800)
    rows = sum(1 for g in (0, 1) if os.path.exists(f"{OUT}/rollouts.{name}.s{g}.jsonl")
               for _ in open(f"{OUT}/rollouts.{name}.s{g}.jsonl"))
    log(f"{name}: running, {rows} cut rows so far")
codes = [p.returncode for p in procs]
stats = subprocess.run(f"cd {OUT} && cat traces.{name}.s*.jsonl 2>/dev/null | python -c \"import sys,json; r=[json.loads(x) for x in sys.stdin]; "
                       f"print(len(r),'traces', sum(not x['correct'] for x in r),'failed', sum(x['gen_tokens'] for x in r),'trace tokens', "
                       f"sum(x['finish']=='length' for x in r),'truncated')\"; cat rollouts.{name}.s*.jsonl 2>/dev/null | python -c \"import sys,json; "
                       f"r=[json.loads(x) for x in sys.stdin]; print(len(r),'cut rows', sum(sum(x['new_tokens']) for x in r),'continuation tokens')\"",
                       shell=True, capture_output=True, text=True).stdout.strip().replace("\n", " | ")
log(f"{name}: exit {codes} (124 = time limit), {stats}")
for g in (0, 1):
    sh(f"tail -n 5 {OUT}/{name}.s{g}.out", check=False)
log("done")
