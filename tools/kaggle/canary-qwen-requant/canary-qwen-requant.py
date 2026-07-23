"""Re-quantize canary-qwen and ship a WORKING small variant.

The published canary-qwen-2.5b-q4_k.gguf produces NaN logits (all-'!' output);
q8_0 is fine. Root cause (Kaggle quant-diff + GGUF header): tensor TYPES are
correct (token_embd/output F16, only the 196 Qwen3-1.7B LLM projections are
Q4_K) — so it is 4-bit *precision* on this small LLM, not a wrong-tensor bug.

This kernel downloads the F16 source and re-quantizes with today's quantizer at
q4_k, then q5_k, then q6_k, validating EACH by transcribing jfk (must hit the
gold key-words) with the NaN-guarded binary. It uploads the SMALLEST passing
variant:
  - q4_k passes            -> upload q4_k (overwrites the broken file in place).
  - q4_k fails, q5/q6 pass -> atomically replace: delete broken q4_k + add the
                              smallest passing k-quant.
  - none pass              -> delete the broken q4_k; q8_0 stays the only quant.
Full-harness utilities; CPU build (validation is CPU-fine, no CUDA needed).
"""

import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

_T0 = time.time()

REPO = Path("/kaggle/working/CrispASR")
BUILD = REPO / "build"
WORK = Path("/kaggle/temp")  # gotcha #22: stage big files off /kaggle/working
WORK.mkdir(parents=True, exist_ok=True)
MODELS = WORK / "models"
MODELS.mkdir(parents=True, exist_ok=True)
HF_REPO = "cstr/canary-qwen-2.5b-GGUF"


def run(cmd, **kw):
    kw.setdefault("capture_output", True)
    kw.setdefault("text", True)
    return subprocess.run(cmd, **kw)


# ── clone FIRST, then import the harness from the clone (repo carries it) ────
print(json.dumps({"step": "start"}), flush=True)
if REPO.exists():
    shutil.rmtree(REPO)
run(["git", "clone", "--recursive", "--depth", "1",
     "https://github.com/CrispStrobe/CrispASR.git", str(REPO)], capture_output=False)
run(["git", "-C", str(REPO), "submodule", "update", "--init", "--recursive", "--depth", "1"],
    capture_output=False, timeout=1800)

sys.path.insert(0, os.path.join(str(REPO), "tools", "kaggle"))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
import kaggle_harness as kh  # noqa: E402

kh.init_progress()
sha = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
kh.step("cloned", sha=sha)

# ── toolchain + token ───────────────────────────────────────────────────────
kh.install_build_toolchain()
TOKEN = kh.resolve_hf_token("HF_TOKEN")
from huggingface_hub import HfApi, hf_hub_download, CommitOperationAdd, CommitOperationDelete  # noqa: E402

# ── build (CPU): crispasr-cli + crispasr-quantize ───────────────────────────
kh.step("configure")
cfg = ["cmake", "-G", "Ninja", "-B", str(BUILD), "-S", str(REPO),
       "-DCMAKE_BUILD_TYPE=Release", "-DCRISPASR_NO_C2PA_NATIVE=ON"] + kh.cache_and_link_flags()
r = run(cfg)
if r.returncode != 0:
    kh.step("configure.FAIL", tail=(r.stderr or "")[-800:]); raise SystemExit(1)
kh.step("build")
with kh.build_heartbeat("build"):
    r = run(["cmake", "--build", str(BUILD), "--target", "crispasr-cli", "crispasr-quantize", "-j",
             str(os.cpu_count())])
if r.returncode != 0:
    kh.step("build.FAIL", tail=(r.stderr or "")[-1200:]); raise SystemExit(1)
CLI = BUILD / "bin" / "crispasr-cli"
QUANT = BUILD / "bin" / "crispasr-quantize"
for b in (CLI, QUANT):
    if not b.exists():
        kh.step("build.MISSING", missing=str(b)); raise SystemExit(1)
kh.step("build.done")

# ── download F16 source ─────────────────────────────────────────────────────
kh.step("download.f16.begin")
with kh.build_heartbeat("download.f16"):
    f16 = Path(hf_hub_download(repo_id=HF_REPO, filename="canary-qwen-2.5b-f16.gguf",
                              local_dir=str(MODELS), token=TOKEN))
kh.step("download.f16.done", gb=round(f16.stat().st_size / 1e9, 2))

JFK = REPO / "samples" / "jfk.wav"
JFK_KEYS = ["fellow", "americans", "country", "ask"]  # gold content words


def _norm(t):
    return re.findall(r"[a-z]+", t.lower())


def validate(gguf):
    """Transcribe jfk with canary-qwen; return (ok, hits, transcript)."""
    r = run([str(CLI), "-m", str(gguf), "--backend", "canary-qwen", "-f", str(JFK),
             "--chunk-seconds", "0", "--no-prints"], timeout=1200)
    lines = [ln.strip() for ln in (r.stdout or "").splitlines() if ln.strip()]
    txt = lines[-1] if lines else ""
    words = set(_norm(txt))
    hits = sum(k in words for k in JFK_KEYS)
    ok = r.returncode == 0 and hits >= 3
    return ok, hits, txt[:180]


# ── quantize + validate, smallest-first ─────────────────────────────────────
ORDER = ["q4_k", "q5_k", "q6_k"]
results = {}
winner = None
for qt in ORDER:
    out = MODELS / f"canary-qwen-2.5b-{qt}.gguf"
    with kh.build_heartbeat(f"quantize.{qt}"):
        r = run([str(QUANT), str(f16), str(out), qt], timeout=1800)
    if not out.exists():
        results[qt] = {"quantized": False, "tail": (r.stderr or "")[-300:]}
        kh.step(f"quantize.{qt}.FAIL", tail=(r.stderr or "")[-300:]); continue
    sz = round(out.stat().st_size / 1e9, 2)
    with kh.build_heartbeat(f"validate.{qt}"):
        ok, hits, txt = validate(out)
    results[qt] = {"quantized": True, "gb": sz, "hits": hits, "pass": ok, "transcript": txt}
    kh.step(f"validate.{qt}", gb=sz, hits=hits, ok=ok, transcript=txt)
    print(f"  [{qt}] {sz}GB hits={hits}/4 pass={ok} :: {txt!r}", flush=True)
    if ok:
        winner = qt
        break  # smallest passing variant found
    out.unlink(missing_ok=True)  # free disk; a failed quant is never uploaded

# ── ship the outcome ────────────────────────────────────────────────────────
api = HfApi(token=TOKEN)
BROKEN = "canary-qwen-2.5b-q4_k.gguf"
if winner == "q4_k":
    with kh.build_heartbeat("upload.q4_k"):
        api.upload_file(path_or_fileobj=str(MODELS / BROKEN), path_in_repo=BROKEN,
                        repo_id=HF_REPO, repo_type="model",
                        commit_message="Fix q4_k: re-quantized F16 (prev file produced NaN logits / all-'!'); ASR-validated on jfk")
    kh.step("uploaded.q4_k", action="replaced broken q4_k in place")
elif winner in ("q5_k", "q6_k"):
    newname = f"canary-qwen-2.5b-{winner}.gguf"
    with kh.build_heartbeat(f"upload.{winner}"):
        api.create_commit(repo_id=HF_REPO, repo_type="model",
                          operations=[
                              CommitOperationAdd(path_in_repo=newname, path_or_fileobj=str(MODELS / newname)),
                              CommitOperationDelete(path_in_repo=BROKEN),
                          ],
                          commit_message=f"q4_k is NaN-corrupt on this small LLM; replace with ASR-validated {winner.upper()} (smallest working k-quant)")
    kh.step(f"uploaded.{winner}", action=f"deleted broken q4_k; added {newname}")
else:
    with kh.build_heartbeat("delete.q4_k"):
        api.delete_file(path_in_repo=BROKEN, repo_id=HF_REPO, repo_type="model",
                        commit_message="Remove NaN-corrupt q4_k (no k-quant validates on this small LLM; use q8_0)")
    kh.step("deleted.q4_k", action="no k-quant passed; removed broken q4_k, q8_0 remains")

RESULTS = {"results": results, "winner": winner, "wall_s": round(time.time() - _T0, 1)}
(WORK / "results.json").write_text(json.dumps(RESULTS, indent=2))
print(json.dumps({"step": "done", "winner": winner, "results": results}), flush=True)
kh.step("done", winner=winner or "none")
