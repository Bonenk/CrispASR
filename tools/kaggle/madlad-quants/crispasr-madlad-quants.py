# %% [markdown]
# # CrispASR — build the missing madlad400-3b-mt F16 and Q8_0 GGUFs (issue #333)
#
# `cstr/madlad400-3b-mt-GGUF` ships ONLY `madlad400-3b-mt-q4_k.gguf`, but its
# README lists F16 and Q8_0 and its own copy-paste examples tell you to download
# `…-q8_0.gguf` — which 404s. That is what #333 reports.
#
# This runs on Kaggle rather than locally for one reason: the source is an
# 11.76 GB fp32 safetensors and the artifacts are ~5.7 GB + ~3.1 GB, so it is
# ~21 GB of transfer on a fast HF link. There is no GPU compute here at all —
# the converter is numpy and `crispasr-quantize` streams tensors — but Kaggle
# CPU workers get no internet, so a GPU kernel is the only way to get the link.
#
# Order matters and follows the port-pipeline rule "never crash before a
# produced artifact is checkpointed to HF":
#
#     download source → convert F16 → VALIDATE → upload F16 → delete source
#                     → quantize Q8_0 → VALIDATE → upload Q8_0 → delete F16
#
# so a failure in the quantize step cannot lose the F16 that was already paid
# for. Every artifact is validated by an actual translation before it is
# uploaded — a GGUF that loads is not a GGUF that works.
#
# NOT done here: a `-ref.gguf`. That needs `tools/reference_backends/madlad.py`
# and a `madlad` arm in `crispasr_diff_main.cpp`, neither of which exists yet —
# see the note at the bottom of this file.

# %% [code]
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

WORK = Path("/kaggle/working")

# ── Kaggle regime: clone CrispASR + import the harness (bundled = fallback) ────
# gotcha #26: a SCRIPT kernel runs only its code_file, so the harness and the
# converter must come from the in-kernel clone, not from bundled siblings.
CRISPASR_URL = "https://github.com/CrispStrobe/CrispASR.git"
REPO = Path("/kaggle/temp/CrispASR")  # gotcha #22: keep /kaggle/working small
REPO.parent.mkdir(parents=True, exist_ok=True)
if not REPO.exists():
    try:
        subprocess.check_call(["git", "clone", "--depth", "1", CRISPASR_URL, str(REPO)])
        sys.path.insert(0, str(REPO / "tools" / "kaggle"))
    except Exception:
        pass
if str(REPO / "tools" / "kaggle") not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
import kaggle_harness as kh  # noqa: E402

kh.init_progress(hf_progress_repo="cstr/crispasr-kaggle-progress")
step = kh.step
step("script.start", issue=333)

CONVERTER = REPO / "models" / "convert-madlad-to-gguf.py"
if not CONVERTER.is_file():
    step("fatal.converter-missing", path=str(CONVERTER))
    raise SystemExit("converter missing — the clone failed; refusing to continue")

# ── HF auth (gotcha #26b: listing the dataset does nothing on its own) ─────────
TOKEN = kh.resolve_hf_token("HF_TOKEN")
if not TOKEN:
    step("fatal.no-token")
    raise SystemExit("no HF token — uploads would fail after ~30 min of work")
step("hf_token.resolved")

step("install-deps.begin")
subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet",
                       "huggingface_hub", "hf_transfer"])
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from huggingface_hub import HfApi, snapshot_download  # noqa: E402

step("install-deps.done")

# ── binaries: madlad landed in v0.8.16, so the v0.8.6 tarball other kernels
# use is TOO OLD and would fail `--backend madlad` at validation time. ─────────
RELEASE = "v0.8.25"
TARBALL = "crispasr-linux-x86_64.tar.gz"
BIN = Path("/kaggle/temp/bin")
BIN.mkdir(parents=True, exist_ok=True)
CRISPASR = BIN / "crispasr"
QUANT = BIN / "crispasr-quantize"
step("binary-download.begin", release=RELEASE)
subprocess.check_call(
    f"wget -q https://github.com/CrispStrobe/CrispASR/releases/download/{RELEASE}/{TARBALL} "
    f"-O /kaggle/temp/c.tar.gz && tar -xzf /kaggle/temp/c.tar.gz -C {BIN} --strip-components=1",
    shell=True)
for b in (CRISPASR, QUANT):
    if not b.is_file():
        step("fatal.binary-missing", which=b.name)
        raise SystemExit(f"{b.name} not in {TARBALL}")
    b.chmod(0o755)
step("binary-download.done")

# ── staging: /kaggle/working is ~20 GB and we peak around 21 GB. /tmp is the
# ~70 GB ephemeral layer (gotcha #18 — and do NOT trust disk_usage's number). ──
MODELS = Path("/tmp/madlad")
MODELS.mkdir(parents=True, exist_ok=True)


def free_gb(p=None):
    try:
        return round(shutil.disk_usage(str(p or MODELS)).free / 1e9, 1)
    except Exception:
        return -1.0


def gb(p):
    return round(Path(p).stat().st_size / 1e9, 2)


SRC_REPO = "google/madlad400-3b-mt"
DST_REPO = "cstr/madlad400-3b-mt-GGUF"
F16 = MODELS / "madlad400-3b-mt-f16.gguf"
Q8 = MODELS / "madlad400-3b-mt-q8_0.gguf"

# ── validation: a translation, not a load ─────────────────────────────────────
# Each case is (source text, -sl, -tl, [substrings any ONE of which must appear]).
# Kept deliberately loose: greedy decode from a quantized 3B will not be
# word-identical run to run, and the point is "did it translate", not BLEU.
CASES = [
    ("Hello world, how are you today?", "en", "de", ["hallo", "welt", "wie geht"]),
    ("Machine learning is changing the world.", "en", "fr", ["apprentissage", "monde", "machine"]),
    ("Bonjour le monde!", "fr", "en", ["hello", "world", "good"]),
]


def _norm(s):
    return re.sub(r"[^a-z0-9äöüàéèç ]", " ", (s or "").lower())


def validate(model_path, label):
    """Translate three ways. Every case must produce output AND look translated."""
    step(f"{label}.validate.begin", gb=gb(model_path))
    results = []
    for text, sl, tl, wants in CASES:
        with kh.build_heartbeat(f"{label}.translate.{sl}-{tl}", 30):
            p = subprocess.run(
                [str(CRISPASR), "--backend", "madlad", "-m", str(model_path),
                 "--text", text, "-sl", sl, "-tl", tl],
                capture_output=True, text=True, timeout=1800)
        out = _norm(p.stdout)
        # An empty transcript or a non-zero exit is a FAIL, never a pass with a
        # shrug — a crash that exits fast must not read as success (gotcha #24).
        ok = p.returncode == 0 and bool(out.strip())
        hit = any(w in out for w in wants)
        # It must not simply echo the source back untranslated.
        echoed = _norm(text).strip() in out
        results.append({"pair": f"{sl}->{tl}", "exit": p.returncode, "ok": ok,
                        "matched": hit, "echoed": echoed, "out": p.stdout.strip()[:200],
                        "err": p.stderr.strip()[-200:] if p.returncode else ""})
        step(f"{label}.translate.{sl}-{tl}", exit=p.returncode, matched=hit,
             echoed=echoed, out=p.stdout.strip()[:120])
    passed = all(r["ok"] and r["matched"] and not r["echoed"] for r in results)
    step(f"{label}.validate.done", passed=passed)
    return passed, results


def upload(path, msg):
    step("upload.begin", file=Path(path).name, gb=gb(path), repo=DST_REPO)
    with kh.build_heartbeat(f"upload.{Path(path).name}", 30):
        HfApi().upload_file(path_or_fileobj=str(path), path_in_repo=Path(path).name,
                            repo_id=DST_REPO, repo_type="model", token=TOKEN,
                            commit_message=msg)
    step("upload.done", file=Path(path).name)


summary = {"release": RELEASE, "source": SRC_REPO, "target": DST_REPO, "artifacts": {}}


def record(name, **kw):
    summary["artifacts"][name] = kw
    (WORK / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))


# ── 1. source ─────────────────────────────────────────────────────────────────
step("source-download.begin", repo=SRC_REPO, free_gb=free_gb("/tmp"))
SRC = Path("/tmp/madlad-src")
with kh.build_heartbeat("source-download", 30):
    snapshot_download(repo_id=SRC_REPO, local_dir=str(SRC), token=TOKEN,
                      allow_patterns=["model.safetensors", "config.json",
                                      "spiece.model", "tokenizer*.json",
                                      "special_tokens_map.json", "added_tokens.json"])
step("source-download.done", free_gb=free_gb("/tmp"),
     src_gb=round(sum(f.stat().st_size for f in SRC.rglob("*") if f.is_file()) / 1e9, 2))

# ── 2. convert → F16 ──────────────────────────────────────────────────────────
step("convert.begin", free_gb=free_gb("/tmp"))
with kh.build_heartbeat("convert-f16", 30):
    p = subprocess.run([sys.executable, str(CONVERTER), "--input", str(SRC),
                        "--output", str(F16)], capture_output=True, text=True, timeout=7200)
if p.returncode != 0 or not F16.is_file():
    step("fatal.convert-failed", exit=p.returncode, tail=p.stdout[-600:], err=p.stderr[-600:])
    raise SystemExit("F16 conversion failed")
step("convert.done", f16_gb=gb(F16), free_gb=free_gb("/tmp"))

ok_f16, res_f16 = validate(F16, "f16")
record("f16", size_gb=gb(F16), validated=ok_f16, cases=res_f16)
if not ok_f16:
    step("fatal.f16-invalid")
    raise SystemExit("F16 failed validation — refusing to upload a broken artifact")

upload(F16, "add F16 (issue #333: the README listed it but the repo never had it)")
record("f16", size_gb=gb(F16), validated=True, uploaded=True, cases=res_f16)

# The source is 11.76 GB and is not needed again — drop it before the quant so
# the peak stays well inside the ~70 GB ephemeral layer.
shutil.rmtree(SRC, ignore_errors=True)
step("source.deleted", free_gb=free_gb("/tmp"))

# ── 3. quantize → Q8_0 ────────────────────────────────────────────────────────
# No madlad/t5 rule in examples/crispasr-quantize/main.cpp, so this takes the
# generic path — the same one that produced the published Q4_K. Validation
# below is what proves that was the right call for Q8_0 too.
step("quantize.begin", free_gb=free_gb("/tmp"))
with kh.build_heartbeat("quantize-q8_0", 30):
    p = subprocess.run([str(QUANT), str(F16), str(Q8), "q8_0"],
                       capture_output=True, text=True, timeout=7200)
if p.returncode != 0 or not Q8.is_file():
    step("fatal.quantize-failed", exit=p.returncode, tail=p.stdout[-600:], err=p.stderr[-600:])
    raise SystemExit("Q8_0 quantization failed (F16 is already on HF, so nothing is lost)")
step("quantize.done", q8_gb=gb(Q8), free_gb=free_gb("/tmp"))

ok_q8, res_q8 = validate(Q8, "q8_0")
record("q8_0", size_gb=gb(Q8), validated=ok_q8, cases=res_q8)
if not ok_q8:
    step("fatal.q8-invalid")
    raise SystemExit("Q8_0 failed validation — refusing to upload a broken artifact")

upload(Q8, "add Q8_0 (issue #333)")
record("q8_0", size_gb=gb(Q8), validated=True, uploaded=True, cases=res_q8)

step("script.done", artifacts=list(summary["artifacts"]))
print(json.dumps(summary, indent=2, ensure_ascii=False))

# %% [markdown]
# ## What this does NOT do, and why
#
# **No `-ref.gguf`.** Baking one is not a step this kernel can just add: the
# diff harness has no madlad arm at all — there is no
# `tools/reference_backends/madlad.py`, madlad is not in `REGISTERED_BACKENDS`
# in `tools/dump_reference.py`, and `examples/cli/crispasr_diff_main.cpp` has no
# `else if (backend_name == "madlad")`. Writing that dumper is the actual work;
# running it is the easy part. Two things to know before someone does:
#
# - the source is **fp32 and 11.76 GB** against Kaggle's ~13 GB RAM, so the
#   dumper has to load lazily (`safe_open(..., framework="pt")`, one layer at a
#   time) — a plain `T5ForConditionalGeneration.from_pretrained` will OOM;
# - T5 is encoder-decoder, so the stages worth dumping are the encoder stack,
#   the decoder self-attention, and the **cross-attention** — that last one is
#   where an encoder-decoder port usually goes wrong and is exactly what a
#   per-stage diff would catch.
#
# **No Q4_K rebuild.** `madlad400-3b-mt-q4_k.gguf` is already published and in
# the model registry, so `-m auto` has always worked; #333 is only about the two
# files the README promised and the repo never had.
