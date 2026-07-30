#!/usr/bin/env python3
"""Speaker-count estimator A/B on VoxConverse dev — BIC+silhouette vs NME-SC.

Why on Kaggle: the sweep is 200+ diarization runs over 20 h of audio. The dev
box it was written on sat between load 13 and 197 all session and killed the
run twice, so the numbers that matter never got produced.

What it measures: speaker-COUNT accuracy first, DER second. The estimator was
found wrong on 4 of 8 files while DER still averaged 7.32%, because merging two
speakers only costs the frames of the one absorbed — so DER alone cannot see
this failure. See tools/diarize_eval.py.

Runs the TUNE split only. Holdout is deliberately not computed: not calculating
it is a stronger guarantee than calculating it and promising not to look.

Kernel notes (tools/../kaggle_usage.md):
  * GPU is enabled ONLY because Kaggle CPU workers get no internet (#3) and a
    script kernel cannot import bundled siblings (#26) — the repo has to be
    cloned. The work itself is CPU-bound.
  * Repo goes to /kaggle/temp, not /kaggle/working, so `kernels output` is not
    page-capped past the artifacts we need (#22).
  * Long phases are wrapped in kh.build_heartbeat so Kaggle does not idle-kill
    a silent build or sweep (#26).
"""

import json
import os
import subprocess
import sys
from pathlib import Path

WORK = Path("/kaggle/working")
TEMP = Path("/kaggle/temp")
TEMP.mkdir(parents=True, exist_ok=True)

CRISPASR_URL = "https://github.com/CrispStrobe/CrispASR.git"
REPO = TEMP / "CrispASR"

# ── clone + harness (must come from the clone, not a bundled sibling) ────────
if not REPO.exists():
    # --recursive: the build needs the bundled ggml submodule, and a plain
    # --depth 1 clone leaves cmake dying on a missing ggml/CMakeLists.txt.
    subprocess.check_call(["git", "clone", "--depth", "1", "--recursive", CRISPASR_URL, str(REPO)])
sys.path.insert(0, str(REPO / "tools" / "kaggle"))
import kaggle_harness as kh  # noqa: E402

kh.init_progress()
kh.step("clone", status="ok", repo=str(REPO))

token = kh.resolve_hf_token()
kh.step("hf_token", status="ok" if token else "MISSING")

# ── build (CPU only; the diarizer never uses the GPU) ────────────────────────
with kh.build_heartbeat("build", 30):
    kh.install_build_toolchain()
    build = REPO / "build"
    # cache_and_link_flags() folds in ccache/mold AND -DCRISPASR_NO_C2PA_NATIVE
    # (the c2pa-audio submodule is irrelevant here and breaks generate).
    subprocess.check_call(
        ["cmake", "-S", str(REPO), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release",
         "-DCRISPASR_BUILD_TESTS=OFF", "-DGGML_CUDA=OFF"] + kh.cache_and_link_flags(),
    )
    kh.sh_with_progress(
        f"stdbuf -oL -eL cmake --build {build} --target crispasr-cli -j{kh.safe_build_jobs(gpu=False)}"
    )
cli = build / "bin" / "crispasr"
if not cli.exists():
    raise SystemExit(f"build produced no {cli}")
kh.step("build", status="ok")

# ── corpus: HF parquet -> wav/ + ref.json ────────────────────────────────────
from huggingface_hub import hf_hub_download  # noqa: E402

CORPUS = TEMP / "corpus"
with kh.build_heartbeat("corpus", 30):
    for i in range(5):
        p = hf_hub_download(
            repo_id="diarizers-community/voxconverse",
            filename=f"data/dev-0000{i}-of-00005.parquet",
            repo_type="dataset",
            token=token,
            cache_dir=str(TEMP / "hf"),
        )
        subprocess.check_call([sys.executable, str(REPO / "tools" / "voxconverse_extract.py"),
                               "--parquet", p, "--out", str(CORPUS)])
ref = json.load(open(CORPUS / "ref.json"))
kh.step("corpus", status="ok", files=len(ref))

# ── models ──────────────────────────────────────────────────────────────────
asr = hf_hub_download(repo_id="ggerganov/whisper.cpp", filename="ggml-tiny.bin",
                      token=token, cache_dir=str(TEMP / "hf"))
emb = hf_hub_download(repo_id="cstr/wespeaker-resnet34-lm-GGUF",
                      filename="wespeaker-resnet34-lm.gguf",
                      token=token, cache_dir=str(TEMP / "hf"))
kh.step("models", status="ok")

# ── the A/B ─────────────────────────────────────────────────────────────────
CMD = (
    f"{cli} -m {asr} -f {{wav}} -t {os.cpu_count() or 4} --diarize "
    f"--diarize-method foxnose --diarize-embedder {emb} --diarize-max-speakers 20 "
    f"-oj -of {{out}}"
)
BASE = [
    sys.executable, str(REPO / "tools" / "diarize_eval.py"),
    "--wav-dir", str(CORPUS / "wav"), "--ref", str(CORPUS / "ref.json"),
    "--workdir", str(TEMP / "evalwork"), "--jobs", "2",
    "--max-speakers", "20", "--split", "tune", "--cmd", CMD,
]

summary = {}
for arm, env_extra in (("bic", {}), ("nme-sc", {"CRISPASR_DIARIZE_COUNT": "nme-sc"})):
    env = dict(os.environ)
    env.update(env_extra)
    out_json = WORK / f"eval_{arm}.json"
    with kh.build_heartbeat(f"eval:{arm}", 30):
        r = subprocess.run(BASE + ["--json-out", str(out_json)], env=env,
                           capture_output=True, text=True)
    (WORK / f"eval_{arm}.txt").write_text(r.stdout + "\n--- stderr tail ---\n" + r.stderr[-4000:])
    tune = [l for l in r.stdout.splitlines() if l.startswith("tune ")]
    summary[arm] = tune[0] if tune else f"NO RESULT (rc={r.returncode})"
    kh.step(f"eval:{arm}", status="ok" if tune else "FAILED", line=summary[arm])
    print(f"[{arm}] {summary[arm]}", flush=True)

(WORK / "summary.json").write_text(json.dumps(summary, indent=1))
print("\n=== TUNE split, speaker-count accuracy is the metric ===")
for arm, line in summary.items():
    print(f"{arm:8} {line}")
kh.step("done", status="ok")
