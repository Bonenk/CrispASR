#!/usr/bin/env python3
"""Validate the CPU-qualified designed defaults on actual CUDA, without builds."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import zipfile

VERSION = "issue555-designed-voices-cuda-v1"
SOURCE = "a58086216a0a1ff9d58eb2a9773a9a83fc5d702a"
RUNTIME_SOURCE = "7411898f8d62add2d8e03ce1f7f26233e17875c3"
FIXTURES = "cstr/crispasr-designed-voice-validation"
REVISION = "34d38a3aeafb43dabebe370efd81829f54bdcf3f"
PINS = {
    "runtime": ("v1/runtime-7411898f8/runtime.tar.gz", "569f84752af82b64cb51a16c56ff199c2661771514d99ba58912d8b98369bace"),
    "candidate": ("v1/source-candidates-3ef249810.zip", "b625988c63b08ce5911513645f9731711b5fa30a18cbc5e16871c4d55e429d8a"),
    "cpu": ("v1/cpu-a580/validation.json", "dcbb2088b7dd98f89a52a8842481fb759c7cee83cccdbd03af02c8879772d899"),
    "approval": ("v1/cpu-a580/approval.json", "546f769477fb7cf7e45ed6a887ccd79407e6b48c21542542bf4c2ba477a89b4e"),
}
OUT = Path("/kaggle/working")
SCRATCH = Path("/kaggle/temp/designed-voices")
SCRATCH.mkdir(parents=True, exist_ok=True)
os.environ.update(TMPDIR=str(SCRATCH), HF_HOME=str(SCRATCH / "hf"), HF_XET_CACHE=str(SCRATCH / "xet"),
                  OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="4")
hardware = subprocess.check_output(["nvidia-smi", "--query-gpu=name,compute_cap,driver_version",
                                   "--format=csv,noheader"], text=True).strip()
assert hardware, "Actual CUDA hardware required"
print(VERSION, "source", SOURCE, "hardware", hardware, flush=True)
repo = SCRATCH / "CrispASR"
for command in (["git", "init", repo], ["git", "-C", repo, "remote", "add", "origin", "https://github.com/CrispStrobe/CrispASR.git"],
                ["git", "-C", repo, "fetch", "--depth=1", "origin", SOURCE], ["git", "-C", repo, "checkout", "FETCH_HEAD"]):
    subprocess.run(list(map(str, command)), check=True)
assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip() == SOURCE
sys.path.insert(0, str(repo / "tools/kaggle"))
import kaggle_harness as kh
kh.init_progress()
kh.provenance(VERSION, repo)
os.environ["HF_TOKEN"] = kh.resolve_hf_token(require=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "numpy<3", "huggingface_hub", "gguf"], check=True)
from huggingface_hub import hf_hub_download


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


inputs = {}
for label in ("approval", "cpu", "candidate", "runtime"):
    name, expected = PINS[label]
    path = Path(hf_hub_download(FIXTURES, name, repo_type="dataset", revision=REVISION))
    assert sha(path) == expected, label
    inputs[label] = path
approval = json.loads(inputs["approval"].read_text())
cpu = json.loads(inputs["cpu"].read_text())
assert approval["run"] == 39942586420 and approval["source"] == SOURCE and approval["device"] == "cpu"
assert approval["all_wer_zero"] and not approval["gpu_executed"]
assert cpu["complete"] and cpu["speech_pass"] and cpu["licensing_evidence_pass"]
assert cpu["curation"]["selected_source_voice"] == "clear" and not cpu["curation"]["encoder_arrays_changed"]
scores = [row["wer"] for row in cpu["cases"].values() if "wer" in row]
assert len(scores) == 13 and all(value == 0 for value in scores)
assert sha(repo / "tools/ci-heavy/qwen3_designed_voice_validate.py") == approval["curation_recipe_sha256"]
with zipfile.ZipFile(inputs["candidate"]) as stream:
    assert all(not Path(name).is_absolute() and ".." not in Path(name).parts for name in stream.namelist())
    stream.extractall(SCRATCH / "source-candidates")
with tarfile.open(inputs["runtime"]) as stream:
    stream.extractall(SCRATCH, filter="data")
bundle = SCRATCH / "bundle"
provenance = json.loads((bundle / "provenance.json").read_text())
assert provenance["source"] == RUNTIME_SOURCE and provenance["build_complete"] and not provenance["gpu_executed"]
for name, expected in provenance["files"].items():
    assert sha(bundle / name) == expected["sha256"] and (bundle / name).stat().st_size == expected["bytes"], name
assert not list(bundle.glob("libcuda.so*")), "Driver/stub must not be bundled"
(OUT / "provenance.json").write_text(json.dumps(dict(version=VERSION, source=SOURCE, runtime_source=RUNTIME_SOURCE,
    hardware=hardware, entry_sha256=sha(__file__), fixture_revision=REVISION, fixture_hashes=PINS,
    cpu_run=approval["run"], native_files=provenance["files"]), indent=2) + "\n")
kh.step("designed.inputs.qualified", source=SOURCE, runtime_source=RUNTIME_SOURCE, cpu_run=approval["run"])
environment = dict(os.environ, LD_LIBRARY_PATH=str(bundle) + ":" + os.environ.get("LD_LIBRARY_PATH", ""))
command = [sys.executable, repo / "tools/ci-heavy/qwen3_designed_voice_validate.py", "--device", "cuda",
           "--bundle", bundle, "--candidate", SCRATCH / "source-candidates/candidate",
           "--scratch", SCRATCH / "models-and-fixtures", "--out", OUT / "cuda"]
with kh.build_heartbeat("designed.cuda.validation"):
    result = subprocess.run(list(map(str, command)), env=environment)
summary = json.loads((OUT / "cuda/validation.json").read_text())
kh.step("designed.cuda.result", returncode=result.returncode, complete=summary["complete"], speech_pass=summary["speech_pass"])
assert result.returncode == 0 and summary["complete"] and summary["speech_pass"]
assert summary["licensing_evidence_pass"]
for size in ("0.6b", "1.7b"):
    assert summary["curation"][size]["sha256"] == approval["default_sha256"][size], size
scores = [row["wer"] for row in summary["cases"].values() if "wer" in row]
assert len(scores) == 13 and all(value == 0 for value in scores), scores
print("DESIGNED_DEFAULT_CUDA_ACCEPTED", VERSION, "all 13 reference/speech WER zero; CPU/GPU asset bytes identical", flush=True)
