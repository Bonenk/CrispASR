#!/usr/bin/env python3
"""Consume successful, source-pinned design/build runs and validate on hosted CPU."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

from qwen3_voice_pack_validate import sha

REPO = "CrispStrobe/CrispASR"
OUT = Path(os.environ["HEAVY_OUT"])
SCRATCH = Path(os.environ["HEAVY_SCRATCH"])
ROOT = Path(__file__).resolve().parents[2]


def gh_json(path):
    return json.loads(subprocess.check_output(["gh", "api", path], text=True))


def fetch_artifact(run_id, source, artifact, destination):
    run = gh_json(f"repos/{REPO}/actions/runs/{run_id}")
    assert run["head_sha"] == source and run["status"] == "completed" and run["conclusion"] == "success", run_id
    assert run["event"] == "workflow_dispatch", run_id
    (OUT / f"input-run-{run_id}.json").write_text(json.dumps(run, indent=2) + "\n")
    destination.mkdir(parents=True, exist_ok=True)
    subprocess.run(["gh", "run", "download", str(run_id), "--repo", REPO, "--name", artifact,
                    "--dir", str(destination)], check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-run", type=int, required=True)
    parser.add_argument("--bundle-run", type=int, required=True)
    parser.add_argument("--candidate-source", required=True)
    parser.add_argument("--bundle-source", required=True)
    args = parser.parse_args()
    assert len(args.candidate_source) == 40 and len(args.bundle_source) == 40
    OUT.mkdir(parents=True, exist_ok=True)
    candidate = SCRATCH / "candidate"
    native = SCRATCH / "native-artifact"
    fetch_artifact(args.candidate_run, args.candidate_source, f"heavy-{args.candidate_run}", candidate)
    fetch_artifact(args.bundle_run, args.bundle_source, f"qwen3-voice-pack-runtime-{args.bundle_source}", native)
    archive = native / "qwen3-voice-pack-cuda-speech-validation.tar.gz"
    receipt = json.loads((native / "validation.json").read_text())
    assert receipt["source"] == args.bundle_source and receipt["build_complete"] and not receipt["gpu_executed"]
    assert sha(archive) == receipt["archive"]["sha256"] and archive.stat().st_size == receipt["archive"]["bytes"]
    with tarfile.open(archive) as tar:
        tar.extractall(SCRATCH, filter="data")
    bundle = SCRATCH / "bundle"
    provenance = json.loads((bundle / "provenance.json").read_text())
    assert provenance["source"] == args.bundle_source and provenance["build_complete"] and not provenance["gpu_executed"]
    for name, expected in provenance["files"].items():
        assert sha(bundle / name) == expected["sha256"] and (bundle / name).stat().st_size == expected["bytes"], name
    assert not list(bundle.glob("libcuda.so*"))
    # Establish the exact recipe against the producer commit, independently of
    # the candidate's own manifest. Only a successful source-pinned run is used.
    subprocess.run(["git", "fetch", "--depth=1", "origin", args.candidate_source], cwd=ROOT, check=True)
    recipe = subprocess.check_output(["git", "show", f"{args.candidate_source}:tools/ci-heavy/qwen3_default_voice_bake.py"], cwd=ROOT)
    assert (candidate / "recipe.py").read_bytes() == recipe
    recipe_license = subprocess.check_output(["git", "show", f"{args.candidate_source}:LICENSE"], cwd=ROOT)
    assert (candidate / "LICENSE-RECIPE.txt").read_bytes() == recipe_license
    origin = json.loads((candidate / "provenance.json").read_text())
    assert origin["producer_commit"] == args.candidate_source and origin["producer_run"] == str(args.candidate_run)
    shutil.copytree(candidate, OUT / "candidate")
    (OUT / "runtime-provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    command = [sys.executable, ROOT / "tools/ci-heavy/qwen3_designed_voice_validate.py", "--device", "cpu",
               "--candidate", candidate, "--bundle", bundle, "--scratch", SCRATCH / "cpu-models", "--out", OUT / "cpu"]
    subprocess.run(list(map(str, command)), check=True)
    result = json.loads((OUT / "cpu/validation.json").read_text())
    assert result["complete"] and result["speech_pass"] and result["licensing_evidence_pass"]
    assert result["listening_review"] == "pending" and result["resemblance_review"] == "pending"
    (OUT / "summary.md").write_text(
        "Designed voice licensing evidence and native CPU speech checks passed.\n\n"
        "Real CUDA and listening/resemblance review remain pending. Upstream training-data\n"
        "rights have not been independently verified; no universal legal clearance is claimed.\n")


if __name__ == "__main__":
    main()
