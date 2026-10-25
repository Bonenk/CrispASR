#!/usr/bin/env python3
"""Bounded, source-pinned CPU A/B for nightly data2vec and CSM failures.

Run through heavy-cpu.yml with pip="numpy gguf==0.17.1 huggingface_hub".
The two source builds run sequentially on one host. Only pinned regression
models are downloaded; model weights and build trees stay in HEAVY_SCRATCH.
Raw CLI logs, small CTC sidecars, CSM WAVs and a receipt survive in HEAVY_OUT.
This is a diagnostic: the manifest's hard gates are reported without edits.
"""

import hashlib
from itertools import groupby
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import wave


ROOT = Path(__file__).resolve().parents[2]
PRIOR = "8bc09441a1f47d0e3aeac44e98bf90cf79819d81"
CURRENT = "96aeb4269a68a7d1e2208cf04d692419198f554f"
SOURCES = (("prior", PRIOR), ("nightly", CURRENT))
sys.path.insert(0, str(ROOT / "tests/regression"))
import run_one as regression  # noqa: E402


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def snapshot():
    mem = {}
    if Path("/proc/meminfo").exists():
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, _, value = line.partition(":")
            if key in ("MemAvailable", "SwapFree"):
                mem[key] = value.strip()
    return {"time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "loadavg": os.getloadavg(), "memory": mem,
            "disk_root": shutil.disk_usage("/").free,
            "disk_scratch": shutil.disk_usage(os.environ["HEAVY_SCRATCH"]).free}


def require_headroom(state):
    available = int(state["memory"].get("MemAvailable", "0 kB").split()[0]) * 1024
    if (available < 4 * 1024**3 or state["disk_scratch"] < 6 * 1024**3 or
            state["disk_root"] < 2 * 1024**3 or state["loadavg"][0] > max(2, 2 * os.cpu_count())):
        raise RuntimeError(f"host lacks build headroom: {state}")


def command(argv, cwd, out, name, timeout=1800):
    out.mkdir(parents=True, exist_ok=True)
    if Path(str(argv[0])).name in ("crispasr", "crispasr-diff"):
        resources = snapshot()
        (out / f"{name}.resources.json").write_text(json.dumps(resources, indent=2) + "\n")
        require_headroom(resources)
    (out / f"{name}.argv.json").write_text(json.dumps([str(x) for x in argv]) + "\n")
    with (out / f"{name}.stdout.log").open("wb") as stdout, (out / f"{name}.stderr.log").open("wb") as stderr:
        try:
            proc = subprocess.run(list(map(str, argv)), cwd=cwd, stdout=stdout, stderr=stderr, timeout=timeout)
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            rc = 124
    print(name, "rc=", rc, flush=True)
    return rc


def required(argv, cwd, out, name, timeout=1800):
    rc = command(argv, cwd, out, name, timeout)
    if rc:
        raise RuntimeError(f"{name} failed (rc={rc}); see retained logs")


def selected(manifest):
    d = next(e for e in manifest["backends"] if e["name"] == "data2vec-base")
    c = next(e for e in manifest["tts_backends"] if e["name"] == "csm-1b")
    a = next(e for e in manifest["backends"] if e["name"] == c["roundtrip_asr_backend"])
    return d, c, a


def wav_info(path):
    if not path.exists():
        return {"exists": False}
    try:
        with wave.open(str(path), "rb") as f:
            return {"exists": True, "sha256": sha256(path), "bytes": path.stat().st_size,
                    "channels": f.getnchannels(), "rate": f.getframerate(),
                    "frames": f.getnframes(), "duration_s": f.getnframes() / f.getframerate()}
    except wave.Error as exc:
        return {"exists": True, "sha256": sha256(path), "bytes": path.stat().st_size,
                "wav_error": str(exc)}


def data2vec(case, cli, diff, d, model, f16, ref, vocab_path, sample, receipt):
    result = {}
    rc = command([cli, "-m", model, "-f", sample], case, case, "data2vec-cli", 600)
    stdout = (case / "data2vec-cli.stdout.log").read_text(errors="replace")
    actual = next((x.strip() for x in reversed(stdout.splitlines()) if x.strip()), "")
    ok, messages = regression.transcript_gate(d, actual)
    result.update(cli_rc=rc, transcript=actual, transcript_gate=bool(rc == 0 and ok), gate_messages=messages,
                  human_wer=regression.compute_transcript_metrics(d["human_reference_transcript"], actual)[1])
    # Same CLI with a sidecar; keep the exact nightly invocation above as the gate.
    prefix = case / "data2vec-logits"
    side_rc = command([cli, "-m", model, "-f", sample, "--return-logits", "-of", prefix],
                      case, case, "data2vec-logits-cli", 600)
    sidecar = Path(str(prefix) + ".ctc-logits.json")
    result["sidecar"] = {"rc": side_rc, "exists": sidecar.exists(),
                         "sha256": sha256(sidecar) if sidecar.exists() else None}
    if side_rc == 0 and sidecar.exists():
        import gguf
        import numpy as np
        payload = json.loads(sidecar.read_text())["ctc_logits"]
        native = np.asarray(payload["data"], dtype=np.float32).reshape(payload["n_frames"], payload["n_vocab"])
        reference = next(t.data for t in gguf.GGUFReader(str(ref)).tensors if t.name == d["transcript_reference"]["stage"])
        vocab = json.loads(vocab_path.read_text())
        inv = {i: token for token, i in vocab.items()}
        blank = vocab[d["transcript_reference"]["blank_token"]]
        ids = np.argmax(native, axis=1).tolist()
        decoded = "".join(inv[i] for i, _ in groupby(ids) if i != blank).replace("|", " ").strip()
        result["sidecar"]["native_ctc_decode"] = decoded
        if reference.shape == native.shape:
            ref_ids = np.argmax(reference, axis=1)
            a_starts = [t for t in range(len(ref_ids)) if ref_ids[t] == vocab["A"] and
                        (t == 0 or ref_ids[t - 1] != ref_ids[t])]
            if len(a_starts) >= 2:
                center = a_starts[1]  # second A is the inserted word after AND SO
                probes = []
                for t in range(max(0, center - 5), min(len(ref_ids), center + 6)):
                    order = np.argsort(native[t])[-2:][::-1]
                    a_value = float(native[t, vocab["A"]])
                    probes.append({"frame": t, "native_top2": [inv[int(x)] for x in order],
                                   "native_a_rank": int(np.sum(native[t] > a_value) + 1),
                                   "native_a_gap_to_top": float(native[t, order[0]] - a_value),
                                   "reference_top": inv[int(ref_ids[t])]})
                result["sidecar"]["inserted_a_probe"] = probes
        else:
            result["sidecar"]["shape_mismatch"] = [list(native.shape), list(reference.shape)]
    # The F16 stage uses the same model/reference/threshold as the nightly.
    diff_rc = command([diff, d["backend_id"], f16, ref, sample], case, case, "data2vec-diff", 600)
    raw = (case / "data2vec-diff.stdout.log").read_text(errors="replace")
    stages = regression.parse_diff_stdout(raw)
    passed, failed, missing, extras = regression.evaluate_stage_thresholds(
        stages, d["diff_thresholds"], **regression.stage_gate_kwargs(d))
    result["stage"] = {"rc": diff_rc, "values": stages, "passed": passed,
                       "failed": failed, "missing": missing, "extras": extras,
                       "gate": bool(diff_rc == 0 and stages and not failed and not missing)}
    receipt["data2vec"] = result


def csm(case, cli, c, tts_model, asr_model, receipt, label="csm"):
    output = case / f"{label}.wav"
    seed = str(c.get("tts_seed", 1234))
    synth = [cli, "-m", tts_model, "--tts", c["tts_phrase"], "--tts-output", output,
             "--seed", seed, "--no-prints"]
    if c.get("voice_preset"):
        synth += ["--voice", c["voice_preset"]]
    synth += c.get("tts_extra_args", [])
    synth_rc = command(synth, case, case, f"{label}-synth", 300)
    info = wav_info(output)
    result = {"seed": seed, "synth_rc": synth_rc, "wav": info,
              "phrase": c["tts_phrase"], "wer_max": c["wer_max"]}
    if synth_rc == 0 and info["exists"] and info["bytes"] >= 1000 and "wav_error" not in info:
        asr_rc = command([cli, "--backend", "parakeet", "-m", asr_model, "-f", output, "--no-prints"],
                         case, case, f"{label}-asr", 180)
        raw = (case / f"{label}-asr.stdout.log").read_text(errors="replace")
        transcript = regression.strip_ai_disclaimer(
            next((x.strip() for x in reversed(raw.splitlines()) if x.strip()), ""))
        wer = regression.compute_transcript_metrics(c["tts_phrase"], transcript)[1]
        result.update(asr_rc=asr_rc, transcript=transcript, wer=wer,
                      gate=bool(asr_rc == 0 and wer <= float(c["wer_max"])))
    else:
        result["gate"] = False
    receipt[label] = result
    return result


def main():
    out = Path(os.environ["HEAVY_OUT"]).resolve()
    scratch = Path(os.environ["HEAVY_SCRATCH"]).resolve() / "nightly-error-ab"
    out.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(scratch)
    # Matrix builds use -march=native; the runner image alone does not identify
    # the CPU instruction path. Retain the actual host and compiler commands.
    shutil.copy2("/proc/cpuinfo", out / "cpuinfo.txt")
    manifest = json.loads((ROOT / "tests/regression/manifest.json").read_text())
    d, c, a = selected(manifest)
    packages = {name: importlib.metadata.version(name) for name in ("numpy", "gguf", "huggingface_hub")}
    receipt = {"scope": "data2vec-base and csm-1b only, same pinned inputs and same host",
               "source": dict(SOURCES), "runner": {"platform": platform.platform(),
               "processor": platform.processor(), "machine": platform.machine(),
               "cpu_count": os.cpu_count(), "image": os.environ.get("ImageOS"),
               "image_version": os.environ.get("ImageVersion")},
               "packages": packages,
               "pins": {"fixtures": manifest["fixtures"], "data2vec": d,
                        "csm": c, "asr": a}, "resources": {"initial": snapshot()}, "cases": {}}
    def save():
        (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    save()
    try:
        require_headroom(receipt["resources"]["initial"])
        required(["bash", ROOT / "tools/ci-apt.sh", "update"], ROOT, out, "apt-update", 600)
        required(["bash", ROOT / "tools/ci-apt.sh", "install", "-y", "libopenblas-dev",
                  "libespeak-ng-dev", "espeak-ng", "pkg-config"], ROOT, out, "apt-install", 600)
        # Download each immutable artifact once, then reuse it across both builds.
        models = scratch / "models"
        models.mkdir(exist_ok=True)
        get = lambda spec: regression.hf_download(spec["repo"], spec["file"], spec["revision"], models)
        q4 = get(d["gguf"])
        f16_spec = {**d["gguf"], **d["diff_gguf"]}
        f16 = get(f16_spec)
        ref = regression.hf_download(manifest["fixtures"]["repo"], d["fixture_ref_path"],
                                     manifest["fixtures"]["revision"], models)
        vocab = get(d["transcript_reference"]["vocab"])
        tts = get(c["gguf"])
        asr = get(a["gguf"])
        sample = ROOT / d["sample"]
        independent = regression.verify_transcript_reference(d, ref, sample, models)
        receipt["independent_transcript"] = independent
        receipt["input_sha256"] = {k: sha256(v) for k, v in
                                   {"q4": q4, "f16": f16, "ref": ref, "vocab": vocab,
                                    "tts": tts, "asr": asr, "sample": sample}.items()}
        save()
        for tag, source in SOURCES:
            case = out / tag
            case.mkdir(exist_ok=True)
            worktree = scratch / f"source-{tag}"
            required(["git", "fetch", "--depth=1", "origin", source], ROOT, case, "fetch", 300)
            required(["git", "worktree", "add", "--detach", worktree, source], ROOT, case, "worktree", 300)
            required(["git", "submodule", "update", "--init", "--recursive"], worktree, case,
                     "submodules", 600)
            required(["git", "submodule", "status", "--recursive"], worktree, case,
                     "submodule-status", 60)
            actual_source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=worktree, text=True).strip()
            if actual_source != source:
                raise RuntimeError(f"{tag} source mismatch: {actual_source}")
            other = json.loads((worktree / "tests/regression/manifest.json").read_text())
            od, oc, oa = selected(other)
            if (other["fixtures"], od, oc, oa) != (manifest["fixtures"], d, c, a):
                raise RuntimeError(f"{tag} regression pins differ from diagnostic manifest")
            case_result = {"source": actual_source, "resources_before_build": snapshot(),
                           "build_flags": ["-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release",
                                           "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
                                           "-DCRISPASR_BUILD_TESTS=OFF", "-DCRISPASR_BUILD_EXAMPLES=ON",
                                           "-DCRISPASR_BUILD_SERVER=OFF"]}
            receipt["cases"][tag] = case_result
            save()
            require_headroom(case_result["resources_before_build"])
            build = scratch / f"build-{tag}"
            required(["cmake", "-S", worktree, "-B", build, *case_result["build_flags"]],
                     worktree, case, "configure", 600)
            required(["cmake", "--build", build, "--target", "crispasr-cli", "crispasr-diff", "-j4"],
                     worktree, case, "build", 1800)
            shutil.copy2(build / "CMakeCache.txt", case / "CMakeCache.txt")
            shutil.copy2(build / "compile_commands.json", case / "compile_commands.json")
            cli, diff = build / "bin/crispasr", build / "bin/crispasr-diff"
            if not cli.is_file() or not diff.is_file():
                raise RuntimeError(f"{tag}: built executables missing")
            case_result["binary_sha256"] = {"cli": sha256(cli), "diff": sha256(diff)}
            required(["ldd", cli], worktree, case, "cli-libraries", 60)
            data2vec(case, cli, diff, d, q4, f16, ref, vocab, sample, case_result)
            csm(case, cli, c, tts, asr, case_result)
            case_result["resources_after"] = snapshot()
            save()
        # One identical-seed CSM repeat only when the two first outputs disagree.
        prior = receipt["cases"]["prior"]["csm"]
        nightly = receipt["cases"]["nightly"]["csm"]
        if (prior.get("transcript"), prior["wav"].get("sha256")) != (
                nightly.get("transcript"), nightly["wav"].get("sha256")):
            for tag, _ in SOURCES:
                case = out / tag
                cli = scratch / f"build-{tag}" / "bin/crispasr"
                csm(case, cli, c, tts, asr, receipt["cases"][tag], "csm-repeat")
                save()
        receipt["complete"] = True
    except Exception as exc:
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        save()
    # The baseline source is a control; the nightly source must retain both gates.
    now = receipt["cases"]["nightly"]
    passed = (now["data2vec"]["transcript_gate"] and now["data2vec"]["stage"]["gate"] and
              now["data2vec"]["sidecar"]["exists"] and now["csm"]["gate"])
    (out / "summary.md").write_text(
        "# Nightly targeted CPU A/B\n\n" +
        "| Source | data2vec transcript | data2vec F16 stage | CSM WER |\n|---|---|---|---|\n" +
        "".join(f"| {tag} `{sha[:9]}` | {receipt['cases'][tag]['data2vec']['transcript_gate']} | "
                f"{receipt['cases'][tag]['data2vec']['stage']['gate']} | "
                f"{receipt['cases'][tag]['csm'].get('wer', 'no ASR output')} |\n"
                for tag, sha in SOURCES) +
        "\nFull raw logs, generated WAVs, CTC logits and pins are in the artifact.\n")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
