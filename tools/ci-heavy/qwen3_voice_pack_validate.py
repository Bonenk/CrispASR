#!/usr/bin/env python3
"""#555: real CLI/public API error handling and compatible-voice speech checks.

Use --device cpu on GitHub, --device cuda only on actual GPU hardware.
Synthetic malformed packs exercise loading only, never speech acceptance.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
import wave

import numpy as np
from huggingface_hub import hf_hub_download
from qwen3_voice_pack_fixtures import write_fixtures

TEXT = "I found the file you asked about and opened it."
PINS = {
    "small": ("cstr/qwen3-tts-0.6b-base-GGUF", "94072f5c394b4dc4d98e97cc874800dde6ecacda", "qwen3-tts-12hz-0.6b-base-q8_0.gguf"),
    "large": ("cstr/qwen3-tts-1.7b-base-GGUF", "1d9fa40e85c64e699f3f9fea5aa7fc69a7b2d4ff", "qwen3-tts-12hz-1.7b-base-f16.gguf"),
    "codec": ("cstr/qwen3-tts-tokenizer-12hz-GGUF", "344e279551177654762e5c7d39e217834325f222", "qwen3-tts-tokenizer-12hz.gguf"),
    "voice": ("cstr/qwen3-tts-voices-GGUF", "6ee1102f782cab1f17842983044ac51cf57884d8", "qwen3-tts-voice-default.gguf"),
    "judge": ("ggerganov/whisper.cpp", "5359861c739e955e79d9a303bcbc70fb988958b1", "ggml-base.en.bin"),
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def wer(expected, heard):
    a, b = [re.findall(r"[a-z]+", text.lower()) for text in (expected, heard)]
    row = list(range(len(b) + 1))
    for i, token in enumerate(a, 1):
        previous, row = row, [i]
        for j, other in enumerate(b, 1):
            row.append(min(row[-1] + 1, previous[j] + 1, previous[j-1] + (token != other)))
    return row[-1] / len(a)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--scratch", required=True, type=Path)
    parser.add_argument("--device", required=True, choices=("cpu", "cuda"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    args.scratch.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, LD_LIBRARY_PATH=str(args.bundle) + ":" + os.environ.get("LD_LIBRARY_PATH", ""),
               GGML_BACKEND_PATH=str(args.bundle), CRISPASR_QWEN3_TTS_MAX_FRAMES="180",
               OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="4")
    receipt = dict(device=args.device, complete=False, speech_pass=False, cases={}, models={})
    def save():
        (args.out / "validation.json").write_text(json.dumps(receipt, indent=2) + "\n")
    save()
    def run(command, label, reject=False, gpu=False):
        print(label, flush=True)
        peak = 0
        with (args.out / (label + ".log")).open("w") as log:
            proc = subprocess.Popen(list(map(str, command)), env=env, stdout=log, stderr=subprocess.STDOUT)
            deadline = time.monotonic() + 1800
            while proc.poll() is None:
                if args.device == "cuda" and gpu:
                    rows = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory",
                                                    "--format=csv,noheader,nounits"], text=True)
                    peak = max([peak] + [int(row.split(",")[1]) for row in rows.splitlines()
                                        if row.strip() and int(row.split(",")[0]) == proc.pid])
                if time.monotonic() > deadline:
                    proc.kill()
                    proc.wait()
                    raise TimeoutError(label)
                time.sleep(.5)
        receipt["cases"][label] = dict(returncode=proc.returncode, peak_gpu_mib=peak)
        save()
        content = (args.out / (label + ".log")).read_text(errors="replace")
        if reject:
            assert proc.returncode > 0 and proc.returncode < 128 and "incompatible voice pack" in content, (label, proc.returncode)
            assert "GGML_ASSERT" not in content and "SIGABRT" not in content
        else:
            assert proc.returncode == 0, (label, proc.returncode)
        if args.device == "cuda" and gpu:
            assert peak >= 512 and "CUDA" in content, (label, peak, "GPU residency/backend evidence missing")
        return content
    try:
        if args.device == "cuda":
            receipt["hardware"] = subprocess.check_output(["nvidia-smi", "--query-gpu=name,compute_cap,driver_version",
                                                          "--format=csv,noheader"], text=True).strip()
            assert receipt["hardware"]
        models = {}
        for name, (repo, revision, filename) in PINS.items():
            path = Path(hf_hub_download(repo, filename, revision=revision, local_dir=args.scratch / "models"))
            models[name] = path
            receipt["models"][name] = dict(repo=repo, revision=revision, filename=filename,
                                          bytes=path.stat().st_size, sha256=sha(path))
            save()
        cli = args.bundle / "crispasr"
        common = ["-t", "4"] + (["-ng"] if args.device == "cpu" else [])
        for name, size in (("small", 1024), ("large", 2048)):
            fixtures = args.scratch / ("fixtures-" + name)
            write_fixtures(fixtures, size)
            log = run([args.bundle / "qwen3-voice-pack-probe", models[name], fixtures, args.device],
                      "loader-" + name, gpu=True)
            assert "QWEN3_VOICE_PACK_PASS checks=11" in log
        # Exact reporter recipe: no --voice, so sibling 0.6B default is discovered.
        run([cli, "--backend", "qwen3-tts", "-m", models["large"], "--tts", "hello world",
             "--codec-model", models["codec"], "--tts-output", args.out / "must-not-exist.wav", *common],
            "reporter-rejection", reject=True, gpu=True)
        assert not (args.out / "must-not-exist.wav").exists()
        run([cli, "--backend", "qwen3-tts", "-m", models["large"], "--voice", models["voice"],
             "--tts", "hello world", "--codec-model", models["codec"], "--i-have-rights",
             "--tts-output", args.out / "explicit-must-not-exist.wav", *common],
            "explicit-rejection", reject=True, gpu=True)
        assert not (args.out / "explicit-must-not-exist.wav").exists()
        # Keep normal file marking; synthetic test speech omits the spoken prefix.
        marking = ["--no-spoken-disclaimer", "--accept-marking-responsibility"]
        reference = args.out / "synthetic-reference.wav"
        run([cli, "--backend", "qwen3-tts", "-m", models["small"], "--tts", TEXT,
             "--codec-model", models["codec"], "--tts-output", reference, *marking, *common],
            "speech-small", gpu=True)
        def judge(path, label, expected):
            with wave.open(str(path)) as wav:
                assert wav.getframerate() == 24000 and wav.getnchannels() == 1 and wav.getsampwidth() == 2
                pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float64) / 32768
            assert len(pcm) > 12000 and np.isfinite(pcm).all() and np.sqrt(np.mean(pcm ** 2)) > .001
            prefix = args.out / (label + "-asr")
            run([cli, "-m", models["judge"], "-f", path, "-ng", "-nt", "-np", "-otxt", "-of", prefix], label + "-judge")
            heard = prefix.with_suffix(".txt").read_text()
            error = wer(expected, heard)
            receipt["cases"][label].update(expected=expected, heard=heard, wer=error, wav_sha256=sha(path))
            save()
            assert error <= .25, (label, error, heard)
        judge(reference, "speech-small", TEXT)
        for label, text in (("speech-large", TEXT), ("hello-large", "Hello world.")):
            wav = args.out / (label + ".wav")
            run([cli, "--backend", "qwen3-tts", "-m", models["large"], "--tts", text,
                 "--codec-model", models["codec"], "--voice", reference, "--ref-text", TEXT,
                 "--i-have-rights", "--tts-output", wav, *marking, *common], label, gpu=True)
            judge(wav, label, text)
        receipt.update(complete=True, speech_pass=True)
    finally:
        save()


if __name__ == "__main__":
    main()
