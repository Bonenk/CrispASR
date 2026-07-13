# CrispASR — MOSS-TTS-Local-Transformer-v1.5 (4B) Kaggle validation (#249, P5)
#
# The ONLY acceptance gate for the 4B port (HARD RULE #3): the decoded ASR
# round-trip. Runs on a CUDA box (P100/T4); the 4B F16 backbone (~9 GB) fits a
# P100 16 GB GPU, Q4_K (~2.7 GB) is the practical target.
#
# Flow:
#   1. Clone CrispASR @ CRISPASR_REF (default feat/moss-tts-local-4b), build CUDA
#      (crispasr-cli + crispasr-quantize).
#   2. Backbone F16 GGUF: download from cstr/moss-tts-local-v1.5-GGUF if present,
#      else convert from the HF weights. Codec: convert MOSS-Audio-Tokenizer-v2
#      -> decode-only companion GGUF (arch moss-tts-local-codec).
#   3. Quantize backbone -> Q4_K.
#   4. GATING round-trip on Q4_K (and F16 best-effort; F16 OOM on <=16 GB = SKIP):
#      synthesize a short + a long text, verify non-silent + plausible length,
#      ASR each with whisper, proof-of-work that the long clip yields more words
#      than the short (kaggle_usage #24: a round-trip is a lie until you prove
#      the work). Acceptance = the ASR text is recognizable (word overlap with the
#      input) — greedy code-parity is NOT the gate (quantized AR near-tie flips,
#      see the tts-port-parity-via-logit-rank memory).
#   5. On success, upload the codec GGUF + Q4_K backbone to the GGUF repo.
#
# Keep /kaggle/working minimal; stage everything large under /tmp.

import json
import os
import re
import struct
import subprocess
import sys
import time
import traceback
import wave
from datetime import datetime, timezone
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except (AttributeError, ValueError):
    pass

WORK = Path("/kaggle/working")
TMP = Path("/tmp")
REPO = TMP / "CrispASR"
BUILD = REPO / "build"
MODELS = TMP / "moss-local-models"
RESULTS = WORK / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)

CRISPASR_REF = os.environ.get("CRISPASR_REF", "feat/moss-tts-local-4b")
CRISPASR_REPO = os.environ.get("CRISPASR_REPO", "https://github.com/CrispStrobe/CrispASR.git")
HF_MODEL = os.environ.get("MOSS_MODEL", "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5")
HF_CODEC = os.environ.get("MOSS_CODEC", "OpenMOSS-Team/MOSS-Audio-Tokenizer-v2")
GGUF_REPO = os.environ.get("MOSS_GGUF_REPO", "cstr/moss-tts-local-v1.5-GGUF")
DO_UPLOAD = os.environ.get("MOSS_UPLOAD", "1") == "1"

PROGRESS = WORK / "progress.txt"
_T0 = time.time()


def log(msg):
    line = f"[{round(time.time() - _T0, 1)}s] {msg}"
    print(line, flush=True)
    with open(PROGRESS, "a") as f:
        f.write(line + "\n")


def run(cmd, check=True, timeout=None, env=None, cwd=None, capture=True):
    print(f"\n$ {' '.join(str(c) for c in cmd)}", flush=True)
    e = os.environ.copy()
    if env:
        e.update(env)
    kw = dict(env=e, cwd=cwd, timeout=timeout, text=True)
    if capture:
        kw.update(stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    r = subprocess.run(cmd, **kw)
    if capture and r.stdout:
        print(r.stdout[-4000:], flush=True)
    if check and r.returncode != 0:
        raise SystemExit(f"command failed (rc={r.returncode}): {cmd}")
    return r


# ── WAV / ASR helpers ──────────────────────────────────────────────────────
def wav_summary(path: Path) -> dict:
    if not path.exists():
        return {"error": "missing"}
    with wave.open(str(path), "rb") as w:
        n, sr, sw, ch = w.getnframes(), w.getframerate(), w.getsampwidth(), w.getnchannels()
        raw = w.readframes(n)
    if sw != 2:
        return {"error": f"sw={sw}"}
    pcm = struct.unpack(f"<{n * ch}h", raw)
    if ch > 1:
        pcm = pcm[::ch]
    if not pcm:
        return {"duration_s": 0.0, "rms": 0.0, "n_samples": 0, "sr": sr}
    rms = ((sum(int(x) * int(x) for x in pcm) / max(1, len(pcm))) ** 0.5) / 32768.0
    return {"duration_s": round(len(pcm) / sr, 3), "rms": round(rms, 6),
            "n_samples": len(pcm), "sr": sr, "ch": ch}


def asr_roundtrip(cli: Path, wav: Path, timeout=900) -> str:
    if not wav.exists():
        return ""
    cmd = [str(cli), "--backend", "whisper", "-m", "auto", "--auto-download",
           "-f", str(wav), "--no-prints"]
    try:
        r = subprocess.run(cmd, timeout=timeout, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True)
        (RESULTS / f"{wav.stem}.asr.log").write_text(r.stdout)
        lines = [ln.strip() for ln in r.stdout.splitlines() if ln.strip()]
        text = " ".join(ln for ln in lines
                        if not ln.startswith(("[", "whisper", "ggml", "load", "crispasr")))
        return text
    except Exception as ex:  # noqa: BLE001
        return f"<asr-error: {type(ex).__name__}>"


def synth(cli: Path, backbone: str, codec: str, text: str, out_wav: Path, extra_env=None, timeout=2400) -> dict:
    cmd = [str(cli), "--backend", "moss-tts-local", "-m", backbone, "--codec-model", codec,
           "--tts", text, "--tts-output", str(out_wav), "--no-prints"]
    env = os.environ.copy()
    env["CRISPASR_MOSS_TTS_LOCAL_DEBUG"] = "1"  # trace stop logits + frame count
    if extra_env:
        env.update(extra_env)
    t0 = time.time()
    try:
        r = subprocess.run(cmd, timeout=timeout, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True, env=env)
        rc, out = r.returncode, r.stdout
    except subprocess.TimeoutExpired as ex:
        rc, out = -1, f"TIMEOUT {timeout}s\n{ex.stdout or ''}"
    (RESULTS / f"{out_wav.stem}.log").write_text(out)
    frames, stopped = None, None
    m = re.search(r"generated (\d+) frames .*?(runaway|stopped naturally)", out)
    if m:
        frames, stopped = int(m.group(1)), (m.group(2) == "stopped naturally")
    return {"rc": rc, "elapsed_s": round(time.time() - t0, 1),
            "wav": wav_summary(out_wav) if out_wav.exists() else {"error": "no-wav"},
            "frames": frames, "stopped": stopped,
            "err_excerpt": out[-1800:]}


def verdict(res: dict, min_dur: float) -> str:
    if res["rc"] != 0:
        return f"FAIL: rc={res['rc']}"
    w = res["wav"]
    if "error" in w:
        return f"FAIL: wav {w['error']}"
    if w["duration_s"] < min_dur:
        return f"FAIL: too short ({w['duration_s']}s < {min_dur})"
    if w["rms"] < 1e-4:
        return f"FAIL: silent (rms={w['rms']})"
    return "PASS"


def word_overlap(asr_text: str, ref_text: str) -> float:
    def norm(s):
        return set(re.findall(r"[a-z]+", s.lower()))
    ref = norm(ref_text)
    got = norm(asr_text)
    if not ref:
        return 0.0
    return round(len(ref & got) / len(ref), 3)


SHORT_TEXT = "Hello world."
LONG_TEXT = ("The quick brown fox jumps over the lazy dog. "
             "Speech synthesis should stay intelligible over a longer passage, "
             "so this sentence exercises many autoregressive steps and the codec "
             "sliding window well past the first few frames.")


def main():
    summary = {"ts": datetime.now(timezone.utc).isoformat(), "ref": CRISPASR_REF,
               "phases": {}, "roundtrip": {}, "gates": {}}

    # ── 1. clone + build ───────────────────────────────────────────────────
    log(f"clone {CRISPASR_REF}")
    if not REPO.exists():
        run(["git", "clone", "--depth", "1", "--branch", CRISPASR_REF, "--recursive",
             CRISPASR_REPO, str(REPO)])
    sys.path.insert(0, str(REPO / "tools" / "kaggle"))
    import kaggle_harness as kh
    kh.init_progress()
    sha = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    summary["sha"] = sha
    log(f"cloned {sha}")

    run(["nvidia-smi", "-L"], check=False)
    gpu = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True).strip()
    summary["gpu"] = gpu
    log(f"gpu {gpu}")

    kh.install_build_toolchain()
    arch = kh.detect_cuda_arch()
    log(f"cuda_arch {arch}")
    # Keep ccache OFF /kaggle/working (else it bloats the kernel output and the
    # download page-caps past progress.txt/results — kaggle_usage #22). Export it
    # globally so both cmake and the sh_with_progress build inherit it.
    os.environ["CCACHE_DIR"] = "/kaggle/temp/.ccache"
    env = os.environ.copy()
    cmake_args = (["cmake", "-G", "Ninja", "-B", str(BUILD), "-S", str(REPO),
                   "-DCMAKE_BUILD_TYPE=Release"]
                  + list(kh.cache_and_link_flags()) + list(kh.cuda_build_flags(arch)))
    run(cmake_args, env=env, timeout=300)
    jobs = kh.safe_build_jobs(gpu=True)
    with kh.build_heartbeat("moss-tts-local CUDA build"):
        kh.sh_with_progress(
            f"stdbuf -oL -eL cmake --build {BUILD} "
            f"--target crispasr-cli crispasr-quantize -j{jobs}")
    cli = BUILD / "bin" / "crispasr"
    if not cli.exists():
        cands = [c for c in BUILD.rglob("crispasr") if c.is_file() and os.access(c, os.X_OK)]
        cli = cands[0] if cands else cli
    quant = BUILD / "bin" / "crispasr-quantize"
    if not cli.exists() or not quant.exists():
        raise SystemExit(f"binaries missing: cli={cli.exists()} quant={quant.exists()}")
    os.environ["LD_LIBRARY_PATH"] = f"{BUILD / 'src'}:{os.environ.get('LD_LIBRARY_PATH', '')}"
    summary["phases"]["build"] = "ok"
    log("build ok")

    # ── 2. backbone F16 (download-or-convert) + codec (convert) ────────────
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                           "huggingface_hub", "safetensors", "gguf"])
    hf_token = kh.resolve_hf_token()
    from huggingface_hub import snapshot_download, hf_hub_download, HfApi
    f16 = MODELS / "moss-tts-local-v1.5-f16.gguf"
    try:
        got = hf_hub_download(GGUF_REPO, f16.name, local_dir=str(MODELS), token=hf_token)
        f16 = Path(got)
        log(f"downloaded existing backbone F16 {f16} ({f16.stat().st_size/1e9:.2f} GB)")
        need_backbone_src = False
    except Exception:  # noqa: BLE001
        need_backbone_src = True
        log("backbone F16 not hosted — will convert from HF weights")

    log("download codec (MOSS-Audio-Tokenizer-v2) -> /tmp")
    codec_src = snapshot_download(HF_CODEC, cache_dir=str(MODELS / "hf"), token=hf_token,
                                  allow_patterns=["*.safetensors", "*.json"])
    codec_gguf = MODELS / "moss-tts-local-v1.5-codec.gguf"

    conv = REPO / "models" / "convert-moss-tts-local-to-gguf.py"
    if need_backbone_src:
        src = snapshot_download(HF_MODEL, cache_dir=str(MODELS / "hf"), token=hf_token,
                                allow_patterns=["*.safetensors", "*.json", "merges.txt",
                                                "vocab.json", "tokenizer.json", "added_tokens.json"])
        log("convert -> f16 backbone + codec")
        run([sys.executable, str(conv), "--input", src, "--codec", codec_src,
             "--output", str(f16), "--codec-output", str(codec_gguf)], timeout=3600)
    else:
        # Backbone already hosted — convert the codec only. The converter's main()
        # requires --input (backbone dir) even for codec, so pull the (small) config
        # + tokenizer of the backbone and let it re-emit the backbone too, or call
        # write_codec_gguf directly. We call write_codec_gguf directly (cheaper).
        log("convert codec only (backbone F16 reused)")
        codegen = (
            "import sys; sys.path.insert(0, r'%s');"
            "import importlib.util as u;"
            "spec=u.spec_from_file_location('conv', r'%s');"
            "m=u.module_from_spec(spec); spec.loader.exec_module(m);"
            "from pathlib import Path;"
            "m.write_codec_gguf(Path(r'%s'), Path(r'%s'))"
        ) % (str(REPO / "models"), str(conv), codec_src, str(codec_gguf))
        run([sys.executable, "-c", codegen], timeout=3600)

    if not codec_gguf.exists():
        raise SystemExit("codec GGUF not produced")
    summary["phases"]["convert"] = {"f16_gb": round(f16.stat().st_size / 1e9, 2),
                                    "codec_gb": round(codec_gguf.stat().st_size / 1e9, 2)}
    log(f"converted: {summary['phases']['convert']}")

    import shutil
    shutil.rmtree(MODELS / "hf", ignore_errors=True)

    # ── 3. quantize backbone -> Q4_K ───────────────────────────────────────
    q4k = MODELS / "moss-tts-local-v1.5-q4_k.gguf"
    log("quantize -> q4_k")
    run([str(quant), str(f16), str(q4k), "q4_k"], timeout=1800)
    summary["phases"]["quantize"] = {"q4k_gb": round(q4k.stat().st_size / 1e9, 2)}
    log(f"quantized: {summary['phases']['quantize']}")

    # ── 4. Round-trip arms (chunked codec + stop-runaway A/B) ──────────────
    # run1 showed the codec decode is CORRECT (f16-long overlap 1.0) but short/
    # quantized synths RAN AWAY (stop head never fired) -> 916GB codec OOM. This
    # run: (a) the codec now query-chunks (bounded memory — f16-long is the
    # correctness oracle: overlap must stay ~1.0); (b) A/B sampled vs greedy audio
    # to test whether sampled-audio feedback is what prevents the stop head firing.
    GREEDY = {"CRISPASR_MOSS_TTS_LOCAL_GREEDY_AUDIO": "1"}
    arms = [
        ("q4k_samp", str(q4k), None, False),    # q4_k, sampled audio (run1 default)
        ("q4k_greedy", str(q4k), GREEDY, True), # q4_k, greedy audio — the GATE candidate
        ("f16_greedy", str(f16), GREEDY, False),# f16, greedy — codec correctness oracle
    ]
    for tag, backbone, extra_env, gating in arms:
        rs = synth(cli, backbone, str(codec_gguf), SHORT_TEXT, RESULTS / f"{tag}_short.wav", extra_env=extra_env)
        rs["verdict"] = verdict(rs, min_dur=0.2)
        rs["asr"] = asr_roundtrip(cli, RESULTS / f"{tag}_short.wav")
        rs["overlap"] = word_overlap(rs["asr"], SHORT_TEXT)
        log(f"{tag} short: {rs['verdict']} rc={rs['rc']} frames={rs['frames']} stopped={rs['stopped']} "
            f"overlap={rs['overlap']} asr={rs['asr'][:120]!r}")

        rl = synth(cli, backbone, str(codec_gguf), LONG_TEXT, RESULTS / f"{tag}_long.wav", extra_env=extra_env)
        rl["verdict"] = verdict(rl, min_dur=2.0)
        rl["asr"] = asr_roundtrip(cli, RESULTS / f"{tag}_long.wav")
        rl["overlap"] = word_overlap(rl["asr"], LONG_TEXT)
        log(f"{tag} long: {rl['verdict']} rc={rl['rc']} frames={rl['frames']} stopped={rl['stopped']} "
            f"overlap={rl['overlap']} asr={rl['asr'][:200]!r}")

        short_words = len((rs["asr"] or "").split())
        long_words = len((rl["asr"] or "").split())
        pow_ok = (rl["verdict"] == "PASS" and rs["verdict"] == "PASS"
                  and long_words > short_words
                  and rl["wav"].get("duration_s", 0) > rs["wav"].get("duration_s", 0))
        # Recognizable = the synth stopped at a sane length AND ASR matches the text.
        recognizable = (rs["overlap"] >= 0.5 and rl["overlap"] >= 0.4
                        and (rs["stopped"] is not False) and (rl["stopped"] is not False))
        rt = {"short": rs, "long": rl, "short_words": short_words,
              "long_words": long_words, "proof_of_work": pow_ok, "recognizable": recognizable}
        summary["roundtrip"][tag] = rt
        arm_pass = pow_ok and recognizable
        oom = (not gating) and (rs["rc"] != 0) and ("out of memory" in (rs["err_excerpt"] or "").lower())
        summary["gates"][f"roundtrip_{tag}"] = ("SKIP(oom)" if oom else
                                                ("PASS" if arm_pass else ("FAIL" if gating else "WARN")))
        log(f"{tag} gate: {summary['gates'][f'roundtrip_{tag}']} "
            f"(pow={pow_ok} recognizable={recognizable} overlap short={rs['overlap']} long={rl['overlap']})")

    # Acceptance = the q4_k greedy-audio arm passes (that becomes the default if so).
    summary["all_gates_pass"] = summary["gates"].get("roundtrip_q4k_greedy") == "PASS"

    # ── 5. upload GGUFs on success ─────────────────────────────────────────
    if summary["all_gates_pass"] and DO_UPLOAD:
        try:
            api = HfApi()
            api.create_repo(GGUF_REPO, repo_type="model", exist_ok=True, token=hf_token)
            for p in (codec_gguf, q4k):
                api.upload_file(path_or_fileobj=str(p), path_in_repo=p.name, repo_id=GGUF_REPO,
                                repo_type="model", token=hf_token)
                log(f"uploaded {p.name}")
            summary["uploaded"] = [codec_gguf.name, q4k.name]
        except Exception as e:  # noqa: BLE001
            summary["uploaded"] = f"upload error: {e}"
            log(f"upload skipped: {e}")

    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n" + "=" * 60)
    print(json.dumps(summary, indent=2))
    print("=" * 60)
    if not summary["all_gates_pass"]:
        log("ROUND-TRIP GATE FAILED — see results/ logs")
        sys.exit(1)
    log(f"MOSS-TTS-Local-v1.5 (4B) PASSES the ASR round-trip on {gpu} (Q4_K)")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001
        log(f"FATAL: {e}")
        log(traceback.format_exc())
        sys.exit(1)
