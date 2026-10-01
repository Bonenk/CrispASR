#!/usr/bin/env python3
"""#478 CPU RALM prefill proof: states/KV/continuation, torch oracle, TTS/ASR.

Run via heavy-cpu.yml with pip='huggingface_hub numpy gguf torch'.
The oracle implements the upstream MiniCPM no-RoPE causal forward using the
same F16 GGUF weights, in FP32. It isolates batching from checkpoint rounding.
"""
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download

REPO = Path(__file__).resolve().parents[2]
OUT = Path(os.environ.get("HEAVY_OUT", "out"))
SCR = Path(os.environ.get("HEAVY_SCRATCH", "scratch"))
OUT.mkdir(parents=True, exist_ok=True)
SCR.mkdir(parents=True, exist_ok=True)
RESULT = {"sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(), "models": {}}


def run(cmd, name, env=None, timeout=3600):
    with (OUT / f"{name}.log").open("w") as log:
        r = subprocess.run([str(c) for c in cmd], env=dict(os.environ, **(env or {})),
                           stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
    text = (OUT / f"{name}.log").read_text()
    print(name, "rc=", r.returncode, text[-1200:], flush=True)
    return r.returncode, text


def oracle(model, prefix):
    import torch
    from gguf import GGUFReader
    from gguf.quants import dequantize
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    reader = GGUFReader(model)
    tensors = {t.name: t for t in reader.tensors}
    hp = json.loads(Path(str(prefix) + ".json").read_text())
    def weight(name):
        t = tensors[name]
        shape = tuple(int(x) for x in reversed(t.shape))
        return torch.from_numpy(np.array(dequantize(t.data, t.tensor_type), dtype=np.float32).reshape(shape))
    def rms(x, w):
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + hp["eps"]) * w
    x = torch.from_numpy(np.fromfile(str(prefix) + ".input.f32", dtype=np.float32).reshape(10, hp["d"]))
    cache = []
    for layer in range(hp["layers"]):
        p = f"ralm.blk.{layer}."
        h = rms(x, weight(p + "attn_norm.weight"))
        q = (h @ weight(p + "attn_q.weight").T).view(10, hp["heads"], hp["head_dim"]).transpose(0, 1)
        k = (h @ weight(p + "attn_k.weight").T).view(10, hp["kv_heads"], hp["head_dim"]).transpose(0, 1)
        v = (h @ weight(p + "attn_v.weight").T).view(10, hp["kv_heads"], hp["head_dim"]).transpose(0, 1)
        # The eager C++ host prefix is [position, KV head, head dimension].
        cache.extend([k.transpose(0, 1).contiguous().numpy().ravel(),
                      v.transpose(0, 1).contiguous().numpy().ravel()])
        grp = hp["heads"] // hp["kv_heads"]
        a = torch.nn.functional.scaled_dot_product_attention(
            q, k.repeat_interleave(grp, 0), v.repeat_interleave(grp, 0), is_causal=True)
        a = a.transpose(0, 1).contiguous().view(10, -1)
        x = x + a @ weight(p + "attn_output.weight").T
        h = rms(x, weight(p + "ffn_norm.weight"))
        h = torch.nn.functional.silu(h @ weight(p + "ffn_gate.weight").T) * (h @ weight(p + "ffn_up.weight").T)
        x = x + h @ weight(p + "ffn_down.weight").T
    def metrics(ref, actual):
        ref, actual = ref.astype(np.float64).ravel(), actual.astype(np.float64).ravel()
        assert ref.shape == actual.shape and np.isfinite(actual).all()
        nr, na = np.linalg.norm(ref), np.linalg.norm(actual)
        return {"cosine": float(np.dot(ref, actual) / (nr * na)),
                "norm_ratio": float(na / nr), "relative_error": float(np.linalg.norm(actual - ref) / nr)}
    results = {}
    for arm in ("eager", "batched"):
        results[arm] = metrics(x.numpy(), np.fromfile(str(prefix) + f".{arm}.f32", dtype=np.float32))
    results["kv"] = metrics(np.concatenate(cache), np.fromfile(str(prefix) + ".kv.f32", dtype=np.float32))
    for m in results.values():
        assert m["cosine"] > .9999 and abs(m["norm_ratio"] - 1) < .005 and m["relative_error"] < .01, results
    return results


for cmd in (["uptime"], ["free", "-h"]):
    subprocess.run(cmd, check=True)
if not shutil.which("ccache") or not shutil.which("ninja"):
    subprocess.run(["sudo", "apt-get", "update", "-qq"], check=True)
    subprocess.run(["sudo", "apt-get", "install", "-y", "ccache", "ninja-build"], check=True)
include = SCR / "probe-target.cmake"
include.write_text(f'''add_executable(voxcpm2-ralm-probe "{REPO}/tools/ci-heavy/voxcpm2_ralm_probe.cpp")
target_link_libraries(voxcpm2-ralm-probe PRIVATE voxcpm2_tts)
set_target_properties(voxcpm2-ralm-probe PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${{CMAKE_BINARY_DIR}}/bin")
''')
build = SCR / "build"
rc, _ = run(["cmake", "-G", "Ninja", "-S", REPO, "-B", build,
             f"-DCMAKE_PROJECT_crispasr_INCLUDE={include}", "-DCMAKE_BUILD_TYPE=Release",
             "-DCMAKE_C_COMPILER_LAUNCHER=ccache", "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
             "-DGGML_CUDA=OFF", "-DGGML_VULKAN=OFF", "-DGGML_BLAS=OFF",
             "-DCRISPASR_BUILD_TESTS=OFF", "-DCRISPASR_BUILD_SERVER=OFF",
             "-DCRISPASR_OPUS=OFF", "-DCRISPASR_AMR=OFF"], "configure")
assert rc == 0
rc, _ = run(["cmake", "--build", build, "--target", "voxcpm2-ralm-probe", "crispasr-cli", "-j2"], "build")
assert rc == 0
cli, probe = build / "bin/crispasr", build / "bin/voxcpm2-ralm-probe"
whisper = hf_hub_download("ggerganov/whisper.cpp", "ggml-base.en.bin")
reference = SCR / "synthetic_ref.wav"
TEXT = "Hello, this is a short test sentence."
REF_TEXT = "This reference voice was generated by the model itself for an automated test."
CLONE_TEXT = "The quick brown fox jumps over the lazy dog."
words = lambda text: re.findall(r"[a-z]+", text.lower())
ok = True
for quant in ("q8_0", "f16", "q4_k"):
    model = hf_hub_download("cstr/voxcpm2-GGUF", f"voxcpm2-{quant}.gguf")
    rc, log = run([probe, model, OUT / quant], f"probe-{quant}")
    entry = {"probe_rc": rc, "probe_pass": rc == 0 and "RALM_PREFILL_PASS" in log, "cases": {}}
    match = re.search(r"WARM T=249 eager_ms=([\d.]+) batched_ms=([\d.]+) speedup=([\d.]+)", log)
    entry["warm"] = dict(zip(("eager_ms", "batched_ms", "speedup"), map(float, match.groups()))) if match else None
    if quant == "f16" and entry["probe_pass"]:
        entry["torch_oracle"] = oracle(model, OUT / quant)
    if quant == "q8_0":
        rc, _ = run([cli, "--backend", "voxcpm2", "-m", model, "--tts", REF_TEXT,
                     "--tts-output", reference, "--seed", "7", "-t", "4", "-ng"], "synthetic-reference")
        assert rc == 0 and reference.stat().st_size > 1000
    for case, text in (("zero", TEXT), ("clone", CLONE_TEXT)):
        entry["cases"][case] = {}
        for arm in ("eager", "batched"):
            wav = SCR / f"{quant}-{case}-{arm}.wav"
            cmd = [cli, "--backend", "voxcpm2", "-m", model, "--tts", text, "--tts-output", wav,
                   "--seed", "2", "-t", "4", "-ng"]
            if case == "clone":
                cmd += ["--voice", reference, "--i-have-rights"]
            env = {"CRISPASR_VOXCPM2_RALM_PREFILL_BATCH": "1" if arm == "batched" else "0",
                   "CRISPASR_VOXCPM2_BENCH": "1"}
            rc, log = run(cmd, f"{quant}-{case}-{arm}", env)
            asr_rc, heard = run([cli, "-m", whisper, "-f", wav, "-np", "-nt", "-ng"], f"asr-{quant}-{case}-{arm}") if rc == 0 else (1, "")
            heard_words, want = words(heard), words(text)
            # ASR stdout includes model diagnostics; match the contiguous expected
            # words anywhere, never score just a possibly truncated tail.
            asr_ok = any(heard_words[i:i + len(want)] == want for i in range(len(heard_words) - len(want) + 1))
            stage = re.findall(r"voxcpm2_bench: ralm_prefill\s+([\d.]+) ms", log)
            e = {"rc": rc, "asr_rc": asr_rc, "asr_ok": asr_ok,
                 "ralm_ms": [float(v) for v in stage], "selected": "RALM prefill graph batched" in log,
                 "max_len_hit": "max_len ceiling" in log}
            entry["cases"][case][arm] = e
            ok = ok and rc == 0 and asr_rc == 0 and asr_ok and not e["max_len_hit"] and e["selected"] == (arm == "batched")
    RESULT["models"][quant] = entry
    ok = ok and entry["probe_pass"]
    (OUT / "result.json").write_text(json.dumps(RESULT, indent=2))
RESULT["passed"] = ok
(OUT / "result.json").write_text(json.dumps(RESULT, indent=2))
(OUT / "summary.md").write_text("# RALM CPU prefill proof\n\n```json\n" + json.dumps(RESULT, indent=2) + "\n```\n")
raise SystemExit(0 if ok else 1)
