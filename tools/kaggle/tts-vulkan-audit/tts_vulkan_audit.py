#!/usr/bin/env python3
"""
#304 follow-up — Vulkan audit of every CrispASR TTS backend SubtitleEdit ships.

SubtitleEdit downloads the *Vulkan* Windows build for every Windows user
(CrispAsrDownloadService.GetUrl() -> WindowsVulkanUrl) and drives these backends
through the crispasr /v1/audio/speech server. cosyvoice3 was confirmed to emit
blank/garbled audio on Vulkan (AR decode collapses; conv vocoder corrupts) and
fixed by routing to CPU under Vulkan. This kernel checks whether the OTHER
SE-exposed TTS backends have the same bug on a REAL NVIDIA Vulkan driver.

Method per backend: synthesize one sentence under --gpu-backend vulkan and under
--no-gpu (CPU baseline), ASR-roundtrip both with whisper-tiny.en, and compare.
A backend is FLAGGED if CPU produces the sentence but Vulkan produces
empty/near-silent/garbage (word-overlap with the CPU transcript collapses).

Uses the prebuilt crispasr-linux-x86_64-vulkan.tar.gz (v0.8.22) — so cosyvoice3
here is UNFIXED and serves as a positive control (must reproduce the bug).
All model repos are public (cstr/*); no HF token required.
"""
import os, sys, subprocess, json, time, wave, math, array, re, urllib.request
from pathlib import Path

WORK = Path("/kaggle/working")
TMP = Path("/tmp/ttsaudit"); TMP.mkdir(parents=True, exist_ok=True)
MODELS = Path("/tmp/models"); MODELS.mkdir(parents=True, exist_ok=True)
PROG = WORK / "progress.txt"
RESULTS = WORK / "audit_results.json"
HERE = Path(__file__).resolve().parent

def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    with open(PROG, "a") as f:
        f.write(line + "\n")

sys.path.insert(0, str(HERE))
try:
    import kaggle_harness as kh
    kh.init_progress()
except Exception as e:
    log(f"harness import skipped (repos are public): {e}")

RELEASE = "v0.8.22"
VK_TARBALL = f"https://github.com/CrispStrobe/CrispASR/releases/download/{RELEASE}/crispasr-linux-x86_64-vulkan.tar.gz"
WHISPER_TINY = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin"
SENT = "The quick brown fox jumps over the lazy dog near the river."
JFK_TEXT = ("And so my fellow Americans, ask not what your country can do for you, "
            "ask what you can do for your country.")
REF_WAV = str(HERE / "jfk.wav")

def sh(cmd, timeout=None, env=None):
    e = dict(os.environ)
    if env:
        e.update(env)
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout, env=e)

# ----------------------------------------------------------------------------
# 1. Enable Vulkan on the NVIDIA box
# ----------------------------------------------------------------------------
def enable_vulkan():
    log("installing vulkan loader + espeak-ng ...")
    sh("apt-get update -qq")
    sh("DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "
       "libvulkan1 vulkan-tools espeak-ng libespeak-ng1")
    r = sh("vulkaninfo --summary 2>/dev/null")
    if "NVIDIA" not in r.stdout:
        # NVIDIA driver present but no ICD registered — create it.
        for lib in ("libGLX_nvidia.so.0", "libnvidia-vulkan-producer.so"):
            probe = sh(f"ldconfig -p | grep -F {lib}")
            if probe.stdout.strip():
                os.makedirs("/usr/share/vulkan/icd.d", exist_ok=True)
                icd = ('{"file_format_version":"1.0.0","ICD":'
                       '{"library_path":"libGLX_nvidia.so.0","api_version":"1.3.277"}}')
                Path("/usr/share/vulkan/icd.d/nvidia_icd.json").write_text(icd)
                log("registered nvidia_icd.json manually")
                break
        r = sh("vulkaninfo --summary 2>/dev/null")
    devs = [l.strip() for l in r.stdout.splitlines() if "deviceName" in l]
    log("Vulkan devices:\n  " + ("\n  ".join(devs) if devs else "(none enumerated)"))
    return r.stdout

# ----------------------------------------------------------------------------
# 2. Fetch prebuilt vulkan crispasr + whisper
# ----------------------------------------------------------------------------
def fetch_binaries():
    log(f"downloading prebuilt vulkan crispasr {RELEASE} ...")
    tb = TMP / "crispasr-vulkan.tar.gz"
    urllib.request.urlretrieve(VK_TARBALL, tb)
    sh(f"cd {TMP} && tar xzf {tb}")
    bins = list(TMP.rglob("crispasr"))
    bins = [b for b in bins if b.is_file() and os.access(b, os.X_OK)]
    if not bins:
        raise RuntimeError("crispasr binary not found in tarball")
    binp = bins[0]
    sh(f"chmod +x {binp}")
    # prebuilt release ships shared libs (libggml-vulkan.so, ...) beside the binary
    bindir = str(binp.parent)
    os.environ["LD_LIBRARY_PATH"] = bindir + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    log(f"crispasr: {binp}  (LD_LIBRARY_PATH={bindir})")
    ver = sh(f"{binp} --version")
    log(ver.stdout.strip() + ver.stderr.strip())
    whisp = MODELS / "ggml-tiny.en.bin"
    if not whisp.exists():
        log("downloading whisper-tiny.en ...")
        urllib.request.urlretrieve(WHISPER_TINY, whisp)
    return str(binp), str(whisp)

# ----------------------------------------------------------------------------
# 3. helpers: download HF files, wav stats, ASR, word-overlap
# ----------------------------------------------------------------------------
def hf_get(repo, fname, dest_dir):
    dest = Path(dest_dir) / fname
    if dest.exists() and dest.stat().st_size > 0:
        return str(dest)
    url = f"https://huggingface.co/{repo}/resolve/main/{fname}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    log(f"  hf get {repo}/{fname}")
    urllib.request.urlretrieve(url, dest)
    return str(dest)

def wav_stats(path):
    try:
        w = wave.open(path, "rb")
        a = array.array("h"); a.frombytes(w.readframes(w.getnframes()))
        if not len(a):
            return dict(dur=0.0, peak=0.0, rms=0.0)
        peak = max(abs(x) for x in a) / 32768.0
        rms = math.sqrt(sum((x / 32768.0) ** 2 for x in a) / len(a))
        return dict(dur=w.getnframes() / w.getframerate(), peak=round(peak, 4), rms=round(rms, 4))
    except Exception as e:
        return dict(dur=0.0, peak=0.0, rms=0.0, err=str(e))

def asr(binp, whisp, wav):
    if not Path(wav).exists():
        return ""
    r = sh(f"{binp} -m {whisp} -f {wav} --language en --no-gpu", timeout=300)
    txt = " ".join(re.sub(r"\[[^\]]*\]", "", l) for l in r.stdout.splitlines() if l.strip().startswith("["))
    return re.sub(r"\s+", " ", txt).strip()

def words(s):
    return set(re.findall(r"[a-z']+", s.lower()))

def overlap(a, b):
    wa, wb = words(a), words(b)
    if not wa:
        return 0.0
    return round(len(wa & wb) / len(wa), 3)

# ----------------------------------------------------------------------------
# 4. backend matrix
# ----------------------------------------------------------------------------
CLONE = ["--voice", REF_WAV, "--ref-text", JFK_TEXT, "--i-have-rights"]
CLONE_NOTEXT = ["--voice", REF_WAV, "--i-have-rights"]

BACKENDS = [
    # control: unfixed cosyvoice3 MUST reproduce the bug on vulkan
    dict(name="cosyvoice3-tts", repo="cstr/cosyvoice3-0.5b-2512-GGUF",
         main="cosyvoice3-llm-q4_k.gguf",
         files=["cosyvoice3-llm-q4_k.gguf", "cosyvoice3-flow-q8_0.gguf",
                "cosyvoice3-hift-f16.gguf", "cosyvoice3-voices.gguf"],
         args=["--voice", "zero_shot"], control=True),
    dict(name="f5-tts", repo="cstr/f5-tts-GGUF",
         main="f5-tts-v1-base-f16.gguf", files=["f5-tts-v1-base-f16.gguf"], args=CLONE),
    dict(name="qwen3-tts", repo="cstr/qwen3-tts-0.6b-base-GGUF",
         main="qwen3-tts-12hz-0.6b-base-q8_0.gguf",
         files=["qwen3-tts-12hz-0.6b-base-q8_0.gguf"],
         extra=[("cstr/qwen3-tts-tokenizer-12hz-GGUF", ["qwen3-tts-tokenizer-12hz.gguf"])],
         args=CLONE),
    dict(name="moss-tts", repo="cstr/moss-tts-v1.5-GGUF",
         main="moss-tts-v1.5-q4_k.gguf",
         files=["moss-tts-v1.5-q4_k.gguf", "moss-tts-v1.5-codec.gguf"],
         args=["--codec-model", "{dir}/moss-tts-v1.5-codec.gguf"] + CLONE),
    dict(name="vibevoice-1.5b", repo="cstr/vibevoice-1.5b-GGUF",
         main="vibevoice-1.5b-tts-q8_0.gguf",
         files=["vibevoice-1.5b-tts-q8_0.gguf"], args=CLONE_NOTEXT),
    dict(name="voxcpm2-tts", repo="cstr/voxcpm2-GGUF",
         main="voxcpm2-q4_k.gguf",
         files=["voxcpm2-q4_k.gguf", "voxcpm2-ref.gguf"], args=CLONE),
    dict(name="zonos", repo="cstr/zonos-v0.1-transformer-GGUF",
         main="zonos-v0.1-transformer-q8_0.gguf",
         files=["zonos-v0.1-transformer-q8_0.gguf"],
         extra=[("cstr/dac-44khz-GGUF", ["dac-44khz-f16.gguf"])],
         args=CLONE_NOTEXT),
    dict(name="indextts", repo="cstr/indextts-1.5-GGUF",
         main="indextts-gpt-q8_0.gguf",
         files=["indextts-gpt-q8_0.gguf", "indextts-bigvgan.gguf"], args=CLONE_NOTEXT),
]

def run_synth(binp, cfg, mdir, mode):
    out = str(TMP / f"{cfg['name']}.{mode}.wav")
    if os.path.exists(out):
        os.remove(out)
    args = [a.replace("{dir}", mdir) for a in cfg["args"]]
    gpuflag = ["--gpu-backend", "vulkan"] if mode == "vulkan" else ["--no-gpu"]
    cmd = ([binp, "-m", f"{mdir}/{cfg['main']}", "--backend", cfg["name"]]
           + gpuflag + args + ["--tts", SENT, "--tts-output", out])
    e = dict(os.environ)
    if mode == "vulkan":
        e["GGML_VK_VISIBLE_DEVICES"] = "0"
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=900, env=e)
    tail = "\n".join((r.stderr or "").splitlines()[-8:])
    ntok = ""
    m = re.search(r"generated (\d+) (?:speech )?tokens", r.stderr or "")
    if m:
        ntok = m.group(1)
    return out, r.returncode, ntok, tail

def audit_backend(binp, whisp, cfg):
    name = cfg["name"]
    log(f"════ {name} ════")
    mdir = str(MODELS / name); os.makedirs(mdir, exist_ok=True)
    rec = dict(backend=name, control=cfg.get("control", False))
    try:
        for f in cfg["files"]:
            hf_get(cfg["repo"], f, mdir)
        for erepo, efiles in cfg.get("extra", []):
            for f in efiles:
                hf_get(erepo, f, mdir)
    except Exception as e:
        rec["error"] = f"download failed: {e}"
        log(f"  DOWNLOAD FAILED: {e}")
        return rec
    for mode in ("cpu", "vulkan"):
        try:
            out, rc, ntok, tail = run_synth(binp, cfg, mdir, mode)
            st = wav_stats(out) if os.path.exists(out) else dict(dur=0, peak=0, rms=0)
            tx = asr(binp, whisp, out) if os.path.exists(out) else ""
            rec[mode] = dict(rc=rc, tokens=ntok, dur=st.get("dur"), peak=st.get("peak"),
                             rms=st.get("rms"), asr=tx, overlap_sent=overlap(SENT, tx))
            log(f"  [{mode}] rc={rc} tok={ntok} dur={st.get('dur')} peak={st.get('peak')} "
                f"ov={rec[mode]['overlap_sent']} asr='{tx[:60]}'")
            if rc != 0:
                log(f"    stderr tail: {tail}")
        except subprocess.TimeoutExpired:
            rec[mode] = dict(rc="timeout")
            log(f"  [{mode}] TIMEOUT")
        except Exception as e:
            rec[mode] = dict(rc="exc", err=str(e))
            log(f"  [{mode}] EXC {e}")
    # verdict: cpu intelligible but vulkan not => Vulkan-broken
    cpu, vk = rec.get("cpu", {}), rec.get("vulkan", {})
    cpu_ok = isinstance(cpu.get("overlap_sent"), float) and cpu["overlap_sent"] >= 0.5 and (cpu.get("peak") or 0) > 0.02
    vk_ok = isinstance(vk.get("overlap_sent"), float) and vk["overlap_sent"] >= 0.5 and (vk.get("peak") or 0) > 0.02
    if cpu_ok and not vk_ok:
        rec["verdict"] = "VULKAN_BROKEN"
    elif cpu_ok and vk_ok:
        rec["verdict"] = "vulkan_ok"
    elif not cpu_ok:
        rec["verdict"] = "cpu_baseline_failed(inconclusive)"
    else:
        rec["verdict"] = "unclear"
    log(f"  ==> {rec['verdict']}")
    # free disk
    sh(f"rm -rf {mdir}")
    return rec

def main():
    t0 = time.time()
    vk_info = enable_vulkan()
    binp, whisp = fetch_binaries()
    results = dict(release=RELEASE, sentence=SENT, vulkan_devices=vk_info, backends=[])
    for cfg in BACKENDS:
        try:
            results["backends"].append(audit_backend(binp, whisp, cfg))
        except Exception as e:
            log(f"backend {cfg['name']} crashed: {e}")
            results["backends"].append(dict(backend=cfg["name"], error=str(e)))
        RESULTS.write_text(json.dumps(results, indent=2))
    # summary
    log("═══════════ SUMMARY ═══════════")
    for r in results["backends"]:
        log(f"  {r['backend']:16s} {r.get('verdict', r.get('error',''))}")
    log(f"done in {int(time.time()-t0)}s")
    RESULTS.write_text(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
