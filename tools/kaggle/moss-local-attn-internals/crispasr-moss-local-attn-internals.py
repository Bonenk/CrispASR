# %% [markdown]
# # MOSS-TTS-Local 4B — layer-0 attention internals (#249 option 2): flash vs eager
#
# The stop bug is a deterministic graph difference (f32 == f16), seeded at layer 0:
# our attention output is 0.3% off with an EXACT input. Prime suspect: our fused
# ggml flash_attn_ext vs the reference's eager softmax(QK^T/sqrt d)V. Reference-FREE
# decisive check: dump our own layer-0 Q_post_rope / Kfull / Vfull / fa_out (the
# core_attn CRISPASR_CORE_ATTN_DUMP_FA_LAYER hook), recompute eager attention from
# those exact Q/K/V in numpy, and compare to our fa_out.
#   eager != our fa_out -> flash_attn_ext is the bug (scale/mask/accumulation)
#   eager == our fa_out -> flash is fine; the 0.3% is upstream in Q/K/V (rope/qk-norm)
# Also compare our layer-0 attn output (post o_proj) to the HF reference module out.

# %% [code]
import json, os, subprocess, sys, gc, math
from pathlib import Path
import numpy as np

REPO = Path("/kaggle/temp/CrispASR")
WORK = Path("/kaggle/working")
REF = os.environ.get("CRISPASR_REF", "fix/249-moss")
HF = "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5"
TEXT = "Hello world."
if not REPO.exists():
    subprocess.check_call(["git", "clone", "--recursive", "--depth", "1", "--branch", REF,
                           "https://github.com/CrispStrobe/CrispASR.git", str(REPO)])
    subprocess.check_call(["git", "-C", str(REPO), "submodule", "update", "--init",
                           "--recursive", "--depth", "1"], timeout=1800)
sys.path.insert(0, str(REPO / "tools" / "kaggle"))
import kaggle_harness as kh  # noqa: E402
kh.init_progress(hf_progress_repo="cstr/crispasr-kaggle-progress")
step = kh.step
step("start", ref=REF)
TOKEN = kh.resolve_hf_token("HF_TOKEN")
kh.install_build_toolchain()
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub", "hf_transfer"])
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from huggingface_hub import hf_hub_download  # noqa: E402

BUILD = REPO / "build"
step("cmake.configure")
subprocess.run(["cmake", "-G", "Ninja", "-B", str(BUILD), "-S", str(REPO),
                "-DCMAKE_BUILD_TYPE=Release"] + kh.crispasr_cmake_flags(), check=True)
with kh.build_heartbeat("cmake.build"):
    kh.sh_with_progress(f"cmake --build {BUILD} --target crispasr-cli -j{kh.safe_build_jobs(gpu=False)}")
CLI = (BUILD / "bin" / "crispasr") if (BUILD / "bin" / "crispasr").exists() else next(iter(BUILD.rglob("crispasr")))
os.environ["LD_LIBRARY_PATH"] = f"{BUILD/'src'}:{BUILD/'ggml'/'src'}:{os.environ.get('LD_LIBRARY_PATH','')}"
step("build.done")

MODELS = Path("/kaggle/temp/models"); MODELS.mkdir(parents=True, exist_ok=True)
with kh.build_heartbeat("download"):
    F16 = hf_hub_download("cstr/moss-tts-local-v1.5-GGUF", "moss-tts-local-v1.5-f16.gguf", local_dir=str(MODELS))
    CODEC = hf_hub_download("cstr/moss-tts-local-v1.5-GGUF", "moss-tts-local-v1.5-codec.gguf", local_dir=str(MODELS))

# ── our layer-0 attention-internals dump ──────────────────────────────────────
fa_path = WORK / "ours_fa0.txt"
sub_path = WORK / "ours_sub0.txt"
env = {**os.environ, "CRISPASR_CORE_ATTN_DUMP_FA_LAYER": "0",
       "CRISPASR_MOSS_TTS_LOCAL_DUMP_FA_PATH": str(fa_path),
       "CRISPASR_MOSS_TTS_LOCAL_DUMP_SUBLAYER": "0",
       "CRISPASR_MOSS_TTS_LOCAL_DUMP_SUBLAYER_PATH": str(sub_path)}
with kh.build_heartbeat("ours.synth"):
    try:
        subprocess.run([str(CLI), "--backend", "moss-tts-local", "-m", F16, "--codec-model", CODEC,
                        "--tts", TEXT, "--tts-output", str(WORK / "o.wav"), "--no-prints"],
                       env=env, timeout=600, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        pass


def read_ne(path):
    # lines: "<name> ne0 ne1 ne2 ne3 v0 v1 ..." -> {name: (ne, np.array)}
    out = {}
    if path.exists():
        for line in path.read_text().splitlines():
            p = line.split()
            if not p:
                continue
            ne = [int(x) for x in p[1:5]]
            vals = np.array([float(x) for x in p[5:]], dtype=np.float64)
            out[p[0]] = (ne, vals)
    return out


def read_sub(path):
    out = {}
    if path.exists():
        for line in path.read_text().splitlines():
            p = line.split()
            if p:
                out[p[0]] = np.array([float(x) for x in p[1:]], dtype=np.float64)
    return out


fa = read_ne(fa_path)
subo = read_sub(sub_path)
step("ours.done", fa_keys=list(fa.keys()), sub_keys=list(subo.keys()))

# ── reference-free flash-vs-eager check ───────────────────────────────────────
result = {"flash_vs_eager": None}
if all(k in fa for k in ("DBG_Q_post_rope", "DBG_Kfull", "DBG_Vfull", "DBG_fa_out")):
    (qne, qv) = fa["DBG_Q_post_rope"]   # ne=(hd, n_q, T)
    (kne, kv) = fa["DBG_Kfull"]         # ne=(hd, Lk, n_q)  (GQA-expanded)
    (vne, vv) = fa["DBG_Vfull"]         # ne=(hd, Lk, n_q)
    (fne, fv) = fa["DBG_fa_out"]        # ne=(hd, n_q, T) expected
    hd, n_q, T = qne[0], qne[1], qne[2]
    Lk = kne[1]
    Q = qv.reshape(T, n_q, hd)          # [t,h,d]
    K = kv.reshape(n_q, Lk, hd)         # [h,k,d]
    V = vv.reshape(n_q, Lk, hd)         # [h,k,d]
    scale = 1.0 / math.sqrt(hd)
    eager = np.zeros((T, n_q, hd), dtype=np.float64)
    for h in range(n_q):
        Qh = Q[:, h, :]                 # (T, hd)
        Kh = K[h]                       # (Lk, hd)
        Vh = V[h]                       # (Lk, hd)
        scores = (Qh @ Kh.T) * scale    # (T, Lk)
        # causal (prefill, n_past=0): key k visible to query t iff k <= t
        mask = np.triu(np.full((T, Lk), -np.inf), k=1)
        scores = scores + mask
        scores -= scores.max(axis=1, keepdims=True)
        w = np.exp(scores); w /= w.sum(axis=1, keepdims=True)
        eager[:, h, :] = w @ Vh         # (T, hd)
    # fa_out reshape: ne=(hd, n_q, T) -> [t,h,d]
    fa_out = fv.reshape(fne[2], fne[1], fne[0]) if fne[2] == T else fv.reshape(T, n_q, hd)
    a, b = eager.reshape(-1), fa_out.reshape(-1)
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
    l2rel = float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-9))
    # also last-position only (the token that drives the stop)
    al, bl = eager[-1].reshape(-1), fa_out[-1].reshape(-1)
    cos_last = float(al @ bl / (np.linalg.norm(al) * np.linalg.norm(bl) + 1e-9))
    result["flash_vs_eager"] = {"cos": round(cos, 6), "l2rel": round(l2rel, 6),
                                "cos_last": round(cos_last, 6), "hd": hd, "n_q": n_q, "T": T, "Lk": Lk,
                                "fa_ne": fne, "q_ne": qne}
step("flash_vs_eager", **(result["flash_vs_eager"] or {"err": "missing FA tensors"}))

# ── reference module-output comparison (the known 0.3%) ───────────────────────
step("ref.load")
import torch  # noqa: E402
from transformers import AutoModel, AutoProcessor  # noqa: E402
torch.set_num_threads(os.cpu_count() or 4)
model = AutoModel.from_pretrained(HF, trust_remote_code=True, torch_dtype=torch.float32,
                                  attn_implementation="eager", token=TOKEN).eval()
proc = AutoProcessor.from_pretrained(HF, trust_remote_code=True)
ref = {}


def mk(name):
    def hook(_m, _i, o):
        hs = o[0] if isinstance(o, (tuple, list)) else o
        ref[name] = hs[:, -1, :].detach().float().reshape(-1).numpy().astype(np.float64)
    return hook


lyr0 = model.transformer.layers[0]
attn_mod = getattr(lyr0, "self_attn", None) or getattr(lyr0, "attn", None)
if attn_mod is not None:
    attn_mod.register_forward_hook(mk("attn_out"))
    for sub in ("q_norm", "k_norm"):
        m = getattr(attn_mod, sub, None)
        if m is not None:
            m.register_forward_hook(mk(sub))
msg = proc.build_user_message(text=TEXT)
feat = proc([msg], mode="generation")
input_ids = feat["input_ids"] if isinstance(feat, dict) else feat.input_ids
with kh.build_heartbeat("ref.gen"):
    model.generate(input_ids=input_ids, max_new_frames=1, do_sample=False)
del model
gc.collect()


def cos(a, b):
    if a is None or b is None or a.shape != b.shape:
        return None
    return round(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)), 6)


result["attn_out_vs_ref"] = cos(subo.get("sub_attn_0"), ref.get("attn_out"))
verdict = "?"
fve = result["flash_vs_eager"]
if fve is not None:
    if fve["cos_last"] < 0.9995:
        verdict = f"FLASH_ATTN is the bug: our fa_out != numpy-eager (cos_last {fve['cos_last']}, l2rel {fve['l2rel']})"
    else:
        verdict = (f"flash==eager (cos_last {fve['cos_last']}); the 0.3% is UPSTREAM in Q/K/V "
                   f"(rope/qk-norm) — attn_out vs ref cos {result['attn_out_vs_ref']}")
(WORK / "attn_internals.json").write_text(json.dumps({"verdict": verdict, **result}, indent=2))
step("done", verdict=verdict, attn_out_vs_ref=result["attn_out_vs_ref"])
print("DONE", verdict, flush=True)
