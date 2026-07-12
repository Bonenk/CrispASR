#!/usr/bin/env python3
"""MOSS-TTS-v1.5 (MossTTSDelay) reference dumper for the crispasr diff harness.

Produces the GREEDY audio code grid the C++ runtime must reproduce byte-for-byte
(Phase 3 code-parity gate), and optionally the codec-decoded reference waveform
(Phase 4). The model's own `generate(do_sample=False)` IS the greedy reference —
we drive it and extract the (n_vq, T_audio) code matrix.

Run inside a GPU box with the 8B model (Kaggle) — see
tools/kaggle/moss-tts-validate/. The high-level HF API surface (processor method
names, generate return shape) is confirmed on that first run; the extraction
below is defensive and logs the actual structure so any mismatch is a one-line
fix, not a silent wrong reference.

Env:
  MOSS_TTS_MODEL   HF id or local dir of MOSS-TTS-v1.5 (default the HF id)
  MOSS_TTS_TEXT    text to synthesize (fixed for parity)
  MOSS_TTS_SEED    RNG seed (default 0; greedy is seed-independent but set it)
  MOSS_TTS_MAXNEW  max_new_tokens (default 512)
  MOSS_TTS_CODEC   HF id or dir of MOSS-Audio-Tokenizer (optional; decode ref)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Set

import numpy as np

DEFAULT_STAGES = [
    "codes",       # (n_vq, T_audio) int32 — the greedy code grid (parity target)
    "waveform",    # (T_audio*1920,) f32 — codec-decoded reference (if codec given)
]

DEFAULT_MODEL = "OpenMOSS-Team/MOSS-TTS-v1.5"
DEFAULT_CODEC = "OpenMOSS-Team/MOSS-Audio-Tokenizer"


def _extract_codes(model, out: Any) -> np.ndarray:
    """Pull the (n_vq, T_audio) int code grid out of a MossTTSDelay generate
    result. Tries the documented attributes, then falls back to scanning for the
    first 2D integer tensor whose minor axis == n_vq. Raises with a dump of the
    structure if nothing matches (so the Kaggle run tells us the real shape)."""
    import torch

    n_vq = int(getattr(model.config, "n_vq", 32))

    def as_np(x):
        return x.detach().to(torch.int32).cpu().numpy() if hasattr(x, "detach") else np.asarray(x)

    # 1. Common attribute names on the custom output object / dict.
    for attr in ("audio_codes", "codes", "audio_tokens", "sequences_audio"):
        v = getattr(out, attr, None) if not isinstance(out, dict) else out.get(attr)
        if v is not None:
            arr = as_np(v)
            arr = np.squeeze(arr)
            if arr.ndim == 2:
                # Orient to (n_vq, T): put the n_vq axis first.
                if arr.shape[0] != n_vq and arr.shape[1] == n_vq:
                    arr = arr.T
                return arr.astype(np.int32)

    # 2. Fallback: scan every tensor-like member for a 2D int grid with an
    #    axis == n_vq.
    candidates = []
    members = out.items() if isinstance(out, dict) else vars(out).items() if hasattr(out, "__dict__") else []
    for name, v in members:
        try:
            arr = np.squeeze(as_np(v))
        except Exception:  # noqa: BLE001
            continue
        if arr.ndim == 2 and n_vq in arr.shape and np.issubdtype(arr.dtype, np.integer):
            candidates.append((name, arr))
    if candidates:
        name, arr = candidates[0]
        print(f"[moss_tts_ref] using generate output member '{name}' shape={arr.shape}", flush=True)
        if arr.shape[0] != n_vq and arr.shape[1] == n_vq:
            arr = arr.T
        return arr.astype(np.int32)

    raise RuntimeError(
        "moss_tts_ref: could not locate the audio code grid in the generate output. "
        f"n_vq={n_vq}. Output type={type(out)}; "
        f"members={[k for k, _ in members]}. "
        "Inspect on Kaggle and point _extract_codes at the right field.")


def dump(
    model_dir: "Path | str | None" = None,
    audio: np.ndarray | None = None,  # ignored (TTS has no input audio)
    stages: Set[str] | None = None,
    max_new_tokens: int | None = None,
    **kwargs: Any,
) -> Dict[str, np.ndarray]:
    """Return {"codes": (n_vq, T) int32[, "waveform": (n,) f32]} for the greedy
    reference. Serialized to a GGUF fixture by the harness."""
    import torch

    if stages is None:
        stages = set(DEFAULT_STAGES)
    model_id = str(model_dir or os.environ.get("MOSS_TTS_MODEL", DEFAULT_MODEL))
    text = os.environ.get("MOSS_TTS_TEXT", "The quick brown fox jumps over the lazy dog.")
    seed = int(os.environ.get("MOSS_TTS_SEED", "0"))
    if max_new_tokens is None:
        max_new_tokens = int(os.environ.get("MOSS_TTS_MAXNEW", "512"))
    torch.manual_seed(seed)

    from transformers import AutoModel, AutoProcessor

    # The 8B bf16 (~16 GB) doesn't fit a 16 GB GPU (Kaggle P100/T4) — force CPU
    # there via MOSS_TTS_REF_DEVICE=cpu (fits ~29 GB host RAM in bf16).
    dev_env = os.environ.get("MOSS_TTS_REF_DEVICE", "auto")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if dev_env == "auto" else dev_env
    dtype = torch.bfloat16  # keep both greedy sides comparable; bf16 fits CPU RAM
    print(f"[moss_tts_ref] loading {model_id} on {device} ({dtype})", flush=True)
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True, torch_dtype=dtype).to(device).eval()
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    # Build the user turn via the processor (mirrors build_user_message /
    # the C++ prompt template). Fall back to a plain text prompt if the
    # processor doesn't expose the helper under the expected name.
    build = getattr(processor, "build_user_message", None) or getattr(processor, "apply_chat_template", None)
    if build is not None:
        try:
            inputs = build(text=text, return_tensors="pt")
        except TypeError:
            inputs = processor(text=text, return_tensors="pt")
    else:
        inputs = processor(text=text, return_tensors="pt")
    inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in dict(inputs).items()}

    print(f"[moss_tts_ref] greedy generate (max_new_tokens={max_new_tokens})", flush=True)
    with torch.no_grad():
        out = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens,
                             return_dict_in_generate=True)

    codes = _extract_codes(model, out)
    print(f"[moss_tts_ref] codes shape={codes.shape} dtype={codes.dtype} "
          f"range=[{codes.min()},{codes.max()}]", flush=True)
    results: Dict[str, np.ndarray] = {"codes": codes.astype(np.int32)}

    # Optional: decode the reference waveform via the codec.
    if "waveform" in stages:
        codec_id = os.environ.get("MOSS_TTS_CODEC", "")
        if codec_id:
            try:
                codec = AutoModel.from_pretrained(codec_id, trust_remote_code=True,
                                                  torch_dtype=torch.float32).to(device).eval()
                with torch.no_grad():
                    ct = torch.from_numpy(codes.astype(np.int64)).unsqueeze(0).to(device)
                    dec = codec.decode(ct) if hasattr(codec, "decode") else codec(ct)
                wav = np.squeeze(dec.detach().float().cpu().numpy())
                results["waveform"] = wav.astype(np.float32)
                print(f"[moss_tts_ref] waveform shape={results['waveform'].shape}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"[moss_tts_ref] codec decode skipped: {type(e).__name__}: {e}", flush=True)

    return results


# Backwards-compatible alias used by some kernels (ref.run(model, idx, out_dir)).
def run(model_dir=None, idx: int = 0, out_dir: "Path | str | None" = None, **kwargs) -> Dict[str, np.ndarray]:
    res = dump(model_dir=model_dir, **kwargs)
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for k, v in res.items():
            np.save(out_dir / f"{k}.npy", v)
    return res


if __name__ == "__main__":
    import sys

    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("moss_tts_ref")
    r = run(out_dir=out)
    for k, v in r.items():
        print(f"  {k}: shape={v.shape} dtype={v.dtype}")
