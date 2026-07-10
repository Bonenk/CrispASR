"""mistralai/Voxtral-4B-TTS-2603 reference dump backend.

Uses vllm-omni (or manual PyTorch) to run the full TTS pipeline and
captures intermediate activations for crispasr-diff comparison.

Stages:

  raw_text           str           input text (→ GGUF metadata)
  voice_name         str           voice preset name (→ GGUF metadata)
  text_token_ids     (N,)          F32 Tekken BPE token IDs
  voice_embedding    (T_voice, D)  F32 pre-summed voice embeddings
  llm_hidden_frame0  (D,)          F32 LLM hidden state at first generated frame
  semantic_codes     (T_gen,)      F32 semantic token IDs per frame
  acoustic_codes     (T_gen, 36)   F32 acoustic FSQ values per frame
  generated_audio    (N_pcm,)      F32 24 kHz mono PCM output
  generated_text     str           echo of input text (→ GGUF metadata)

Usage:

  python tools/dump_reference.py --backend voxtral-tts \\
      --model-dir /hf/Voxtral-4B-TTS-2603 \\
      --audio samples/jfk.wav \\
      --output /mnt/volume1/tmp-overflow/voxtral-tts-ref.gguf

  Env vars:
    VOXTRAL_TTS_TEXT   — synthesis text (default "Hello world.")
    VOXTRAL_TTS_VOICE  — voice preset (default "fr_female")

Requires: torch, safetensors, sentencepiece
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Set

import numpy as np

DEFAULT_STAGES = [
    "text_token_ids",
    "voice_embedding",
    "semantic_codes",
    "acoustic_codes",
    "generated_audio",
    "generated_text",
]


def dump(*, model_dir: Path, audio: np.ndarray, stages: Set[str],
         max_new_tokens: int) -> Dict[str, np.ndarray]:
    """Run Voxtral TTS reference forward and return stage captures.

    This is a stub that will be populated once the model inference
    flow is validated on Kaggle. For now, it captures the text tokens
    and voice embedding from the model's tokenizer and preset voices.
    """
    import json
    import torch

    out: Dict[str, np.ndarray] = {}
    model_dir = Path(model_dir)

    syn_text = os.environ.get("VOXTRAL_TTS_TEXT", "Hello world.")
    voice_name = os.environ.get("VOXTRAL_TTS_VOICE", "fr_female")

    out["generated_text"] = syn_text

    # Load params
    params_path = model_dir / "params.json"
    if not params_path.exists():
        from huggingface_hub import hf_hub_download
        cache = os.environ.get("HF_HOME", "/tmp") + "/hub"
        params_path = Path(hf_hub_download(str(model_dir), "params.json",
                                            cache_dir=cache))
        model_dir = params_path.parent

    with open(params_path) as f:
        params = json.load(f)

    print(f"  model_type: {params.get('model_type')}")
    print(f"  text: {syn_text!r}")
    print(f"  voice: {voice_name}")

    # Load voice embedding
    if "voice_embedding" in stages:
        voice_path = model_dir / "voice_embedding" / f"{voice_name}.pt"
        if voice_path.exists():
            data = torch.load(str(voice_path), map_location="cpu", weights_only=True)
            if isinstance(data, torch.Tensor):
                out["voice_embedding"] = data.float().numpy()
                print(f"  voice embedding: {out['voice_embedding'].shape}")

    # Tokenize text (Tekken BPE — load and encode)
    if "text_token_ids" in stages:
        try:
            # Try using mistral_common if available
            from mistral_common.tokens.tokenizers.tekken import SpecialTokenPolicy, Tekken
            tekken_path = model_dir / "tekken.json"
            if tekken_path.exists():
                tok = Tekken(str(tekken_path))
                ids = tok.encode(syn_text, bos=False, eos=False)
                out["text_token_ids"] = np.array(ids, dtype=np.float32)
                print(f"  text tokens: {len(ids)}")
        except ImportError:
            print("  mistral_common not available — text_token_ids skipped")

    # Full pipeline stages require vllm-omni or manual inference
    # (too large for a simple reference dumper — stub for now)
    if {"semantic_codes", "acoustic_codes", "generated_audio"} & stages:
        print("  NOTE: full pipeline stages require vllm-omni inference")
        print("  These will be captured on the Kaggle GPU kernel")

    return out
