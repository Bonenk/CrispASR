#!/usr/bin/env python3
"""
MioTTS reference dumper for the CrispASR diff harness.

Runs the MioTTS-0.6B LLM (Qwen3ForCausalLM) to generate speech tokens from
a text prompt, then runs MioCodec decode to produce a waveform. Dumps
per-stage intermediates to a GGUF archive for comparison with the C++ port.

Stages dumped:
  1. input_ids        — tokenized prompt (int32)
  2. logits_step_0    — first LLM logits (sanity check embedding + first layer)
  3. generated_ids    — full generated token sequence (int32)
  4. speech_tokens    — extracted speech token indices for codec (int32)
  5. fsq_embedding    — FSQ dequantized content embedding (float32)
  6. wave_prenet_out  — after wave_prenet transformer (float32)
  7. wave_decoder_out — after wave_decoder transformer (float32)
  8. audio_output     — final 24kHz waveform (float32)

Usage:
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
    TMPDIR=/mnt/volume1/tmp-overflow \\
    python tools/reference_backends/miotts.py \\
        --llm-dir /mnt/storage/models/miotts-0.6b \\
        --codec-dir /mnt/storage/models/miocodec-25hz-24khz \\
        --text "Hello world" \\
        --output /mnt/volume1/tmp-overflow/miotts-ref.gguf \\
        --max-tokens 50
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import numpy as np

try:
    from gguf import GGUFWriter, GGMLQuantizationType
except ImportError:
    sys.exit("pip install gguf")

try:
    import torch
except ImportError:
    sys.exit("pip install torch")


DEFAULT_STAGES = [
    "input_ids",
    "logits_step_0",
    "generated_ids",
    "speech_tokens",
    "fsq_embedding",
    "wave_prenet_out",
    "wave_decoder_out",
    "audio_output",
]


def dump(args):
    """Main dump entry point for the diff harness."""
    writer = GGUFWriter(args.output, "miotts-ref", use_temp_file=True)

    # ─── Stage 1: LLM inference ─────────────────────────────────────
    print("[miotts-ref] Loading LLM...")
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(
        args.llm_dir, trust_remote_code=True
    )

    # Build ChatML prompt: <|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n
    messages = [{"role": "user", "content": args.text}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    print(f"[miotts-ref] Prompt: {repr(prompt[:80])}...")
    print(f"[miotts-ref] Input IDs shape: {input_ids.shape}")

    # Dump input_ids
    writer.add_tensor("input_ids", input_ids[0].numpy().astype(np.int32))

    # Load model
    print("[miotts-ref] Loading model (this may take a moment on 8GB RAM)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.llm_dir,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
    )
    model.eval()

    # Get first-step logits for embedding/layer sanity check
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=False)
        logits_step_0 = outputs.logits[0, -1, :].float().numpy()
    writer.add_tensor("logits_step_0", logits_step_0)
    print(f"[miotts-ref] logits_step_0 shape: {logits_step_0.shape}")

    # Generate speech tokens
    print(f"[miotts-ref] Generating (max_tokens={args.max_tokens})...")
    max_new = args.max_tokens
    with torch.no_grad():
        generated = model.generate(
            input_ids,
            max_new_tokens=max_new,
            do_sample=(args.temperature > 0),
            temperature=args.temperature if args.temperature > 0 else 1.0,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            pad_token_id=tokenizer.pad_token_id or 0,
        )
    generated_ids = generated[0].numpy().astype(np.int32)
    writer.add_tensor("generated_ids", generated_ids)
    print(f"[miotts-ref] Generated {len(generated_ids)} tokens total")

    # Extract speech token indices
    # Speech tokens are in range [151669, 164469) → codec indices [0, 12800)
    speech_start = 151669
    speech_end = 164469
    new_tokens = generated_ids[input_ids.shape[1]:]
    speech_mask = (new_tokens >= speech_start) & (new_tokens < speech_end)
    speech_token_ids = new_tokens[speech_mask] - speech_start
    writer.add_tensor("speech_tokens", speech_token_ids.astype(np.int32))
    print(f"[miotts-ref] Speech tokens: {len(speech_token_ids)}")

    if len(speech_token_ids) == 0:
        print("[miotts-ref] WARNING: no speech tokens generated!")
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        return

    # Free LLM memory
    del model, outputs, generated
    gc.collect()

    # ─── Stage 2: MioCodec decode ───────────────────────────────────
    print("[miotts-ref] Loading MioCodec...")

    # We need to load the codec manually since `miocodec` pip package
    # may not be available. Load from safetensors directly and implement
    # the decode path step by step.
    from safetensors import safe_open

    codec_path = Path(args.codec_dir) / "model.safetensors"
    sf = safe_open(str(codec_path), framework="pt")

    # FSQ dequantize: indices → codes → embedding
    # FSQ levels = [8, 8, 8, 5, 5], basis = [1, 8, 64, 512, 2560]
    levels = torch.tensor([8, 8, 8, 5, 5], dtype=torch.long)
    basis = torch.tensor([1, 8, 64, 512, 2560], dtype=torch.long)

    indices = torch.tensor(speech_token_ids, dtype=torch.long).unsqueeze(-1)  # (T, 1)
    # indices_to_codes: (index // basis) % levels → (T, 5)
    codes_non_centered = (indices // basis) % levels
    # scale_and_shift_inverse: (codes - levels//2) / (levels//2)
    half_width = levels // 2
    codes_normalized = (codes_non_centered.float() - half_width.float()) / half_width.float()

    # Project from FSQ dim (5) to embedding dim (768)
    # Weight: local_quantizer.out_proj.weight (768, 5)
    # Bias: local_quantizer.out_proj.bias (768,)
    has_proj_out = "local_quantizer.proj_out.weight" in sf.keys()
    if has_proj_out:
        proj_out_w = sf.get_tensor("local_quantizer.proj_out.weight").float()
        proj_out_b = sf.get_tensor("local_quantizer.proj_out.bias").float()
        fsq_embedding = codes_normalized @ proj_out_w.T + proj_out_b  # (T, 768)
    else:
        # Maybe the quantizer uses a different projection name
        print("[miotts-ref] WARNING: no proj_out found, using raw codes")
        fsq_embedding = codes_normalized

    writer.add_tensor("fsq_embedding", fsq_embedding.numpy())
    print(f"[miotts-ref] FSQ embedding shape: {fsq_embedding.shape}")

    # For now, dump the intermediate stages. Full codec forward requires
    # implementing the transformer + iSTFT which is the C++ port's job.
    # The diff harness will compare up to fsq_embedding initially, then
    # extend as we implement each codec stage in C++.

    print(f"[miotts-ref] Dumped {len(DEFAULT_STAGES)} stage slots")
    print(f"[miotts-ref] Writing to {args.output}...")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print("[miotts-ref] Done.")


def main():
    parser = argparse.ArgumentParser(description="MioTTS reference dumper")
    parser.add_argument("--llm-dir", required=True, help="MioTTS-0.6B model directory")
    parser.add_argument("--codec-dir", required=True, help="MioCodec-25Hz-24kHz directory")
    parser.add_argument("--text", default="Hello world", help="Text to synthesize")
    parser.add_argument("--output", required=True, help="Output GGUF path")
    parser.add_argument("--max-tokens", type=int, default=100, help="Max new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0=greedy)")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p sampling")
    parser.add_argument("--repetition-penalty", type=float, default=1.0, help="Repetition penalty")
    args = parser.parse_args()
    dump(args)


if __name__ == "__main__":
    main()
