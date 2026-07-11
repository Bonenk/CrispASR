# MOSS-Transcribe-Diarize 0.9B Backend (#242) — Handover

## What This Model Does

Joint ASR + speaker diarization + timestamps in a single 0.9B model.
Input: audio. Output: `[0.48][S01]Welcome everyone[1.66][12.26][S02]The new pipeline is ready[13.81]`

Apache-2.0, multi-language, hotword support. Tiny: 1.82 GB BF16, ~500 MB Q4_K.

## Architecture

```
16 kHz PCM → 80-bin Whisper mel → Conv1d stem → 24L Whisper encoder (1024d)
           → 4x temporal merge (reshape T/4 × 4096) → VQAdaptor (4096→1024)
           → [time markers injected every 5s into audio pad sequence]
           → Qwen3-0.6B LM (28L, 1024d, 16Q/8KV, tied embeddings)
           → text with [timestamps][Sxx] speaker labels
```

## This Is a NEW Backend (Not a Delta on moss-transcribe)

| | moss-transcribe (preview-2B) | moss-transcribe-diarize (0.9B) |
|---|---|---|
| Encoder | Qwen3-Omni-MoE (Conv2d, 32L, 1280d, 128 mel, windowed attn) | Stock Whisper (Conv1d, 24L, 1024d, 80 mel, global attn) |
| Adapter | GatedMLP (2048→8192→2048, no bias) | VQAdaptor (Linear+SiLU+Linear+LayerNorm, 4096→1024, bias) |
| Merge | None (1:1) | 4x temporal merge |
| LM | Qwen3-1.7B (2048d) | Qwen3-0.6B (1024d) |
| Output | Plain text | `[timestamp][Sxx]text[timestamp]` |
| Diarization | No | Yes |
| Timestamps | No | Yes |
| Hotwords | No | Yes (prompt-level) |

## Reusable Components

| Component | Source | What to reuse |
|---|---|---|
| **Whisper encoder** | `crisp_audio` (used by qwen3_asr, higgs_stt) | Full encoder: mel → Conv1d stem → 24L transformer → output |
| **Qwen3 LM decoder** | `core_attn::kv_self_attn` + `core_ffn::swiglu` | Same as every other LLM backend |
| **BPE tokenizer** | `core/bpe.h` | Same Qwen3 vocab (151936) |
| **Mel computation** | `core/mel.h` | Whisper mel (80 bins, n_fft=400, hop=160) |

## Files to Create

1. `models/convert-moss-transcribe-diarize-to-gguf.py` — converter
2. `src/moss_transcribe_diarize.h` — C API
3. `src/moss_transcribe_diarize.cpp` — runtime
4. `examples/cli/crispasr_backend_moss_transcribe_diarize.cpp` — CLI adapter

## Converter Tensor Mapping

```python
# Whisper encoder (24 layers)
"model.whisper_encoder.conv1.weight"     → "enc.conv1.weight"      # (1024, 80, 3)
"model.whisper_encoder.conv1.bias"       → "enc.conv1.bias"
"model.whisper_encoder.conv2.weight"     → "enc.conv2.weight"      # (1024, 1024, 3)
"model.whisper_encoder.conv2.bias"       → "enc.conv2.bias"
"model.whisper_encoder.embed_positions.weight" → "enc.pos_embed.weight"  # (1500, 1024)
"model.whisper_encoder.layer_norm.weight/bias" → "enc.ln_post.weight/bias"
"model.whisper_encoder.layers.{i}.self_attn.q_proj.weight/bias" → "enc.blk.{i}.attn_q.weight/bias"
"model.whisper_encoder.layers.{i}.self_attn.k_proj.weight"      → "enc.blk.{i}.attn_k.weight"  # NO bias
"model.whisper_encoder.layers.{i}.self_attn.v_proj.weight/bias" → "enc.blk.{i}.attn_v.weight/bias"
"model.whisper_encoder.layers.{i}.self_attn.out_proj.weight/bias" → "enc.blk.{i}.attn_out.weight/bias"
"model.whisper_encoder.layers.{i}.self_attn_layer_norm.weight/bias" → "enc.blk.{i}.attn_norm.weight/bias"
"model.whisper_encoder.layers.{i}.fc1.weight/bias"              → "enc.blk.{i}.ffn_up.weight/bias"
"model.whisper_encoder.layers.{i}.fc2.weight/bias"              → "enc.blk.{i}.ffn_down.weight/bias"
"model.whisper_encoder.layers.{i}.final_layer_norm.weight/bias" → "enc.blk.{i}.ffn_norm.weight/bias"

# VQAdaptor (3 parameterized layers)
"model.vq_adaptor.layers.0.weight/bias"  → "adaptor.fc1.weight/bias"    # (1024, 4096)
"model.vq_adaptor.layers.2.weight/bias"  → "adaptor.fc2.weight/bias"    # (1024, 1024)
"model.vq_adaptor.layers.3.weight/bias"  → "adaptor.norm.weight/bias"   # LayerNorm(1024)

# Qwen3-0.6B LM (28 layers, tied embeddings)
"model.language_model.embed_tokens.weight"  → "token_embd.weight"  # (151936, 1024)
"model.language_model.norm.weight"          → "output_norm.weight"
"model.language_model.layers.{i}.input_layernorm.weight" → "blk.{i}.attn_norm.weight"
"model.language_model.layers.{i}.self_attn.q_proj.weight" → "blk.{i}.attn_q.weight"
"model.language_model.layers.{i}.self_attn.q_norm.weight" → "blk.{i}.q_norm.weight"
# ... same pattern as existing Qwen3 backends
```

## KV Metadata

```python
writer.add_uint32("moss_diarize.enc.n_layers", 24)
writer.add_uint32("moss_diarize.enc.d_model", 1024)
writer.add_uint32("moss_diarize.enc.n_heads", 16)
writer.add_uint32("moss_diarize.enc.n_mels", 80)
writer.add_uint32("moss_diarize.adaptor.in_dim", 4096)  # after 4x merge
writer.add_uint32("moss_diarize.adaptor.out_dim", 1024)
writer.add_uint32("moss_diarize.llm.n_layers", 28)
writer.add_uint32("moss_diarize.llm.dim", 1024)
writer.add_uint32("moss_diarize.llm.n_heads", 16)
writer.add_uint32("moss_diarize.llm.n_kv_heads", 8)
writer.add_uint32("moss_diarize.llm.head_dim", 128)
writer.add_uint32("moss_diarize.llm.ff_dim", 3072)
writer.add_uint32("moss_diarize.llm.vocab_size", 151936)
writer.add_bool("moss_diarize.llm.tied_embeddings", True)
writer.add_uint32("moss_diarize.audio_token_id", 151671)
writer.add_float32("moss_diarize.time_marker_interval", 5.0)
writer.add_float32("moss_diarize.audio_tokens_per_second", 12.5)
writer.add_uint32("moss_diarize.audio_merge_size", 4)
```

## Prompt Template

```
<|im_start|>system
请将音频转写为文本，每一段需以起始时间戳和说话人编号（[S01]、[S02]、[S03]…）开头，
正文为对应的语音内容，并在段末标注结束时间戳，以清晰标明该段语音范围。
[可选: 热词提示：word1, word2, word3]<|im_end|>
<|im_start|>user
<|audio_start|><|audio_pad|>×N_with_time_markers<|audio_end|><|im_end|>
<|im_start|>assistant
```

Time markers: every 5 seconds of audio (every 62.5/4 ≈ 15.6 merged frames), inject
digit tokens for the current timestamp into the audio_pad sequence. E.g., at 5.00s:
`...<|audio_pad|>×15 [5.00] <|audio_pad|>×15 [10.00] ...`

## Output Parsing

The model generates text like:
```
[0.48][S01]Welcome everyone[1.66]
[12.26][S02]The new pipeline is ready[13.81]
```

Parse into `crispasr_segment` with:
- `seg.t0 = start_timestamp * 100` (centiseconds)
- `seg.t1 = end_timestamp * 100`
- `seg.text = "Welcome everyone"`
- `seg.speaker_id = 1` (from [S01])

## Implementation Order

1. **Converter** — straightforward tensor rename, no fusing needed
2. **Runtime skeleton** — init + mel + encode (reuse crisp_audio or inline Whisper)
3. **VQAdaptor** — 3 lines of ggml (matmul+silu+matmul+layernorm)
4. **4x merge** — ggml_reshape
5. **Prompt builder** — tokenize system+user template, inject audio_pad tokens with time markers
6. **LLM decode** — copy from moss_transcribe, adjust dims
7. **Output parser** — regex `\[(\d+\.\d+)\]\[S(\d+)\](.*?)\[(\d+\.\d+)\]`
8. **CLI adapter** — CAP_TIMESTAMPS_NATIVE | CAP_DIARIZE | CAP_AUTO_DOWNLOAD
9. **Build wiring** — CMakeLists, backend factory, registry
10. **Kaggle kernel** — convert + quantize + upload to cstr/moss-transcribe-diarize-GGUF

## Verification

```bash
# Build
cmake --build build -j$(sysctl -n hw.ncpu)

# Test (once GGUF is on HF)
./build/bin/crispasr --backend moss-diarize \
  -m moss-transcribe-diarize-0.9b-q4_k.gguf \
  -f meeting_recording.wav \
  --hotwords "MOSS,CrispASR,OpenAI" \
  -osrt

# Expected SRT output:
# 1
# 00:00:00,480 --> 00:00:01,660
# [Speaker 1] Welcome everyone
#
# 2
# 00:00:12,260 --> 00:00:13,810
# [Speaker 2] The new pipeline is ready
```

## Notes

- The model is tiny (~500 MB Q4_K) — fits easily on any device
- The Whisper encoder is standard/stock — no custom ops needed
- The VQAdaptor with 4x merge is the only "new" architectural piece
- Diarization is text-based (no separate clustering/embedding head)
- Hotwords are prompt-injected, not architecturally special
- Time markers in the audio pad sequence are the clever trick for timestamps
- `crisp_audio` should handle the Whisper encoder if configured for 80 mel bins

## Implementation Status (2026-07-11)

**DONE** — backend merged to main, all features working:

- Diff harness: 4/4 stages PASS (mel, conv_stem, encoder, audio_embeds: cos=1.000)
- ASR: correct JFK transcript
- Diarization: [S01]/[S02] speaker labels working
- Timestamps: [start][end] timestamps working on multi-speaker audio
- 30s audio chunking: correctly handles audio > 30s
- Time markers: every 5s, bare digit tokens (from processor_config.json)
- Prompt format: single user turn (no system turn) — audio first, instruction after
- Hotwords, set_ask, language hint: all wired through C API + CLI

**Key learnings for future ports:**
1. ggml Conv1d input layout: MelsTime `mel[f*T+t]` IS the correct ggml ne=(T, IC) layout — do NOT transpose
2. HF model repo: `OpenMOSS-Team/MOSS-Transcribe-Diarize` (not `-0.9B`)
3. Prompt: NO system turn — instruction in user turn after `<|audio_end|>`
4. Time markers: `processor_config.json` overrides class default (5s not 2s)
5. Converter: watch for `.replace()` chain clobbering (e.g. `norm.` inside `layernorm.`)
6. The `torch_compilable_check` import needs stubbing for older transformers
