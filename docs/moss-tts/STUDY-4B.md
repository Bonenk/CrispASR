# MOSS-TTS-Local-Transformer-v1.5 (4B) — port STUDY (#249 second deliverable)

Model: `OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5` (`model_type: moss_tts_local`,
`MossTTSLocalModel`). **No C++ reference exists** — pwilkin/openmoss ported only the
8B MossTTSDelay. This is a from-scratch port; the blueprint below is decoded
line-by-line from the HF `modeling_moss_tts.py` / `gpt2_decoder.py` / `qwen3_decoder.py`
/ `config.json` (HARD RULE #1).

## How it differs from the 8B (already shipped as `moss-tts`)

| | 8B `moss-tts` (MossTTSDelay) | 4B `moss-tts-local` (MossTTSLocal) |
|---|---|---|
| Backbone | Qwen3-8B: 4096d, 36L, vocab 155648, **untied** lm_head | Qwen3-4B: **2560d**, 36L, 32Q/8KV, head_dim 128, SwiGLU 9728, vocab **151936**, RoPE NEOX 1e6, rms 1e-6, **TIED** embeddings |
| Audio codebooks | 32, under a staggered **delay pattern**, 32 parallel heads on the backbone hidden | **12**, generated **depth-first by a 1-layer local transformer** (RQ-Transformer style) — NO delay |
| Codec | MOSS-Audio-Tokenizer (24 kHz, 32 cb) | **MOSS-Audio-Tokenizer-v2 (48 kHz, 12 cb)** — separate study (Phase 3) |
| Text/stop | full-vocab text head + im_end | **binary local text head** (assistant-slot vs audio-end) each frame |

The delay state machine and 32-head extraction of the 8B are **gone**. Replaced by a
two-level hierarchy: the Qwen3 backbone emits one per-frame hidden, then a small local
transformer autoregressively emits the 12 codebooks *within* that frame.

## Config (config.json)

- `n_vq = 12`, `audio_vocab_size = 1024`, `audio_codebook_sizes = [1024]*12`,
  `audio_pad_token_id = audio_pad_code = 1024`.
- `sampling_rate = 48000`, `audio_tokenizer = OpenMOSS-Team/MOSS-Audio-Tokenizer-v2`.
- `local_transformer_layers = 1`, `local_text_head_mode = "binary"`,
  `use_static_local_kv_cache = True`.
- Tokens: pad 151643, im_start 151644, im_end 151645, audio_start 151669,
  audio_end 151670, user_slot 151654, assistant_slot 151656 (== assistant_gen_slot).
- `qwen3_config` (== `language_config`): hidden 2560, layers 36, heads 32, kv 8,
  head_dim 128, intermediate 9728, vocab 151936, rope_theta 1e6, rms_eps 1e-6,
  `tie_word_embeddings=True`, silu.
- `gpt2_config` (local transformer): n_embd 2560, n_head 32 (head_dim **80**),
  n_inner 9728, n_layer 1 (overridden by `local_transformer_layers`),
  `position_embedding_type="rope"`, `rope_base=1e6`, `activation_function="silu"`,
  `layer_norm_epsilon=1e-6`, `scale_attn_weights=True`, vocab 151936. At build time
  `n_positions = n_ctx = n_vq + 1 = 13`.

## Modules (modeling_moss_tts.py)

- `transformer = MossQwen3Model(qwen3_config)` — the backbone (reuse CrispASR's
  in-house Qwen3 KV path, as the 8B did; just smaller dims + tied embeddings).
- `local_transformer = MossTTSNanoGPT2Model(local_gpt2_config)` with `wte = Identity()`
  (consumes `inputs_embeds` directly, no token embedding). Static KV cache of length
  `n_vq+1 = 13`.
- `audio_embeddings[0..11]`: `Embedding(1024, 2560)` per codebook.
- `text_lm_head = Linear(2560, 151936, bias=False)` — **tied to `transformer.embed_tokens.weight`**.
- `audio_lm_heads[0..11] = Linear(2560, 1024, bias=False)` — each **tied to
  `audio_embeddings[k].weight`**.
- `local_text_lm_head = Linear(2560, 2, bias=False)` — the **binary** continue/stop
  head, initialized from `text_lm_head.weight[[assistant_slot, audio_end]]`. Candidate
  ids order = `[audio_assistant_slot_token_id, audio_end_token_id]`.

### Local GPT2 block (MossTTSNanoGPT2, gpt2_decoder.py)

Pre-norm GPT2-style, 1 layer:
- `ln_1` LayerNorm(2560, eps 1e-6, **with bias**) → attention → residual.
- `ln_2` LayerNorm → MLP → residual. Final `ln_f`.
- Attention: `c_attn = Linear(2560, 3*2560, bias=True)` fused QKV, `c_proj =
  Linear(2560, 2560, bias=True)`. 32 heads, head_dim 80. **RoPE NEOX (`rotate_half`),
  base 1e6**. Causal. `scale = 1/sqrt(head_dim)`.
- MLP: `fc_in = Linear(2560, 9728)` → **SiLU** → `fc_out = Linear(9728, 2560)` (plain
  2-matrix, NOT SwiGLU; both with bias).

## Input embedding (`_build_inputs_embeds`) — same shape family as the 8B

`input_ids` is `[batch, seq, 1 + n_vq]` (13 channels). Embedding =
`embed_tokens(col0) + Σ_k audio_embeddings[k](col_{k+1})`, with pad channels
(`== audio_pad_token_id`) masked to zero. Identical structure to the 8B's summed
input embedding → CrispASR's `moss_tts_compute_input_embeddings` generalizes (n_vq 12).

## Generate loop (the core — `generate`, depth-first)

Per frame (backbone KV-cached; local KV static, reset each frame):
1. `global_hidden = transformer(inputs_embeds=row).last_hidden_state[:, -1]`.
2. `local_prefix = local_transformer(inputs_embeds=global_hidden.unsqueeze(1))` →
   `local_hidden = local_prefix[:, -1]`. (Local position 0 input = the global hidden.)
3. **Binary text step**: `next_text = candidate_ids[argmax/sample(local_text_lm_head(
   local_hidden))]`. If `next_text == audio_end` → mark finished/stop; if
   `assistant_slot` → continue.
4. **Depth-first 12 codebooks**: for k in 0..11:
   - `code_k = sample(audio_lm_heads[k](local_hidden))` (per-channel repetition penalty
     over the history `generated[:, :, k]`).
   - if k < 11: `local_hidden = local_transformer(
     inputs_embeds = audio_embeddings[k](code_k), past_kv = local_prefix_kv)[:, -1]`
     (advances the local KV cache one step per codebook).
5. `next_row = [assistant_slot, code_0..code_11]` (13 channels), append to the
   backbone sequence; backbone KV grows by 1.

Extraction: no un-delay. The generated frames stacked = `(n_vq, T)` code grid directly
(slice from the last `audio_start`). `_find_last_equal(audio_start)` bounds the segment.

Sampling knobs: `text_temperature/top_p/top_k` for the binary head;
`audio_temperature/top_p/top_k/repetition_penalty` for the codebooks (defaults fall back
to `temperature/top_p/top_k/repetition_penalty`). Greedy = `do_sample=False` → argmax.

## Port plan (phases; heavy steps on Kaggle — Mac loadavg high, 4B+codec won't fit)

- **P1 converter** `models/convert-moss-tts-local-to-gguf.py`: arch `moss-tts-local`
  (llm.* Qwen3 backbone [tied → emit lm_head from embed], local.* GPT2 block,
  moss.audio_embed.k, moss.audio_head.k [tied], moss.text_head [tied], moss.local_text_head)
  + companion codec `moss-tts-local-codec` (Tokenizer-v2). Reuse the 8B converter's
  Qwen3 name-map; add the local-transformer + 12-vs-32 codebook handling.
- **P2 runtime** `src/moss_tts_local.{h,cpp}`: reuse the backbone KV path
  (`core_attn::kv_self_attn`, like `moss_tts.cpp`); NEW 1-layer local transformer
  (LayerNorm+bias, fused-QKV, RoPE NEOX 1e6, SiLU MLP) with a 13-step static KV; the
  depth-first generate loop + binary stop head. n_vq 12.
- **P3 codec** MOSS-Audio-Tokenizer-v2 (48 kHz, 12 cb) — STUDY separately; likely a
  variant of the v1 transformer RVQ codec (`moss_tts_codec.cpp`) at 48 kHz / different
  hop / 12 codebooks. Confirm stereo vs mono (`sampling_rate` is set; check the codec
  for channel count).
- **P4 integration** (12-point, per contributing.md): CLI adapter, factory/detect,
  registry, quantize rules, c_api inline synthesize, bindings, README/docs, ref backend.
- **P5 validate on Kaggle**: decoded round-trip → ASR (HARD RULE #3 = the only gate);
  greedy code-parity is diagnostic only (quantized AR — see the 8B's logit-rank note in
  [[tts-port-parity-via-logit-rank]] / PLAN.md).

## Open questions to resolve during implementation
- Codec-v2 exact architecture + whether output is **stereo** (config `sampling_rate`
  48000; `MOSS-Audio-Tokenizer-v2` internals not yet read).
- Local transformer block order details (pre/post-norm) + whether `c_attn`/`c_proj`
  biases are used at inference (they exist; confirm non-zero).
- Backbone: tied embeddings means the converter emits `lm_head` from `embed_tokens`
  (the 8B was untied). QK-norm present? (Qwen3 has QK-norm — confirm in qwen3_decoder.py.)
