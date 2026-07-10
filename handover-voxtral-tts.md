# Voxtral-4B-TTS (#93) — Handover for synthesis pipeline implementation

## Status

**Done:**
- GGUF converter: `models/convert-voxtral-tts-to-gguf.py` — all 3 components + 20 voices + Tekken tokenizer
- GGUFs on HF: `cstr/voxtral-4b-tts-GGUF` — F16 (7.6 GB), Q8_0 (4.0 GB), Q4_K (2.2 GB)
- C++ skeleton: `src/voxtral_tts.h` + `src/voxtral_tts.cpp` — model loading, tokenizer, KV cache init all verified working
- CLI adapter: `examples/cli/crispasr_backend_voxtral_tts.cpp` — CAP_TTS, voice selection
- Full wiring: CMakeLists, backend factory, C ABI, registry, README
- **Model loads and tokenizes on the Q4_K** — verified output:
  ```
  voxtral_tts: LLM 26L d=3072 heads=32/8
  voxtral_tts: FM  3L d=3072 heads=32/8 rope_theta=10000
  voxtral_tts: Codec d=1024 semantic_cb=8192 acoustic_fsq=21×36
  voxtral_tts: 20 voices, 151000 tokens loaded
  ```

**Remaining:** Implement the 3 forward pass stages in `voxtral_tts_synthesize()` in `src/voxtral_tts.cpp`.

## Test command

```bash
# Build
CCACHE_DIR=/path/to/.ccache cmake -G Ninja -B build \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(sysctl -n hw.ncpu)

# Run (downloads Q4_K automatically)
./build/bin/crispasr --backend voxtral-tts -m auto --auto-download \
  --tts "Bonjour le monde." --tts-voice fr_female -o test.wav

# Or with explicit model path:
./build/bin/crispasr --backend voxtral-tts \
  -m /path/to/voxtral-4b-tts-q4_k.gguf \
  --tts "Hello world." --tts-voice neutral_female -o test.wav
```

## Architecture (3 components in one GGUF)

### Component A: LLM backbone (Ministral 3B)
- 26 layers, dim=3072, 32 heads / 8 KV heads, head_dim=128, FFN=9216
- SwiGLU FFN, RoPE θ=1e6, RMSNorm ε=1e-5
- Tied embeddings (token_embd.weight = output head)
- **Identical architecture to voxtral4b's LLM** — use `core_attn::kv_self_attn` + `core_ffn::swiglu`
- Pattern to copy: `voxtral4b_run_llm_kv()` at `src/voxtral4b.cpp:1397`
- But voxtral-tts needs hidden states (not logits) from the last token

### Component B: FM acoustic transformer (3 layers)
- dim=3072, 32/8 heads, head_dim=128, FFN=9216, RoPE θ=10000
- **Bidirectional** (no causal mask)
- Sequence length is always **3 tokens**: `[input_proj(x_t), time_emb, llm_proj(h)]`
- Same attention/FFN helpers, just no causal mask and different rope_theta
- Called once per AR-decoded frame, 7 times per frame (Euler ODE steps)

### Component C: Voxtral codec decoder
- dim=1024, 8/8 heads, head_dim=128, FFN=4096
- 4 conv layers (strides 1,2,2,2) interleaved with 4 transformer blocks (2 layers each)
- Uses **ALiBi** attention (not RoPE) — causal sliding window (starts at 16, not yet in codebase)
- QK-norm (RMSNorm ε=1e-6), layer_scale (learnable per-dim scaling)
- ConvTranspose1d for upsampling — use `core_hifigan::conv_transpose_1d` from `src/core/hifigan.h`
- Output: 240 PCM samples per frame at 24 kHz

## Inference pipeline (what to implement)

### Step 1: Build prompt embeddings
Voice embeddings are pre-summed `(T_voice, 3072)` tensors in the GGUF (tensor name `voice.<name>`).
Text tokens are embedded via `token_embd.weight` lookup.
Prompt = `[voice_frames] [text_tokens]`.

### Step 2: LLM AR decode
- **Prefill**: run all prompt embeddings through the LLM with KV cache
- **Decode loop** (one step per audio frame):
  1. Get hidden state `h` (dim=3072) from last LLM position
  2. Run FM to get semantic token + 36 acoustic FSQ values (Step 3)
  3. Embed the generated audio codes: sum 37 codebook embeddings from `audio_embd.weight`
     - Semantic: index = `sem_tok` (offset 0 in the 9088-entry table)
     - Acoustic: index = `semantic_cb_size + cb * acoustic_cb_size + acou_code` per codebook
  4. Feed the summed embedding as the next LLM input (single token, KV cached)
  5. Stop when semantic token ≥ `semantic_cb_size + 2` (EOA)

**Key reuse**: Copy the `voxtral4b_build_graph_llm_kv` / `voxtral4b_run_llm_kv` pattern.
The only difference: voxtral-tts doesn't have adaptive RMSNorm (ada_rms_norm), and we need
the hidden state (pre-output-projection) not logits. The hidden state is `cur` after the
final RMSNorm + output_norm mul, before the tied-embedding matmul.

### Step 3: FM ODE per frame (7 Euler steps, CFG α=1.2)

For each AR frame, given hidden state `h` from the LLM:

1. **Semantic token**: `logits = fm.semantic_output @ h` → argmax (greedy, always)
2. **Acoustic tokens** via Euler ODE:
   - `x_0 ~ N(0, I)` in R^36 (seed for reproducibility)
   - For step `i` in 0..6:
     - `t = i / 7`, `dt = 1/7`
     - Build 3-token input: `[fm.input_proj @ x_t, fm.time_proj @ sinusoidal(t), fm.llm_proj @ h]`
     - Run 3-layer bidirectional transformer (GQA + SwiGLU, RoPE θ=10k, no causal mask)
     - Extract velocity from first token: `v = fm.acoustic_output @ out[0]`
     - **CFG**: also run with `h_zeros` (all-zeros LLM hidden) → `v_uncond`
       - `v_cfg = 1.2 * v_cond + (1 - 1.2) * v_uncond`
     - `x_{t+1} = x_t + v_cfg * dt`
   - Quantize: `code = round(clamp(x, -1, 1) * 10 + 10)` → integers 0..20

**Sinusoidal time embedding**: `sin_emb[i] = sin(t * exp(-i * log(10000) / (d/2)))` for i < d/2,
`cos` for i ≥ d/2. Then `fm.time_proj` projects d→d. (The `fm.time_proj.weight` is a single
linear layer, no bias, no SiLU — simpler than f5_tts's 2-layer MLP.)

### Step 4: Codec decode

Input: per-frame `(semantic_code, acoustic_codes[36])`.

1. **Latent construction**: semantic VQ lookup in `codec.semantic_cb.weight` (8192, 256) → 256-d.
   Acoustic FSQ rescale: `(code * 2 / 20) - 1` → 36-d. Concatenate → 292-d per frame.
2. **Conv0** (k=3, s=1): causal Conv1d 292→1024. Weight: `codec.dec.conv.0.weight` (1024, 292, 3).
   Causal padding: pad left by `k-1=2`.
3. **4× [Transformer + Conv]**:
   - Transformer (2 layers): ALiBi attention + QK-norm + layer_scale + SwiGLU FFN
   - Conv (stride 2): ConvTranspose1d 1024→1024. Use `core_hifigan::conv_transpose_1d`.
4. **Output conv** (k=7): causal Conv1d 1024→240. Weight: `codec.output.weight` (240, 1024, 7).
5. **Reshape**: (T_upsampled, 240) → flatten to PCM samples.

**ALiBi implementation** (new — not in the codebase yet):
ALiBi adds a position-dependent bias to attention scores before softmax.
For head `h` with slope `m_h = 2^(-8h/H)` (H = total heads):
  `bias[q][k] = -m_h * |q - k|` (causal: only q ≥ k)
With sliding window W: mask positions where `q - k ≥ W` to `-inf`.

For the codec decoder this is simpler than RoPE — it's just a bias tensor added to the
attention scores. The sliding window starts at 16 and halves upon each downsampling stage
in the encoder (but the decoder doubles it: 2, 4, 8, 16 from first to last block).

**Practical shortcut**: for short utterances (< 1000 frames after upsampling), the sliding
window never activates and ALiBi is just the distance-based bias. Implement that first;
add the window mask as a follow-up if needed.

## Key files to read

| File | What to learn |
|------|---------------|
| `src/voxtral4b.cpp:1397-1449` | LLM KV-cached forward (copy this pattern) |
| `src/voxtral4b.cpp:1530-1619` | Graph builder with `core_attn::kv_self_attn` |
| `src/core/attention.h:665-900` | `kv_self_attn` — handles KV write, RoPE, flash_attn, GQA |
| `src/core/ffn.h:28-50` | `core_ffn::swiglu` — SwiGLU without biases |
| `src/core/hifigan.h:108-130` | `conv_transpose_1d` — ConvTranspose1d with crop |
| `src/f5_tts.cpp:1503-1590` | Euler ODE solver with CFG (reference pattern) |
| `src/chatterbox.cpp` | Two-component TTS pattern (AR + vocoder) |

## GGUF tensor names

### LLM (26 layers)
```
token_embd.weight              (131072, 3072) — text embeddings (tied to output)
audio_embd.weight              (9088, 3072)   — audio codebook embeddings
output_norm.weight             (3072,)
blk.{0-25}.attn_norm.weight    (3072,)
blk.{0-25}.attn_q.weight       (3072, 3072)   — no bias
blk.{0-25}.attn_k.weight       (1024, 3072)   — GQA: 8 KV heads
blk.{0-25}.attn_v.weight       (1024, 3072)
blk.{0-25}.attn_output.weight  (3072, 3072)
blk.{0-25}.ffn_norm.weight     (3072,)
blk.{0-25}.ffn_gate.weight     (9216, 3072)
blk.{0-25}.ffn_up.weight       (9216, 3072)
blk.{0-25}.ffn_down.weight     (3072, 9216)
```

### FM transformer (3 layers)
```
fm.input_proj.weight           (3072, 36)    — acoustic state → dim
fm.llm_proj.weight             (3072, 3072)  — LLM hidden → dim
fm.time_proj.weight            (3072, 3072)  — time embedding → dim
fm.semantic_output.weight      (8320, 3072)  — → semantic logits
fm.acoustic_output.weight      (36, 3072)    — → acoustic velocity
fm.norm.weight                 (3072,)
fm.blk.{0-2}.attn_norm/q/k/v/output, ffn_norm/gate/up/down — same naming as LLM
```

### Codec decoder
```
codec.semantic_cb.weight       (8192, 256)   — VQ codebook
codec.dec.conv.{0-3}.weight    — Conv1d / ConvTranspose1d (fused weight-norm)
codec.dec.tfm.{0-3}.blk.{0-1}.attn_norm/q/k/v/o.weight — transformer layers
codec.dec.tfm.{0-3}.blk.{0-1}.q_norm/k_norm.weight     — QK norm
codec.dec.tfm.{0-3}.blk.{0-1}.attn_scale/ffn_scale     — layer_scale
codec.dec.tfm.{0-3}.blk.{0-1}.ffn_norm/gate/up/down.weight
codec.output.weight            (240, 1024, 7)
voice.{name}                   (T_voice, 3072) — 20 preset voices
```

## Verification

1. **Smoke test**: `./build/bin/crispasr --backend voxtral-tts -m voxtral-4b-tts-q4_k.gguf --tts "Hello." --tts-voice neutral_female -o test.wav`
2. **ASR roundtrip**: `./build/bin/crispasr -m whisper-large-v3-turbo.gguf -f test.wav` → should recognize the input text
3. **Format before commit**: `./tools/format.sh --fix src/voxtral_tts.cpp`
4. **French quality test**: `--tts "La sémiologie psychiatrique est une discipline fondamentale." --tts-voice fr_female`

## Notes

- The model is CC-BY-NC-4.0 licensed
- 24 kHz mono output (set in `tts_sample_rate()`)
- The Q4_K (2.2 GB) fits comfortably in 16 GB RAM
- No codec encoder in the checkpoint → preset voices only (no raw-audio voice cloning)
- The FM transformer runs 7 * 2 = 14 forward passes per frame (7 steps × {cond + uncond} for CFG)
- Each frame = 80 ms of audio (12.5 Hz frame rate)
