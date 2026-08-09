# CrispASR v0.8.26

Twelve reported issues, and three of them were the same bug wearing different
clothes: a runtime that had quietly drifted from the reference the diff harness
checks it against.

## Fixed — CosyVoice3 voice cloning used the wrong speaker embedding (#334)

Every `--voice ref.wav` clone was conditioned on a speaker embedding at cosine
**0.737** against upstream's `campplus.onnx`. Not a crash — a wrong-but-plausible
192-d vector, so it read as "cloning isn't very good" rather than as a bug.

CAMPPlus ends `transit3.linear(Conv1d) → out_nonlinear(BatchNorm+ReLU) →
StatsPool`, and CosyVoice3's ONNX export **folds that BatchNorm into the
convolution** — the graph carries a bare ReLU and the conv gains a fused bias.
We never applied a conv bias there, so the fold was dropped. Now 0.999997.

The baked voice bank was never affected (its embeddings come from the ONNX model
in Python), which is exactly why this survived: only the runtime WAV path was
wrong.

## Fixed — CosyVoice3 decode could produce no audio at all (#334)

The talker had no minimum length. Upstream forbids a stop token for the first
`2 × (target text tokens)` steps; without that floor one unlucky sample at step 0
ended the decode — `AR decode produced 0 tokens`, no output. Short of total
failure it let the model rush a line into far too few 40 ms frames, which is
heard as sped-up, pitched-up speech.

## New — CosyVoice3 `--ref-text` is now optional (#334)

A transcript that doesn't match the reference clip is the single most damaging
input to this backend, and requiring one just moved that job to the caller.
The reference is now auto-transcribed (`--ref-asr` to choose the engine) and
cached beside the clip. On a 17.7 s reference, a plausible one-sentence guess
lost the requested line entirely; auto-transcribed, it came out in full.

If you do pass `--ref-text`, the runtime now warns when its length can't match
the audio.

## New — CosyVoice3 RL talker (#334)

Upstream ships a second, reinforcement-learning-tuned talker (`llm.rl.pt`).
Published as `cosyvoice3-llm-rl-{f16,q4_k}.gguf`; `--backend cosyvoice3-tts-rl`
swaps just that file — flow, HiFT, CAMPPlus, tokenizer and voice bank are shared.

## Fixed — Voxtral TTS could index the embedding table out of bounds (#338)

Reported by **@MassimoDePietro** with the diagnosis and a tested fix. A Tekken
vocabulary blob may serialize more pieces than a checkpoint activates —
Voxtral-4B-TTS has 131072 rows and 1000 specials, so only the first 130072 BPE
pieces are live. We admitted the whole blob to the encoder, so a text whose merge
path reached the inert tail produced a token id past the table: an assertion on
CPU, non-finite output and runaway generation on CUDA. Input-dependent, which is
why it never showed up in a smoke test.

The bound now lives in one header shared with the diff harness's reference
tokenizer, which had always enforced it. `voxtral4b` had the same defect and is
fixed too.

## Fixed — qwen3-tts could emit 300 s of audio for one sentence (#337)

Reported by **@adam-schneider-dev** with an exceptionally thorough matrix ruling
out quantization, model size, flash-attention, HIP graph capture and the voice
reference. `max_frames` was the KV cache ceiling, so **any** input was allowed
4096 frames — and a degenerate trajectory ran to it, returning a valid WAV with
exit code 0. It is now bounded by the input text, and hitting either bound says
plainly that the output is a runaway.

Worth stating because the report reasonably suspected otherwise: **the GPU is not
miscomputing.** Measured CPU-vs-GPU under greedy, the talker logits agree to
cos 0.99992 at frame 0 — better than we treat as normal — and neither
`--no-flash-attn` nor F32 KV moves it. Ordinary backend arithmetic, amplified by
the autoregressive loop until the argmax flips.

Also: `--temperature` now actually reaches the talker (it only ever reached the
code predictor), and `CRISPASR_QWEN3_TTS_GREEDY=1` forces argmax so two backends
can be compared at all.

## Fixed — madlad400 quantization and T5 parity (#333)

F16 hits cosine **1.000000** on all 14 stages against the PyTorch blueprint
(Q8_0 0.9999, Q4_K 0.9933, same first token at every precision). The F16 and Q8_0
artifacts the card promised are now actually published. A T5-specific quantizer
rule was measured, lost, and defaults off.

## Fixed — Kokoro dropped punctuation, and the contextual G2P was switched off (#316)

Punctuation is in Kokoro's vocabulary and we were discarding it — in German,
French and Spanish too. The contextual G2P rules had shipped disabled for two
releases. Acronyms are now read out the way misaki does.

## Fixed — the target-language knob was dead on three surfaces (#329, #13273)

`-tl` / `--target-lang` was silently discarded by cosyvoice3 and omnivoice.
omnivoice now also guesses the language when nobody supplies one.

## Fixed — silence transcribed as fabricated speech (cohere)

Digital silence decoded into invented words. Also: `-l` is now validated against
the model's own language list, and LID probing encodes once instead of once per
language.

## Fixed — the arch→backend table had drifted by 113 strings (#335)

Two copies, one table now.

## Also in this release

- `--strict-pipeline` could not fire on a VAD download failure (#311)
- Rust and Dart diarize ABI mirrors were 24 bytes short; FoxNose exposed (#332)
- `session_output_sample_rate` + channel getters, and ABI-mirror rules written
  into `docs/contributing.md` (#332)
- Segment hygiene (§W2/W5/W6) and `--sensitivity` presets, with the C-ABI given
  its own arm rather than a mirror that could drift
- The n-gram loop guard was a no-op on CJK; firered-asr had no loop collapse wired
- The default Piper voice was a private repo, and nothing would have caught it (#331)
- `--help` advertised 2 of 6 `--vad-model` keywords
- Metal: ggml pin bumped to restore the batch-1 im2col occupancy win the v0.17
  sync had dropped
- `resample_polyphase` truncated its filter on every downsample — real, small,
  and the existing test's tolerance was wider than the defect

## Known gaps

- The qwen3-tts runaway is bounded and reported, but still exits 0. Making it a
  non-zero exit changes the contract for callers relying on truncated output.
- CosyVoice3 auto-transcription is CLI/server only; the session C-ABI still
  requires an explicit `--ref-text` for a WAV, as f5-tts always has.
- CosyVoice3's 10 s prompt-mel cap is ours, not upstream's. A 17.7 s reference
  round-trips fine, so it is an untested divergence rather than a known defect.
