# CrispASR v0.8.41

This release adds Index-Echo 2B and 9B speech translation, fixes Dia's full-dialogue speech generation, improves Nemotron realtime CPU inference, and adds matching Windows CUDA 12.6 packages. Browser applications gain asynchronous model loading and transcription on the proxy compute thread. Native CPU thread requests now take effect in Nemotron, Paraformer and Dia.

## Index-Echo 2B and 9B

The public `cstr/index-echo-9b-GGUF` repository contains the accepted F16 encoder/decoder pair, approximately 17.9 GiB in total. Independent source checks pass on CPU and real CUDA hardware: numerical stages, cached decoding, five complete translation cases with timestamps and context, and three synthesized-speech round trips with zero word errors. The 2B Q8 default remains available and passes its protected regression checks. Experimental 9B quantizations are not included.

See [the acceptance receipt](https://github.com/CrispStrobe/CrispASR/blob/v0.8.41/docs/index-echo-9b-acceptance-2026-10-02.json) for exact source/model pins and retained failed diagnostic controls.

Both checkpoints produce bilingual subtitles with native timestamps, using the released Chinese-to-English/Japanese/Spanish recipe. They share the `index-echo` backend and work through the CLI and the generic session API used by the language bindings and server. GGUF metadata detection is tested through the actual C ABI with anonymous filenames. The contributing checklist and shipped shared-library audit pass; see [the wiring audit](https://github.com/CrispStrobe/CrispASR/blob/v0.8.41/docs/index-echo-wiring-audit-2026-10-02.md).

Select the smaller default or explicitly request the larger model:

```sh
crispasr -m auto --backend index-echo --auto-download -f audio.wav --target-lang en -osrt
crispasr -m index-echo-9b-f16.gguf --auto-download -f audio.wav --target-lang en -osrt
```

Auto-download resolves the matching decoder and Silero VAD companion. Keep the tower and decoder together when downloading manually. The 2B Q8 pair is approximately 2.589 GiB; the 9B F16 pair is 17.914 GiB, before runtime memory. Use this release or the accepted newer source for 9B; older releases cannot load its projection connector.

| Model pair | Accepted precision | Download size | Selection |
|---|---|---|---|
| Index-Echo 2B | F16 and Q8 | Q8: 2.589 GiB; F16: approximately 4.85 GiB | `-m auto --backend index-echo` chooses Q8 |
| Index-Echo 9B | F16 | 17.914 GiB | `-m index-echo-9b-f16.gguf` |

The source checkpoints and published model pairs are Apache-2.0. The 2B residual connector and 9B projection connector are selected from model metadata. The learned encoder, connector and hybrid Qwen3.5 decoder execute through ggml; decoding uses persistent attention/recurrent state, Flash Attention where enabled, and fused GDN. Multi-GPU graph reuse remains disabled, and these T4 measurements do not establish CUDA graph capture.

The port preserves the released frontend padding, Silero waveform context and sample timing, window extraction rounding, prompt and prior-window context. The shared Qwen3.5 query/key normalization uses the source's additive epsilon. Conversion excludes absent prediction-layer weights. An additional VAD fix keeps weights on the CPU backend used by its scheduler when GPU inference is requested; the accepted 9B corpus verifies CPU and GPU-requested VAD results are bitwise equal.

On two T4 GPUs, resident warm 9B inference is about 1.33× faster for JFK and 1.36× faster for the Chinese test than the original Python blueprint, with exact tested output. This compares native F16 with source BF16 and their recorded device placements; it is not a same-precision algorithm-only benchmark. Generation remains the dominant cost, and these measurements are slightly slower than realtime. See [the profile receipt](https://github.com/CrispStrobe/CrispASR/blob/v0.8.41/docs/index-echo-9b-profile-2026-10-02.json).

Pinned 2B coverage joins the regular regression matrix; 9B has a separate weekly/manual large CPU workflow with complete file cases and real Piper round trips. Physical CPU and CUDA runtime acceptance is complete. Build checks cover WASM, but the 9B pair exceeds ordinary WASM32 address space; physical 9B Metal/Vulkan runtime acceptance is not claimed. Plain and selective 9B Q8 experiments failed decoded-output gates and remain excluded.

See the [2B acceptance record](https://github.com/CrispStrobe/CrispASR/blob/v0.8.41/docs/index-echo-acceptance-2026-10-01.json), [CPU comparison](https://github.com/CrispStrobe/CrispASR/blob/v0.8.41/docs/index-echo-cpu-2026-10-01.json), and [physical CUDA checks](https://github.com/CrispStrobe/CrispASR/blob/v0.8.41/docs/index-echo-cuda-2026-10-01.json) for its independent source comparisons and precision-specific bounds. Experimental Q4 is excluded from the accepted 2B release pairs.

## Dia and Nemotron realtime

Dia now generates complete speech rather than stopping at the previous hidden 200-step CPU limit. Cross-attention position encoding, nucleus sampling and delayed audio-codebook starts are corrected; explicit generation limits reach the decoder, and complete dialogues preserve speaker context. Independent F16 source comparisons and F16/Q8 speech round trips pass. Physical Metal runtime acceptance remains pending.

The nucleus sampler retains the token crossing the probability threshold, preventing an empty candidate set. Delayed BOS masking follows the official decoder step, and explicit limits flush delayed EOS correctly. Limits too short for the codebook delay are rejected. The CLI/server synthesis planner keeps the whole dialogue together instead of restarting generation and speaker context for each short sentence. The independent official-source comparison covers 127 steps with minimum cosine above 0.999999; tested default F16/Q8 CPU speech has zero word errors. Q8 audio is identical for each tested prompt at 1/4/8 threads.

Nemotron uses aligned incremental frontend windows on CPU, with exact token, text and confidence checks at 1/4/8 threads. CUDA retains the original full frontend after the incremental GPU confidence check failed; real T4 regression checks pass F16/Q8/Q4. Native Nemotron realtime sessions support `CRISPASR_NEMOTRON_MAX_TURN_SECONDS=1..300` (default 30), and WebSocket header/pong handling is corrected. CPU thread requests are forwarded consistently through the CLI and C ABI.

`CRISPASR_NEMOTRON_STREAM_CHUNKS_PER_STEP` controls how many native chunks are gathered per realtime update; the defaults are four on CPU and one on GPU. CPU processing recomputes aligned frontend windows and pre-encodes only the newest audio instead of repeating the full history. The GPU route keeps the validated full frontend. Live WebSocket tests cover ordinary streaming, VAD, and a configured 45-second turn. Header matching accepts any capitalization of `Sec-WebSocket-Key`, and ping frames receive pong responses. No unmeasured realtime speedup is claimed.

See [Dia validation](https://github.com/CrispStrobe/CrispASR/blob/v0.8.41/docs/dia-full-generation-2026-10-02.json) and [Nemotron validation](https://github.com/CrispStrobe/CrispASR/blob/v0.8.41/docs/nemotron-realtime-2026-10-02.json) for source pins, output comparisons and retained failed controls.

## CPU thread handling and model-load cleanup

Nemotron, Paraformer and Dia now apply requested thread counts to their actual model-owned CPU backend. Previously those runtimes could retain four workers despite a positive request. Dia's runtime thread setter updates that backend too; nonpositive native requests retain the four-worker default. CLI and C ABI callers benefit from the runtime fix through their existing APIs.

Worker-count tests execute real CPU graphs in linked and module builds, with failing pre-fix controls. Live Nemotron and Paraformer runs preserve repeated full-JFK output at 1/4/8 threads. Paraformer also rejects null/empty model paths and releases backend handles after failed loads. See [the thread-count validation receipt](https://github.com/CrispStrobe/CrispASR/blob/v0.8.41/docs/cpu-threads-2026-10-02.json); Dia's later full-speech acceptance is recorded separately above.

## Windows CUDA packages

New `-cuda126` CLI and shared-library packages use CUDA Toolkit 12.6.3 with matching runtime DLLs. Use their `cuda126-runtime.zip`; the existing CUDA 12.8 package uses `cuda-runtime.zip`. Avoid mixing toolkit/runtime versions. CUDA 12.8 and CUDA 13 packages remain available, and native A100/H100 targets plus general SM90 PTX are restored.

CLI and library archives are checked against the same runtime hashes. Windows tests also verify reported toolkit versions and driverless C ABI loading with an AVX2 CPU floor; the packaging-input DLL version and staged archive hashes are recorded in the proof receipt. Mixed 12.8/12.6 archives are rejected. See [package verification](https://github.com/CrispStrobe/CrispASR/blob/v0.8.41/docs/cuda126-packaging-2026-10-02.json) for architecture coverage and test scope; no GTX 16xx MMQ performance claim is made.

CUDA 12.8 remains the option with native Blackwell support. The CUDA 12.6 packages provide a matching older-toolkit choice rather than asking users to swap DLLs inside a 12.8 archive. Hosted Windows checks compile the supported architecture lists and exercise packaged imports and CPU fallback without an NVIDIA driver; they do not establish physical GTX/MX GPU execution or performance.

## Browser and scheduler verification

Callback-based browser model-open and transcription run through the proxy compute thread. Calls reject when the proxy is unavailable or unready. Cross-browser worker tests cover Chromium, Firefox and WebKit; existing synchronous/native APIs remain available.

The new callback entry points are `asrOpenAsync`, `ttsOpenExplicitAsync`, and `asrTranscribeAsync`, alongside existing asynchronous synthesis. Native compute runs on the proxy thread, and callbacks deliver results on the JavaScript servicer. Dedicated-worker tests cover two/four-thread transcription, cancellation, reload with downloads disabled, and Kokoro synthesis followed by Moonshine transcription in all three browsers.

Optional CMake settings `CRISPASR_WASM_INITIAL_MEMORY` and `CRISPASR_WASM_THREAD_POOL` allow heap and worker-pool experiments. Five build variants pass, including 128 MiB single-thread/proxy variants. Existing defaults are preserved: browser A/B checks show that threading helps Phonon but can hurt Moonshine, so no universal threading or reduced-heap performance recommendation is made.

The cached cross-backend scheduler fix already shipped in v0.8.40 is retained. A permanent Vulkan regression now checks repeated computations, source restoration and recycled graph metadata, with failing controls for both replay and disposal regressions.

See [the scheduler regression receipt](https://github.com/CrispStrobe/CrispASR/blob/v0.8.41/docs/scheduler-pr480-review-2026-10-02.json). This is executed Vulkan/lavapipe correctness coverage; it is not a physical GPU speed measurement.

## Build and regression reliability

Windows Vulkan SDK downloads now retry transport failures and discard partial installers, protecting CI and release builds from transient download errors. Generated capability headers retain their expected file mode, and the WASM size budget records measured Index-Echo decoder growth.

The Windows Piper live proof sets the existing seed option to 42 and verifies repeated synthesis before applying the original ASR readback criteria. This removes random sampling from that regression check without changing Piper's production defaults or weakening its speech gates.

Independent model regressions can run concurrently when they only read pinned artifacts; producer/publication serialization remains protected. Public 9B publication checks exact accepted file hashes and clean repository history. The diff harness compares captures along the correct contiguous GGUF axis, inspects both tower and decoder precision, and retains magnitude, cached-token and decoded-output gates.

Main integration passes 13 CI jobs, 10 lint jobs, five WASM build variants, and all 11 selected-regression jobs, covering seven live backends. The separate complete 9B ARM workflow and physical two-T4 corpus pass. Receipts distinguish original-source precision, failed diagnostic controls, compile coverage and actual runtime acceptance so numerical agreement alone cannot hide a decoded-output failure.
