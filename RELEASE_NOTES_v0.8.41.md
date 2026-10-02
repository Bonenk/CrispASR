# CrispASR v0.8.41

This release adds independently validated Index-Echo 9B F16 translation, fixes Dia's full-dialogue speech generation, improves Nemotron realtime CPU inference, and adds matching Windows CUDA 12.6 packages. Browser applications gain asynchronous model loading and transcription on the proxy compute thread.

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

On two T4 GPUs, resident warm 9B inference is about 1.33× faster for JFK and 1.36× faster for the Chinese test than the original Python blueprint, with exact tested output. This compares native F16 with source BF16 and their recorded device placements; it is not a same-precision algorithm-only benchmark. Generation remains the dominant cost, and these measurements are slightly slower than realtime. See [the profile receipt](https://github.com/CrispStrobe/CrispASR/blob/v0.8.41/docs/index-echo-9b-profile-2026-10-02.json).

Pinned 2B coverage joins the regular regression matrix; 9B has a separate weekly/manual large CPU workflow with complete file cases and real Piper round trips. Physical CPU and CUDA runtime acceptance is complete. Build checks cover WASM, but the 9B pair exceeds ordinary WASM32 address space; physical 9B Metal/Vulkan runtime acceptance is not claimed. Plain and selective 9B Q8 experiments failed decoded-output gates and remain excluded.

## Dia and Nemotron realtime

Dia now generates complete speech rather than stopping at the previous hidden 200-step CPU limit. Cross-attention position encoding, nucleus sampling and delayed audio-codebook starts are corrected; explicit generation limits reach the decoder, and complete dialogues preserve speaker context. Independent F16 source comparisons and F16/Q8 speech round trips pass. Physical Metal runtime acceptance remains pending.

Nemotron uses aligned incremental frontend windows on CPU, with exact token, text and confidence checks at 1/4/8 threads. CUDA retains the original full frontend after the incremental GPU confidence check failed; real T4 regression checks pass F16/Q8/Q4. Native Nemotron realtime sessions support `CRISPASR_NEMOTRON_MAX_TURN_SECONDS=1..300` (default 30), and WebSocket header/pong handling is corrected. CPU thread requests are forwarded consistently through the CLI and C ABI.

See [Dia validation](https://github.com/CrispStrobe/CrispASR/blob/v0.8.41/docs/dia-full-generation-2026-10-02.json) [Nemotron validation](https://github.com/CrispStrobe/CrispASR/blob/v0.8.41/docs/nemotron-realtime-2026-10-02.json), and [CPU thread forwarding](https://github.com/CrispStrobe/CrispASR/blob/v0.8.41/docs/cpu-threads-2026-10-02.json) for scoped evidence.

## Windows CUDA packages

New `-cuda126` CLI and shared-library packages use CUDA Toolkit 12.6.3 with matching runtime DLLs. Use their `cuda126-runtime.zip`; the existing CUDA 12.8 package uses `cuda-runtime.zip`. Avoid mixing toolkit/runtime versions. CUDA 12.8 and CUDA 13 packages remain available, and native A100/H100 targets plus general SM90 PTX are restored.

CLI and library archives are checked against the same runtime hashes. Windows tests also verify reported toolkit versions, the packaged runtime API version, and driverless C ABI loading with an AVX2 CPU floor. Mixed 12.8/12.6 archives are rejected. See [package verification](https://github.com/CrispStrobe/CrispASR/blob/v0.8.41/docs/cuda126-packaging-2026-10-02.json) for architecture coverage and test scope; no GTX 16xx MMQ performance claim is made.

## Browser and scheduler verification

Callback-based browser model-open and transcription run through the proxy compute thread. Calls reject when the proxy is unavailable or unready. Cross-browser worker tests cover Chromium, Firefox and WebKit; existing synchronous/native APIs remain available.

The cached cross-backend scheduler fix already shipped in v0.8.40 is retained. A permanent Vulkan regression now checks repeated computations, source restoration and recycled graph metadata, with failing controls for both replay and disposal regressions.
