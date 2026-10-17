# CrispASR — Pending work

Updated 2026-10-17. This file contains current actions, dependencies and a
legacy audit inventory. Completed work and experiment receipts live in
[HISTORY.md](HISTORY.md#plan-reconciliation-archive--2026-10-15).
The archived snapshot preserves detailed older requirements without treating
its stale status labels as current facts.

## Working protocol

Read the sibling development guide and AGENTS.md. Work in a separate worktree;
large models, builds and artifacts belong under /mnt/storage. Before coding,
add a dated CLAIMED block naming the owner, worktree, scope and acceptance
criteria, and push that claim to main. Remove the claim when the work lands.
Read complete issue/PR threads before acting. Audit old notes against code.

Use pinned original Python → conversion/quantization → dump-reference → diff
with cosine, norms and relative error → actual decoded-output acceptance.
Only then profile or promote defaults. Preserve failures and source/input
hashes; use distinct run directories so failed evidence cannot be overwritten.
Follow docs/contributing.md for all affected surfaces, including the C ABI.

Main checkpointad2998ab passes all13 general CI jobs (38855803220), Ruby,
secret scan and Pages. Native-fork worker5417ed6d passes all10 lint jobs
(38855559850) and heavy38855559740; independent auditc0fac6c1 passes642
payloads. These are exact-source checks. The diagnostic integration and future
release tips require fresh CI; older documentation-tip queues are superseded.
Last published release is v0.8.43; v0.8.44 remains untagged. Dia sampling controls are integrated from candidate7d0f68f3 (all13 CI and10
lint jobs passed); all non-document source bytes are preserved. Fresh main
CI remains pending. Echo observer integration preserves the complete non-document
tree of3a13e0c6 (all13 combined CI and10 lint jobs passed). Routing controls do
not establish native speech acceptance.

## Ordered roadmap

1. **Dia correctness:** validate current-main speech, reconcile reference/diff
   wiring, then broaden prompts/seeds before GPU/Q4 and profiling.
2. **CANN hardware preparation:** compose fallback/ACL controls and execute on
   physical Ascend when available; do useful CPU/SDK preparation meanwhile.
3. **Remaining PR decisions:** salvage independently accepted #492/#521 pieces
   with contributor history intact; retain failed quality/performance gates.
4. **Quantization:** bounded Echo Q4 and Nemotron precision experiments.
5. **Documentation:** keep this plan current and archive terminal work promptly.
6. **Release:** freeze accepted scope and run immutable package/wheel validation
   before scripts/bump-version.sh, tag and publication.

Hardware waits do not justify repeating accepted diagnostics or waiting hours
without independent useful work. GH handles heavy CPU/builds; Kaggle handles
actual supported GPU tests per ../kaggle_usage.md. Neither supplies Ascend.

## Dia correctness — current dependencies

The completed reduction/cache/RNG, broader matrix and recognition investigations
are in HISTORY.md and their pinned machine receipts. Current claims above carry
the direct-original and full-model observer dependencies.

1. Direct-original CPU construction/recognition is audited and fails speech.
   Original README F16 CUDA is independently audited: both fox seeds pass,
   legacy and both hello seeds fail; both recognizer positives pass. Its81 payloads
   and82 anonymous public downloads are verified. The subsequent trajectory
   experiment is fully audited (201 payloads/202 anonymous checks): no invalid
   codes were clamped, and all three quiet paths use one common nine-book vector
   for over99% of frames. Both hello and fox cases pass exact observer identity;
   first legacy OFF0 differs slightly, so the global observer gate remains failed.
   The matched CUDA precision/recipe experiment is independently audited:261
   payloads and262 anonymous file checks. Precision/recipe alone did not recover
   the three quiet cases; see docs/dia-cuda-factorial-results-2026-10-16.json.
   Next isolate categorical draw placement with unchanged CUDA neural compute.
   Existing full-F32 broader quality fails
   on both recognizer architectures; numerical/code equality alone is insufficient.
   Once bounded fixed cases pass unchanged speech gates, test the existing
   heads-only-F32/F16 neural export with full-F32 positive controls before GPU/Q4
   and profiling. Preserve source/physical audits and all earlier failures.
   Do not add token overrides, change seeds or relax thresholds as a fix.
2. Earlier module Q differs6.90e-7 from actual full-model Q despite exact keys
   and normalized input. Preserve that qualification. A new intervention must
   use the actual original tensors and native identity controls, with cosine,
   magnitudes, relative errors and decision-margin effects. Do not repeat the
   rejected qualified-Q, norm/head/late-layer/source-RoPE sweeps unchanged.
3. Any production candidate must run current runtime without reference injection,
   preserve controls, and pass real speech on bounded additional prompts/seeds,
   then actual GPU/Q4 and performance. Single-seed diagnostic recovery is not
   production acceptance. If no new operator hypothesis is justified, advance
   independent CANN preparation instead of repeating accepted diagnostics.

## CANN — hardware follow-ups after merged PR544

Implementation and official SDK builds are integrated. The portable Moonshine
validator and its original/CPU/CLI/Session proof are ready. Remaining actions:
1. Run the fixed full-model fixture on physical310P/910B, preserving actual
   placement/offload, original stage/magnitude metrics and decoded transcripts.
2. Prove composed scheduler fallback: supported CANN work plus rejected BF16
   CPU operations, with actual output checks; physical310P required.
3. Prove actual ACL capture/replay, repeated calls and changing shapes on910B.
4. After floating-point acceptance, validate quantization and then profile.

Maintainer owns the preparation; contributor hardware help is optional. No
further rerun of the already accepted rotary repair is requested. Details and
commands: [CANN TODO](docs/cann-todo.md) and
[portable validator](docs/cann-model-validator-preparation-2026-10-14.md).

## Remaining PRs — scoped salvage and acceptance

- **PR492 / MiMo:** author KokerZhou closed the PR unmerged on2026-10-15;
  it is no longer an open merge candidate. Independent opt-in mel/cached-decode pieces are merged.
  Dynamic Q8 activation amplification is measured; the existing F32 cache-read
  hypothesis is independently rejected and archived in HISTORY.md.
  [Four-arm result](docs/mimo-kvread-ab-2026-10-15.md): hidden/logit error remains
  5.73%/5.13%, with a0.5% gate. The default Q4 original-stage gap also remains.
  Next select a specific attention/activation arithmetic hypothesis on owned
  inputs under a fresh claim; do not repeat the old BLAS/cache-read sweeps.
  CANN RVQ/F16 embedding and ggml NZ need physical Ascend and PR-only/NZ-only/composed attribution. The measured
  full tokenizer path was7–10% slower; do not enable it by default. Review the
  complete latest thread before proposing any further split or merge.
- OmniVoice diagnostic data recovery is complete: the recorded HF pin returns
  404, but exact original GH artifact11600052124 has been recovered, independently
  array-audited and permanently republished with a verified anonymous download.
  See HISTORY.md and docs/omnivoice-clone-diagnostic-2026-10-09.json. Remaining
  work is runtime/input/precision/quality diagnosis, not another proof search.
- **Issue518 / draft PR521 / OmniVoice:** audio/text utilities match upstream;
  the fixed clone gate fails2/9 WER in both original Python and native output.
  The standalone utilities are integrated with complete original/native array
  replay and hosted checks; see HISTORY.md. The full runtime PR remains a draft.
  Do not call the onset repaired or relax the clone gate. Preserve controls and
  contributor history.
- **Breeze:** accepted guided/voice/server fixes are merged. Investigate canonical
  BF16 grid equality (0/8 despite passing scoped speech controls) and unresolved
  model-quality reports from merged contributor PR530 only with a new hypothesis.

## Quantization — no unproven model promotion

- **Index-Echo / issue485:** higher-precision accepted paths remain available.
  All four plain recipes and three calibrated recipes fail complete acceptance.
  Collection covers all96 FFN statistics and sensitive source F32 tensors are
  preserved. Locate the first actually failing metric and attribute FFN
  gate/up/product/down on the same input; first quantized block0/4 alone does
  not establish first failure. In the middle recipe, block4 cosine0.999908
  already passes the0.999 gate.
  then test targeted precision changes against complete EN/ZH stages, magnitudes,
  cache behavior, decoded/subtitle output and TTS-ASR. Do not repeat rejected
  recipes unchanged. Evidence: docs/index-echo-q4-imatrix-validation-2026-10-09.json.
- **Echo performance:** scheduler opt-in has narrow T4 evidence; new CPU/Metal/
  CUDA A/B must establish output and timing before any default change.
- **Nemotron:** accepted experimental hybrid T4 controls are not compact-Q4
  acceptance. Hybrid is1,179,413,408 bytes, only14,155,776 Q4 bytes and8.3%
  smaller than F16. Audit promotion/package scope separately; seek meaningful
  compression with preserved original stage/speech/session/streaming behavior.
  No speed or production pin is established. Evidence:
  docs/nemotron-position-q4-gpu-2026-10-10.json.

## Reporter, hardware and asset dependencies

- **Issue532 / Canary:** CPU/T4 reuse/allocation controls did not reproduce the
  crash; await original recording/device details. Do not apply a speculative fix.
- **Issue483:** matched CUDA/MMQ packages are prepared; actual GTX1660/MX150
  correctness and timing remain external. No generic-GPU speed conclusion.
- **Issue461 / VoxCPM2:** profile VAE setup and Vulkan LocDiT on relevant hardware.
  Eight-step RTF1.019 and poorer reported quality do not justify changing the
  ten-step quality default. Issue478 Windows/Khmer retest, Q4 batching and
  other-ISA/GPU coverage remain separate; verify current issue state first.
- **Issue456 / Nyra:** gated NC model/lexicon/GMM-HMM/Viterbi assets and licence
  scope must be resolved before a port; espeak-ng cannot be assumed bundled.
- **Dolphin:** merged original-reference fix does not prove reporter Kylin/GCC9,
  private clips or GPU behavior. Obtain those inputs if further work is needed.

## Release v0.8.44

Comprehensive notes exist in RELEASE_NOTES_v0.8.44.md and
[the next-release draft](docs/release-notes-next.md). Remaining work:
- Freeze accepted source and scope; exclude rejected Dia/Q4/NPU experiments.
- Run fresh immutable package/wheel dry-run and inspect artifacts, dependency
  closure and relevant CLI/C-ABI/binding behavior on the frozen source.
- Require applicable tip CI and acceptance evidence; use scripts/bump-version.sh
  for propagated version and annotated tag, then publish and inspect assets.
- Update final release notes to describe shipped behavior and known limitations.

Passing current source CI does not prove new v0.8.44 assets have been built.
Unrelated future research need not be advertised as accepted to release proven
improvements. Existing failed gates must stay explicit.

## Research backlog — Drama3 (issue545)

Issue546 requests the same backend, explicitly including GGUF conversion and
zero-shot voice cloning; track those requirements together with545. Cloning
must cover the original reference-audio processing and actual CLI/C ABI paths,
with closed-loop decoded-speech/voice validation before claiming support.

Metadata/source/licence triage is complete; no weights or implementation started.
Pin fishaudio/drama-3 e0a9218aa48888270f06ef1ed9ea67622c0e17d2 and read the
actual inference/codec blueprint before defining a port. Scope includes40 hybrid
layers,256-expert/8-selected MoE, shared experts, zero-centered norms, partial
RoPE, a4-layer ten-codebook MTP decoder and codec, with FP8-block/BF16 weights.
Fish Audio Community Licence is custom, not Apache2; establish redistribution
terms before conversion/public artifacts. Audit ggml operation/backend coverage,
then original conversion/dump/diff/decoded-speech pipeline on suitable hosted
hardware, sensitive-tensor quantization and full contributing checklist. Keep
behind correctness work and avoid an unbounded35B download on this VPS.

## Research backlog — independently trained EmbeddingGemma speech reader

User requested a concrete task for an independently trained reader, word bank
and reproducible decoder. This is a research backlog task, not implemented
support or a claim to reproduce the author's unpublished reader. Feasibility:
`docs/embeddinggemma2-asr-feasibility-2026-10-13.md`.

1. Pin and read the official encoder implementation and preprocessing. Compare
   temporal audio-tower features with features after the shared backbone; do
   not substitute pooled recording embeddings for time-resolved features.
2. Freeze the encoder and train a small temporal CTC reader. Compare ordinary
   subword CTC with embedding-matching word CTC on the same features/data so
   the shared word space's contribution is measurable. CTC is our proposed
   recipe, not an assertion about Shahir's implementation.
3. Build a licensed, versioned word bank with normalization, stable IDs and
   pinned text embeddings. Start around 30–50k words; define handling for
   names/numbers/unseen words. Exclude held-out evaluation transcripts from
   vocabulary construction. A 229k-by-768 FP16 bank alone is 351,744,000 bytes;
   measure dense scoring cost before choosing compression or approximate search.
4. Specify the exact decoding contract: sample rate, feature layer, chunking,
   overlap, score normalization/temperature, blank score, CTC repeat/blank
   handling, boundary merging and optional beam search. Begin with greedy CTC;
   keep punctuation separate from the initial acoustic-quality experiment.
5. Pilot on LibriSpeech train-clean-100 with separate development/evaluation
   sets, then expand training only after measurable held-out progress. Report
   WER on clean/other speech, silence and out-of-vocabulary behavior, memory
   and feature-extraction/training throughput. Do not promise the author's
   ~5% WER, 288x speed or eight-H100 training time on a Kaggle GPU.
6. Use Kaggle for heavy encoder extraction/training and GH CI for conversion,
   reproducibility and integration checks. Keep datasets/features/checkpoints
   under `/mnt/storage`; checkpoint produced artifacts before later risky work.
7. After a useful Python prototype, port via convert → reference dump → stage
   diff with cosine AND magnitudes → decoded transcripts. Establish F32/BF16
   and Q8 baselines before lower quants; validate CPU/Metal/actual CUDA and all
   `docs/contributing.md` integration points, including C ABI and bindings.
   Reuse audited CrispEmbed components where applicable; its typed-decision
   backend is not already a temporal audio encoder or speech reader.

Planning estimate, not a delivery promise: 1–2 engineering weeks for a research
prototype, then several more for training iterations and a validated ggml port.
First decision gate: useful held-out transcripts from a small frozen-feature
reader versus its subword baseline. Reader/checkpoint/bank provenance and
redistribution terms must be documented. Matching advertised quality/speed is
an experimental target, not a release acceptance assumption.
Sources: https://arxiv.org/abs/2306.11473,
https://huggingface.co/google/embeddinggemma-2/blob/main/config.json,
https://www.openslr.org/12. Upstream ggml support and base Apache-2.0 weights
already exist; the app's own reader assets were not located in our source check.


## Older backlog — audit against current code before scheduling

These are preserved requirements, not assertions that an old OPEN label still
means missing implementation. None are active claims. Consult the complete
[archived requirements](HISTORY.md#plan-reconciliation-archive--2026-10-15),
search the heading below, audit code/tests and current issue state, then either
schedule the remaining action here or record completion in HISTORY.md. Never
repeat completed work solely because the archived note says OPEN.

- Audit remaining scope: live transcribe + translate: follow-ups
- Audit remaining scope: §247 shared scheduler profiler, then hidden-F16 audit
- Audit remaining scope: (optional) — #416 Sidon quant: CUDA MMQ arm
- Audit remaining scope: #387 Quds follow-up
- Audit remaining scope: canary follow-ups (after #375, record in HISTORY)
- Audit remaining scope: vibevoice-asr 7B answers "[Silence]" on time-stretched audio
- Audit remaining scope: vibevoice-bitnet advertises caps its default output cannot support
- Audit remaining scope: TQ2_0 has no Metal kernels: BitNet models are silent on GPU
- Audit remaining scope: carried out of the #316 round-2 work
- Audit remaining scope: follow-ups from #332 (Rust/Dart diarize ABI, landed 2026-08-05)
- Audit remaining scope: #326 diarization: the count estimator is the last accuracy item
- Audit remaining scope: stamp the kartoffel repos (needs ~15 GB free; this box has 11)
- Audit remaining scope: Why not the hash-chained log the sibling projects built
- Audit remaining scope: follow-ups from #300 / #308 (landed 2026-07-27, see HISTORY)
- Audit remaining scope: #316 Kokoro G2P English residue
- Audit remaining scope: VAD + mel front-end parallelization campaign (#305 → fleet-wide)
- Audit remaining scope: #296 mel-band-roformer — follow-ups PROMISED on the issue (do not drop)
- Audit remaining scope: #266 follow-up — hoist speaker orchestration into the library (PARKED, LOW)
- Audit remaining scope: Untrusted-input parser hardening — follow-ups
- Audit remaining scope: Diff-harness extensions (detail for "Ready to take" #5 and #6)
- Audit remaining scope: CometBeat handoff — singing-voice-conversion vocoders (OPEN, NOT STARTED)
- Audit remaining scope: Gemma-4 12B (gemma4_unified) ASR support (OPEN)
- Audit remaining scope: crispasr-diff harness extensions — catch decode-policy / quality bugs (OPEN)
- Audit remaining scope: #201 follow-up — generate a TADA voice ref from audio+transcript at query time (C-ABI + server DONE gated; roundtrip pending)
- Audit remaining scope: §192 follow-up — native-Vulkan TADA garbled output FIXED (codec on CPU)
- Audit remaining scope: moss-transcribe follow-ups (OPEN, LOW)
- Audit remaining scope: #218 qwen3-asr long-audio root cause — quantized audio tower + prompt contract
- Audit remaining scope: #218 glm-asr long-form blueprint parity — instruction + multi-window
- Audit remaining scope: #218 qwen3-family rebake + CUDA validation — kernel results
- Audit remaining scope: Priority ordering
- Audit remaining scope: §214 follow-up — chatterbox T3 batched-CFG (B=2) — deliver the win (OPEN)
- Audit remaining scope: §215 — batched-CFG (B=2) for the remaining TTS backends (OPEN)
- Audit remaining scope: §210 follow-up — shape-stable bucketed decode for remaining LLM/AR backends (CUDA-graph capture) (OPEN, CONDITIONAL)
- Audit remaining scope: §219 — more permissive audio input formats for crispasr_audio_load
- Audit remaining scope: §166 follow-up — WASM `asr*` session surface needs a build-verify
- Audit remaining scope: §175 Surgical DRY — share pure helpers across the CLI/library boundary
- Audit remaining scope: §177 VibeVoice #171 — remaining layer: RDNA4/RADV quantized-matmul path (OPEN)
- Audit remaining scope: 40. More Moonshine model variants
- Audit remaining scope: 97. More Parakeet variants
- Audit remaining scope: 98. Hotwords / contextual biasing (OPEN)
- Audit remaining scope: funasr — perf follow-ups (LOW priority, not blocking)
- Audit remaining scope: §187 Cross-runtime embed fast path sweep
- Audit remaining scope: §201 Kaggle CUDA backend failures — full sweep 2026-06-20 (OPEN)
- Audit remaining scope: 51c. MiMo-V2.5-ASR F16 step decode — open (validation only)
- Audit remaining scope: 52. Qwen3-TTS
- Audit remaining scope: 54-follow-up. granite-speech-4.1 plus speaker labels + word timestamps — open
- Audit remaining scope: 56. Kokoro multilingual phonemizer (espeak-ng)
- Audit remaining scope: 57. Commercial-friendly TTS backend expansion (OPEN — Phase 1 done)
- Audit remaining scope: Ecosystem expansion (lower priority)
- Audit remaining scope: 59. Cross-binding C-ABI parity
- Audit remaining scope: 65-residual. JS / emscripten word-accessor surface — open
- Audit remaining scope: 67. Deferred follow-ups carry-over (mid-May 2026 session)
- Audit remaining scope: 70. Streaming TTS via chunked VAE decode (latency win, vibevoice / qwen3-tts)
- Audit remaining scope: 75-followups. /v1/audio/speech OpenAI round 2 — open
- Audit remaining scope: 168. GPU scheduler migration — gallocr-only backends
- Audit remaining scope: 91. CrispASR CLI features missing from CrisperWeaver (OPEN — parity gaps)
- Audit remaining scope: 92. All-backend regression suite (nightly CI)
- Audit remaining scope: 95. IndexTTS-1.5 Chinese TN — binary alternative to the Python `wetext` hook (OPEN)
- Audit remaining scope: 102. RapidTP-Aligns — dedicated NN timestamp predictor (survey)
- Audit remaining scope: 104. Stateful frame-streaming TDT decode for parakeet long-form (issue #89)
- Audit remaining scope: 105. WhisperX word alignment models — wav2vec2 CTC zoo
- Audit remaining scope: 106. TEN-VAD — low-latency cross-platform VAD
- Audit remaining scope: 125. Issue #125 — multi-backend bug sweep from montvid (mostly DONE — validation + longer-term threads open)
- Audit remaining scope: 127. Coverage gaps from the 2026-05-26 overlap-save sweep close-out
- Audit remaining scope: §136 — funasr CUDA !-loop fix (issue #125)
- Audit remaining scope: §139 Beam search — remaining ASR backends (issue #136 follow-up)
- Audit remaining scope: §TTS-PROV: TTS AI-provenance compliance (watermark, C2PA, consent, disclaimer)
- Audit remaining scope: 156. Permissive G2P phonemizer
- Audit remaining scope: §167 — Beam search, MAES, and KV caching expansion
- Audit remaining scope: §176 Runtime optimization pass — 18/20 DONE, 2 open
- Audit remaining scope: §ARK — ARK-ASR-3B support (⚠️ EXPERIMENTAL / WIP; branch feat/arkasr-3b)
- Audit remaining scope: llama.cpp comparison — perf-tricks to adopt (ANALYSIS / OPEN)
- Audit remaining scope: 221. Issue #89 hardening + v0.8.8 release
- Audit remaining scope: §222 Aligner model expansion — permissive-license fleet (OPEN)
- Audit remaining scope: §224 Issue #222 follow-ups — silero LID ggml graph (DONE) + diarize/firered CPU perf (OPEN)
- Audit remaining scope: §226 irodori-tts GPU + codec GGUF fix
- Audit remaining scope: §227 starling comparison — CUDA decode-loop optimization options (OPEN)
- Audit remaining scope: §229 transcribe.cpp perf audit — vs handy-computer/transcribe.cpp (OPEN)
- Audit remaining scope: §232 qwen3-tts code predictor perf — fuse 15 graph dispatches into 1 (#245, OPEN)
- Audit remaining scope: §235 next perf targets — triage after the §232 qwen3-tts sweep (2026-07-11)
- Audit remaining scope: §234 omnivoice — persistent step graphs + silence root cause
- Audit remaining scope: §232 transcribe.cpp parity — close the RTF gap (mostly CLOSED; residual items OPEN)
- Audit remaining scope: Runtime speedup roadmap (2026-07-11 cross-repo sweep)
- Audit remaining scope: §247 roll #81 techniques out to the other runtimes (OPEN)
- Audit remaining scope: §248 — audio.cpp competitive benchmark + gap analysis (IN PROGRESS)
- Audit remaining scope: §249 WebRTC VAD — remaining follow-ups (backend DONE 2026-07-19, see HISTORY)
- Audit remaining scope: §250 — Music transcription backends (piano_transcription + Basic Pitch)
- Audit remaining scope: §251 — Music-analysis infrastructure + the remaining CometBeat roster
- Audit remaining scope: §252 — CosyVoice3 native-Vulkan: fp32-accumulation lever (LOW priority, likely dead end)

Legacy publishing/bootstrap claims require checking current registry/workflow
state; do not assume the old404s or missing credentials remain true. Legacy
MioTTS/Glint integration and completed native sample-rate/binding/UTF-8 fixes
are historical, not work to reimplement. Consumer-requested binding expansion
can be scheduled after checking actual exported and wrapped surfaces.


## CLAIMED 2026-10-16 — Isolate Dia categorical draw placement

Owner: coordinator; fresh worktree /mnt/volume1/crispasr-dia-sampler-device-20261016,
branch diag/dia-sampler-device-20261016, frozen source4d622e7e9574c360c0f38154797a3453f26e2096.
Kernelcrispasr-dia-sampler-device-v1/version1 is COMPLETE; uploaded source is
byte-exact, SHA4db1fd69a641f6960cff2c9f5969ed6570ee0c47890dcff75f80f5468781d435.
Tiny actual original sampler/ID/RNG and generation/delay/frame-count controls
pass under Torch2.6 and2.11. Exception restoration, broken-restoration and invalid
input controls pass; read-only review finds no blocking source defect. Three
recent sessions were terminal before one submission; terminal collector is active.
Actual T4 tiny controls completed and all73 public files are anonymously
hash-verified at revision65b7524ef49366fc4c570870be2b1620b8446501. CPU/CUDA
differ on all288 corresponding nondegenerate IDs; both one-hot controls match.
CPU placement matches direct CPU IDs/RNG, returns IDs to CUDA, and leaves its
RNG unchanged. Independent local Torch2.11 replay matches432 CPU IDs and all
six after-RNG states. First five CUDA-draw syntheses checkpointed publicly at
revision15ce3f04cac9d7274a1ff59fa76613a666e2d87f before the CPU-draw arm.
Final raw7bc65637442e50bd17f8a6abb5139781f7ba9883 is independently audited:
288 payloads/289 full anonymous downloads. CUDA arms pass both fox cases; CPU
draw fails both. All three remaining cases fail speech in every arm. Actual CPU
replay matches2880 first64 IDs; all15 feedback reconstructions match original
generate with stubbed neural steps. Public independent audit pack is publishing.
See [preparation receipt](docs/dia-sampler-device-preparation-2026-10-16.json).
Scope: original F32 neural computation
and filtering on one supported CUDA host; same model instance and fixed native
recipe1.2/45/.95/CFG3/cap3072, original five texts/seeds. Predeclare CUDA draw,
CPU draw, CUDA draw anchor (15 outputs). Middle arm copies final probabilities
to CPU for the original torch.multinomial and returns IDs to CUDA; no other
inference arithmetic, token forcing, prompts, weights, seeds or codec changes.
This is an explicit sampler placement intervention, not observer transparency.
Before model downloads, capture tiny identical CPU/CUDA uniform/linear/one-hot
9x1028 probability draws, RNG states and deterministic one-hot controls. Retain
runtime framework source revision. Source-preserving local tiny operator and
original sampler tests must pass, with changed-placement/restoration negatives.
For full generation preserve CPU/CUDA states before/after, draw counts and
byte-exact probability transfer assertions; unchanged CPU state in CUDA arms,
unchanged CUDA state in CPU-draw arm. Require anchor speech outcomes to match,
both sealed recognizer positives to pass, all15 PCM/raw recognition cells retained.
No precision/default/Q4/speech promotion from diagnostic completion. Submit once
after current session inventory per ../kaggle_usage.md; no CPU neural fallback,
framework replacement, quota workaround or draw fishing. Public HF checkpoints
before weights and between arms; independently audit terminal pinned evidence.
All large assets under /mnt/storage/crispasr/dia-sampler-device-20261016.


## CLAIMED 2026-10-17 — Dia final delayed BOS boundary

Owner: coordinator; fresh worktree /mnt/volume1/crispasr-dia-bos-boundary-20261017,
branch diag/dia-bos-boundary-20261017. The pinned original prefill explicitly
places BOS in row15/book8 (delay15), but original generate decrements its BOS
countdown before checking >0, so write step14 disables preservation and overwrites
that prepared BOS at target row15. Native dia_tts.cpp mirrors the same boundary.
This is a concrete inconsistency to test, not a proven quality cause or production
fix. Tiny actual original generate/DecoderOutput must expose the invariant failure
before intervention; use no neural weights. Predeclare original / preserve final
prepared BOS / original anchor, with only update_one's apply_mask flag at target
row15 changed. Assert original update method restoration and a single changed
book8 slot; wrong-step/removed-preservation negatives must fail. Cover sampled and
capacity EOS, existing delay replays and sampler controls. Preserve original
sources and all old failures. If tiny gates pass, one supported CUDA experiment
uses unchanged original normal F32 CUDA model/filter/RNG, fixed native1.2/45/.95/
CFG3/cap3072 and original five texts/seeds; same instance and both fox plus sealed
recognizer positives. No seed/prompt/forced-code changes. Each arm checkpoints
publicly; keep actual probabilities/IDs, BOS before/after, RNG states, PCM and raw
recognition. Independently audit full output before any native graph/default/Q4
change. No full neural weights locally; all assets under
/mnt/storage/crispasr/dia-bos-boundary-20261017. Actual GPU submission follows
../kaggle_usage.md session checks, with no unsupported-host retry loop.
