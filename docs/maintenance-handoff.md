# Maintainer handoff — 2026-10-18

Start with AGENTS.md, the optional sibling development guide, and PLAN.md.
Machine-specific paths, session names and operational recovery instructions belong
in gitignored CLAUDE.md. Public receipts use `${CRISPASR_STORAGE}` and
`${CRISPASR_WORKTREES}` as location aliases, not literal checkout requirements.
Historical artifact hashes describe original retained bytes; sanitizing a location
in a tracked receipt does not change an immutable public artifact's hash.

This update reconciles #217/#549 and the integrated #547 source fix. Other
fronts below retain their October 17 audit state; inspect current evidence
before assuming their queued/pending labels are still current.

## What is integrated

The bounded CPU attention diagnostic and Echo9B owned-input FFN worker, including
its observer-setup repair, are now on main with original authorship preserved.
They add diagnostic tools; they do not change inference arithmetic or defaults.
The attention source02bd2aee passed all13 CI jobs39003202215 and all10 lint
jobs39003053402. Echo repair sourcec2deea37 passes six local actual released
Transformers5.6.0/Torch2.6CPU controls and all secret checks. Its hosted model job
is still pending. Earlier source967a62e0 passes all10 lint39001788270 jobs but
its first heavy model diagnostic failed; that failure remains archived.

The combined main tip needs its own applicable CI/lint. Check Actions for the
actual main SHA; do not substitute a different branch's green checks. No new
release tag was cut during this handoff.

The #549 metadata source `741141118` passes all 13 CI and 10 lint jobs; long-JA
and legacy overlap token ownership are now implemented with six local GCC/Clang
fixture tests; this follow-up still needs full-library CI and Japanese audio proof. #217's independently authored
character-CTC workspace is opt-in (`CRISPASR_CTC_ALIGN_OPT=1`, default off).
A separate subword workspace now has its own default-off gate
`CRISPASR_SUBWORD_CTC_ALIGN_OPT=1` and five GCC/Clang acceptance tests; the
subword complete-path/ABI failure fix remains integrated. The character synthetic heavy run39112822169 passes all seven shapes;
full-library, subword hosted and actual-model acceptance remain pending. The
[comparison and clean-room audit](alignment-cleanroom-2026-10-18.md) records
scope, local synthetic measurements and the absence of a matched Subtimer
accuracy/runtime benchmark. No external code or fitted asset is ported/vendored.

## First actions

1. Check both Echo jobs below and current main CI/lint. Retrieve terminal logs,
   artifacts and public checkpoints before changing the frozen worker.
2. Audit the integrated issue547 utterance-open throttle fix `dd14099c3` and
   the existing `tools/ci-heavy/issue547_nemotron_multiturn.py` worker/results.
   Do not reimplement that fix or conflate remaining hardware/reporter
   validation with speculative cache teardown or experimental Q4 promotion.
3. Audit Echo's actual same-input FFN tensors if produced. Only then propose a
   bounded precision change. If the job fails, preserve it and repair a fresh
   branch/worktree; do not mutate the tested source or reuse its fixture prefix.
4. Continue MiMo's actual same-input attention capture, then the other quality
   investigations below. Hardware and reporter waits are separate dependencies.
5. Freeze accepted release scope, validate exact packages/wheels and finalize
   release notes after applicable CI. Research failures need explicit scope,
   rather than unsupported fixes or relaxed quality gates.

## Concrete remaining tasks

| Front | Next bounded action and entry points | Acceptance / dependency |
|---|---|---|
| Alignment / issue217 | Character synthetic benchmark `567fc431e`/heavy `39112822169` passes all seven shapes (audited aggregate receipt); combined contract source `deaf9c010`, CI `39113844983`, lint `39113846749`. Subword workspace source `a1434c9b2` has heavy `39115492999`, CI `39115494689` and lint `39115496152` queued; audit its exact-source artifacts. The hosted-only actual-model worker `tools/ci-heavy/alignment_model_profile.py` now pins a licensed 6.36s Japanese FLEURS recording, retains native logits, and checks real C ABI/Python/CLI parity; source `08aea3537` is frozen on `diag/217-japanese-model-profile-20261018`, heavy `39116431721` queued; audit per its method. Boundary-labeled corpus acceptance remains separate. See PLAN.md and the clean-room audit. | Six character-kernel, four subword-contract and five subword-workspace tests pass locally; character larger synthetic evidence is accepted; full-library/subword/model evidence is pending. Preserve default-off gating, exact timestamps and model-family distinctions. No direct Subtimer performance/quality ranking is established. |
| Python metadata / issue549 | Preserve accepted source `741141118`, CI `39104662258`, lint `39104663518`. Japanese slice/gap-fill and legacy overlap now preserve owned token blocks; shared repair fixes tied-frame duplication, and both explicit-output splitters retain same-frame continuation tokens with their owning word. Final source `ebf69c437` is frozen on `diag/549-token-boundaries-20261018`: CI `39117921894`/lint `39117923785` queued. Audit those, then actual Japanese ASR multi-window fixtures per PLAN.md. #217 CTC aligner evidence is a different route. | Existing API fixes pass all 13 CI/10 lint, 2,033 unit cases and 11 metadata tests. All 12 metadata and six ownership tests pass under GCC/Clang, including 2,000 frozen-grouping comparisons; fresh exported-getter CI and a licensed Japanese ASR clip remain. Audit generic CLI timestamp filtering separately before a whole-pipeline parity claim. Not yet in v0.8.43 wheels. |
| Echo9B Q4 / issue485 | Inspect [heavy39008086579](https://github.com/CrispStrobe/CrispASR/actions/runs/39008086579) and [lint39008088033](https://github.com/CrispStrobe/CrispASR/actions/runs/39008088033), sourcec2deea373f9bd27f526f873bc01cb97068e1bff7. Worker: tools/ci-heavy/echo_owned_ffn_run.py; method: tools/ci-heavy/echo-owned-ffn/README.md. Public dataset cstr/crispasr-regression-fixtures, prefix index-echo-9b/owned-ffn-20261017/v2. | First require native FFN OFF/ON/OFF block/logit/prompt identity and complete32-layer JFK/ZH actual-original replay. Independently verify every payload/source/input/physical-weight ledger, cosine, both norms and relative errors. No numerical9B result exists from failedv1. |
| Echo targeted precision | From the completed attribution, identify gate/up/product/down error on identical native inputs, including original down projection on native down input. Choose one measured intervention; use a new public prefix and exact model/source pins. | Re-run complete original EN/ZH stages, magnitudes, cache behavior, decoded/subtitle and TTS-ASR checks. All seven existing Q4 recipes remain rejected. Component attribution alone cannot promote a recipe. |
| Nemotron streaming / issue547 | Source fix `dd14099c3` is integrated; audit the dedicated multi-turn worker and current results before new diagnosis. Read [the complete report](https://github.com/CrispStrobe/CrispASR/issues/547) and any new comments. Exact WAV/PCM and Python repro are offered but not attached as of handoff. Reproduce eb633d90/v0.8.43 and current main, CPU F16 first: four reported Spanish turns with1.2s gaps, stdin s16le16k mono, --stream --stream-json --vad, final-on-silence400ms; compare3000/500ms step, redecode/prefix, reordered and fresh-process controls. Entry points: examples/cli/crispasr_run.cpp, crispasr_backend_nemotron.cpp, src/nemotron.cpp. | Assert every standalone-recognized turn retains a nonempty correct final and appropriate partials across repeated reset cycles, ordering and packet sizes. Capture VAD input spans, consumed samples, frontend/encoder counters, host/device caches and RNNT state. Then C-ABI/session, Q4 and Windows CPU validation. Reporter fixture/Windows evidence is pending. |
| MiMo / closed PR492 | Add actual same-input Q/K/V, mask, scores/probabilities and pre-output-projection capture on the71-row fixture. Entry points: src/core/attention.h existing DBG capture hooks, src/mimo_asr.cpp extraction API, tools/ci-heavy/attention-owned-* diagnostic. Use native OFF/ON/OFF controls and replay original operands, not two separately drifted histories. | Find the earliest unequal operator with norms/relative error. The old capture has no Q/K/V. The tiny scalar F16 accumulator mechanism does not explain71-row tiled prefill. Preserve the0.5% hidden/logit gate, original-stage and real CLI/C-ABI speech checks. Do not repeat old BLAS/cache-read sweeps or reopen the complete closed PR. |
| Dia speech | Read the latest results for original CUDA, precision/recipe factorial, sampler-device and final delayed-BOS controls. Preserve fixed failing legacy/hello cases and passing fox/recognizer controls. Choose a new source-backed operator/input hypothesis before code or another model run. | Original source itself fails the three fixed cases under tested controls. CPU/CUDA draw placement explains only the controlled fox gap; final-BOS preservation does not recover speech. Earlier numerical parity is qualified, not production acceptance. A candidate must improve real speech on unchanged bounded cases/seeds, then broaden, then test actual GPU/Q4 and profile. |
| OmniVoice / issue518, draft PR521 | Utilities are independently accepted and integrated. Full clone gate still fails2/9 WER in both original and native. Use the recovered permanent raw proof and matched input/precision/runtime diagnosis; retain contributor history. | Closed-loop clone/transcript and voice controls must pass without widening gates before undrafting or merging the full runtime. No more proof-archive searches or reimplementation of already integrated utilities. |
| Breeze | Existing guided/voice/server fixes are integrated. Audit canonical BF16 equality on the retained0/8 grid against F32 controls; pick a new hypothesis and fixed input before a new run. | Preserve all original-stage, magnitude and decoded-speech gates. Existing scoped speech passes do not establish BF16 grid equality. |
| CANN / merged PR544 | CPU validator, composed fallback probe and official SDK compilation are already integrated. Run fixed floating-point model plus composed supported/rejected graph on physical310P, and actual ACL capture/replay with repeated/changing shapes on910B. Entry: docs/cann-todo.md and tools/ci-heavy/cann-composed/README.md. | Record actual placement/offload, fallback/ACL execution, original stage/magnitude metrics, CLI and Session transcripts. SDK/operator passes do not establish full-model behavior. GH/Kaggle do not supply Ascend; do not repeat accepted SDK rotary work. |
| Canary / issue532 | Request/recover reporter recording, exact device and command; compare current main CPU and actual reported GPU route. Existing CPU/T4 allocation/reuse controls do not reproduce the crash. | Reproduce before fixing; retain input/source/device hashes and post-fix repeat/ASan checks. Missing reporter evidence is an external dependency. |
| CUDA compatibility / issue483 | Use the accepted matched CUDA12.6.3/MMQ packages for paired correctness and timing on the actual GTX1660/MX150. | Exact same clips/config/build flags, decoded output and repeated timing with actual device placement. Existing T4 checks do not establish those GPUs' speed. |
| VoxCPM2 / issue461 | Profile VAE setup and Vulkan LocDiT on the reporter's Intel B390V hardware, keeping the10-step quality default. Reconcile Windows/Khmer issue478 and batching separately against current state. | Output/voice quality and controlled timings before optimization/default changes. Eight-step RTF1.019 plus lower quality is not an accepted default. Hardware/reporter evidence pending. |
| Nemotron compact quantization | Keep the accepted experimental hybrid separate from issue547. Current hybrid is only8.3% smaller than F16 and contains just14,155,776 Q4 bytes. Choose a meaningful bounded precision recipe only after original-stage attribution. | Full original-stage/magnitude and speech/session/streaming acceptance, real GPU correctness, then performance. No compact-Q4 or production pin is established. |
| Echo performance | New CPU/Metal/CUDA scheduler A/B only after output acceptance, using fixed source/model/audio and actual cache/placement traces. | Repeated warm timings plus output equality/quality. Narrow T4 opt-in evidence is not a cross-platform default decision. |
| Nyra / issue456 | Resolve gated NC model, lexicon, GMM-HMM/Viterbi assets and redistribution scope before implementation; inspect original inference flow. | Lawful available pinned assets, original reference/alignment acceptance and complete contributing checklist. Do not assume espeak-ng is bundled. |
| Drama3 / issues545 and546 | Treat both reports as one backend request including GGUF conversion and zero-shot cloning. Pin the metadata/source already recorded in PLAN.md, resolve the custom Fish Audio Community Licence, then read full inference/codec blueprint and audit ggml ops. | No unbounded35B VPS download. Original conversion/dump/diff, clone and speech controls, sensitive-tensor quantization and complete contributing surfaces are required on suitable hosted hardware. This is research, not advertised support. |
| EmbeddingGemma speech reader | Follow the detailed research task in PLAN.md: pinned temporal encoder features, independently trained reader, licensed/versioned word bank and exact decoder contract; compare subword CTC with embedding-matching CTC. | Held-out transcripts/WER, silence/OOV, provenance and memory/throughput before a ggml port. No claim to reproduce the author's unpublished reader, quality or speed. |
| Dolphin reporter platform | Existing original-reference repair is integrated; obtain Kylin/GCC9 build details, clips and actual device path for remaining platform reports. | Reproduction and platform output/CI evidence; generic CPU success does not certify the reporter's setup. |
| Release v0.8.44 | Freeze accepted source, reconcile RELEASE_NOTES_v0.8.44.md and docs/release-notes-next.md, run release empty-tag package dry-run and isolated wheel relocation/load/C-ABI checks. Previous998f8d18 preflight built0.8.43, not next-release assets. | Exact tip CI/lint, portable ISA/dependency closure and accepted scope first; then scripts/bump-version.sh, propagated versions, annotated tag, publication and inspection. Retain explicit Dia/Q4/OmniVoice/Ascend limitations. |

## Important Nemotron reset qualification

The adapter already calls nemotron_stream_reset. The native clear function resets
host vectors/counters/decoder state, while the persistent backend cache is cleared
in the encoder on its next `first` call. A retained `dev_state` pointer alone does
not prove stale cache contents. Trace the actual call path and counters on both
reporter and current source; do not add an unconditional teardown as a speculative
fix. The CPU backend also uses backend-resident state, despite an older GPU-only
comment in the stream struct.

## Evidence and older backlog

Latest receipts: docs/index-echo-owned-ffn-preparation-2026-10-17.json,
docs/index-echo-owned-ffn-failed-2026-10-17.json and
docs/mimo-attention-operator-2026-10-17.md. Dia's final results and immutable public
proofs are linked near the top of HISTORY.md. Completed utilities, package
preflights and hardware preparation stay historical; do not repeat them because
an old OPEN label survives elsewhere.

PLAN.md's older inventory is an untriaged requirements list. To activate one:
locate its archived heading in HISTORY.md, inspect current code/tests and complete
issue discussion, define a reproducible missing behavior, then claim a bounded
worktree task with acceptance and hardware/data requirements. Close stale items
into HISTORY.md. Legacy reference/test tooling that assumes a developer mount
should use explicit CLI/environment paths when touched; avoid a broad unrelated
portability rewrite during a numerical diagnosis.

## Handoff checks

Use `gh issue list`, `gh pr list` and `gh run list --branch main` to refresh state.
The sole open PR at handoff is draft521 at381f3a3a; issues456,461,483,485,518,532,
545,546,547 remain open. Read whole discussions before acting on new updates.
Run the secret/privacy suite with the maintainer account environment supplied
privately so the account guard is not skipped. Check document anchors, generated
LEARNINGS index, workflow shell and version consistency after documentation edits.
Keep all heavy models/builds on suitable hosted workers; local operations follow
the ignored maintainer storage instructions. Never print or commit credentials.
