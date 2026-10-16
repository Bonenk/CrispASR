# v0.8.44 — Dolphin decoding, Breeze synthesis and CANN device routing

**Release candidate notes; v0.8.44 has not been tagged or published.**

Changes since v0.8.43 include merged PRs #541, #542, #531, #543 and #544,
plus the validated backend-operator wrapper. The Breeze integration preserves
contributor history from #529 and #536; the session integration preserves #530.
The former v0.8.43 fixes are documented in [its release notes](RELEASE_NOTES_v0.8.43.md).

Dolphin non-streaming models now decode with their actual checkpoint masking
and convolution rules, and automatic binding sessions recognize their GGUFs.
Breeze receives instructions and reference audio together, initializes dynamic
GPU backends correctly, and handles server voice-cache changes. TTS sessions
preserve owned buffers and correct conversion rates. CANN exposes backend and
device routing with independently validated x86_64 and ARM64 compilation.

The notes below distinguish decoded model evidence, source-level regression
controls and SDK compilation. They do not announce a new default quantization,
performance gain or completed Ascend hardware validation.

## Validation still pending

The broader Dia CPU matrix completed all four predeclared cases but failed
speech quality: original and native ASR outputs are empty. The official3.02GB
artifact digest and1829-payload independent audit are verified: all352 stage
rows pass and20556 original sampling decisions match. Crossed recognizer replay
is independently verified across all12 cells: both old/current libraries pass
legacy/JFK but return empty text for all four expanded original clips. Original
Whisper independently passes both controls and fails all16 expanded audio views;
its complete77-payload input/loader/output audit is verified. Direct original
generation without added observers completed all five clips, then hit an ASR
library-alias validator failure. Its94-payload source/PCM audit is verified;
recognition continuation now passes its115-payload/seven-cell audit: both sealed
ASR positives have WER0, but all five newly generated direct clips return empty
text/WER1, without resynthesis. Expanded PCM is
extremely close but not byte-identical, and legacy length/amplitude differ from
the older capture; paired environment controls are required for cause attribution. Complete failed
evidence is permanently public. No broader speech or F16/Q4/GPU/default promotion.
Echo's fused-product observer repair passes all13 general CI and10 lint jobs.
The full-model observer passes its independent1946-payload audit on EN/ZH,
including all690 numerical comparisons and actual decoded/CLI/CABI controls.
Complete3.505GB evidence and auditor pack are permanently public and anonymously
hash-verified. Landing compositiond51b6cf3 remains pending build/lint acceptance
before integration. Original Dia's F16 CUDA control completed on T4: both fox
seeds pass WER0, while legacy and both hello seeds fail with empty text/WER1.
Both recognizer positives pass. All81 payloads and82 full anonymous downloads
are independently verified, with complete public raw and audit proof. This does
not isolate precision, device, Torch or recipe effects and establishes no native
or default acceptance. See [CUDA results](docs/dia-original-cuda-results-2026-10-16.json) and
[PLAN.md](PLAN.md), with completed investigations in [HISTORY.md](HISTORY.md).

Dia top-k/top-p routing through Session and CLI is prepared on a frozen candidate,
including missing CLI flags and restoration of configured defaults between server
requests. Literal-source positive and old-source negative controls pass; hosted
CI (13 jobs) and lint (10 jobs) pass; integration remains pending. This enables the planned recipe comparison
without changing the shipped sampling defaults or establishing speech acceptance.

## OmniVoice allocation failure handling

Both public code-generation and PCM-decoding output allocations now return
null and set output length to zero when allocation fails. Previously these
paths could pass null to memcpy and crash. The regression extracts the actual
production function bodies: five normal/OOM/gain cases pass; removing either
guard reproduces the crash, as do the frozen original functions. All13 general
CI jobs and all10 lint jobs pass on the byte-identical integrated candidate.
This bounded fix does not establish full PR521 or clone-quality acceptance.

## Dolphin: non-streaming decoding and automatic binding sessions

Honor the checkpoint causal flag in both depthwise convolution sites and preserve
the original causal/noncausal LayerNorm/padding order. Register Dolphin in the
shared architecture table so automatic binding sessions can open its GGUFs.

Both non-streaming and streaming models pass Chinese/JFK original references:
64 stage comparisons, exact CTC frame argmax, exact original raw transcripts
and matching automatic C ABI/Python sessions. Independent checks cover all
128 complete native/reference arrays. The 12 original-module/native controls
and causal-only/normalization-order negative controls are also retained.
See [the full proof receipt](docs/dolphin-noncausal-2026-10-12.json). All 45 applicable general CI checks pass, one intentional skip; the fix is
merged on main. Reporter Kylin/GCC9/private-clip and GPU validation are not
established.

## Breeze: direct GPU initialization

Register dynamic backend plugins before selecting the Breeze compute device.
Direct library initialization now finds CUDA; verbose initialization reports the
actual compute backend. All 36 applicable PR checks pass, with two intentional
skips. The composition package exercises the identical initialization prefix on
Tesla T4: Breeze uses 6482/6924 MiB in stage/guided dumps, 107 original stage
comparisons pass, and all 20 speech/interface outputs remain byte-identical to
the prior real-CUDA run. The later branch-cache code differs between composition
and standalone builds; this is not a complete standalone-binary GPU proof.
See [the CUDA dispatch receipt](docs/breeze-cuda-dispatch-2026-10-12.json).

## TTS sessions: reference ownership, conversion rates and speaker boundaries

Rejected Breeze reference updates preserve the previous reference; replaced Moss
reference resources are released. Melo/OpenVoice conversions use the actual
input rate and return owned PCM at the converter model's output rate. Supertonic
direct speaker getters now safely reject null contexts and invalid indices.
Contributor history from #530 is preserved. All 48 applicable CI checks pass,
one intentional skip; the hosted native tier passes all 2,009 unit tests and
all 140 ONNX-enabled CLI/C ABI capability rows. [The contract receipt](docs/contributor-tts-contracts-2026-10-11.json) preserves the original failure controls and complete ONNX refresh. Separate model-quality reports
in #530 remain unresolved.


The same session integration adds Rust `Session::num_speakers()` and
Supertonic named speaker selection/listing. It routes reference voices for
Breeze, Irodori and Moss-TTS-local, accepts Bark `.npz` speaker files, and
loads Mini-Omni2's SNAC companion through the codec-loading path. Melo can
use an explicitly loaded OpenVoice2 companion for reference-voice conversion.
Reference input rates are selected through the backend contract, including
F5, Pocket and Index; Piper and Parler advertise their native output rates.
Confucius checks required companions before inference, and the VibeVoice
voice-audio override uses the `CRISPASR_VIBEVOICE_VOICE_AUDIO` name.
TADA and Dots advertise their existing voice-cloning capability in the
regenerated catalogue. These are session routing, API and metadata changes;
the contract/capability tests do not establish new speech-quality acceptance
for every named model. Those separate reports remain open.

## Breeze: instructions, reference audio and server voices

Breeze's refreshed package additionally passes all 107 original stage/magnitude
comparisons and all 20 actual-T4 speech/interface roundtrips, with byte-identical
speech against the previous GPU run. Maximum ASR WER is 0.111111; speaker clone
similarity is 0.894997 versus the 0.527019 baseline. See [the refreshed package
receipt](docs/breeze-cuda-integration-package-2026-10-12.json). Canonical complete
BF16 code equality remains 0/8 and is retained as a separate diagnostic;
fixtures, recipes, defaults and its comparator are unchanged. PR #543 merged
as `8dfa0d72cbe5cc24d7853d4631fa758a4134b927` after 51 passing checks and
one intentional skip, preserving contributor history from #529/#536. Instructions
and reference audio now reach synthesis together through the C ABI; server
voice caches include transcript-sidecar changes and reject unknown voices while
preserving subsequent valid requests. See [the merge receipt](docs/breeze-merge-2026-10-12.json).

## CANN: backend/device selection and independent compile validation

The CANN follow-up exposes CLI backend/device selection and adds a separate
opt-in x86_64/910B and arm64/310P compile matrix for CLI, shared library, diff
and quantizer. Ten production-source routing controls pass, with four failing
against the previous helper. Current-source x86_64/910B (ACL graph on) and
ARM64/310P (ACL graph off) builds both pass all four targets and ggml-cann.
See [the complete compile receipt](docs/cann-pr544-compile-2026-10-12.json). PR #544
merged as `a86e366043f4a23251f10b67d982778cd89f83dd` after 46 passing
checks and one intentional skip. See [the merge receipt](docs/cann-merge-2026-10-12.json).
Hosted compilation cannot establish Ascend inference or speed.

The standalone composed-graph probe is now integrated. Its actual CPU graph
passes18 computations over three shapes against an independent integer oracle;
both official310P/910B SDK final executable links,13 general CI jobs and10 lint
checks pass. The standalone static-link repair forwards the configured SDK
library directories. Physical mixed CANN/CPU fallback and ACL execution remain
open. [Complete source-bound proof](docs/cann-composed-preparation-2026-10-15.md).

The operator-validation wrapper is merged on main. Hosted ARM CPU runs one
real positive ADD case and rejects three actual upstream false-success cases:
missing backend, empty parameter selection and unknown operator. It preserves
full logs, hashes and incremental receipts. [The hosted wrapper receipt](docs/cann-hardware-wrapper-2026-10-12.json) records the complete test outputs. The [physical-device procedure](docs/cann-hardware-validation-2026-10-12.md)
has returned physical 310P3 results (SDK 9.1.0, ARM64, ACL graphs off).
RMS_NORM 51/51, SOFT_MAX 107/107 and ROPE 109/109 pass, with unsupported
cases retained. Restricted F32 ADD and F32/F16-weight matrix controls pass;
the original broad ADD/MUL_MAT suites aborted on unsupported BF16 inputs.
The integrated repair rerun passes all 924 executed cases with unsupported
cases retained; both SDK architectures and all 13 general CI jobs pass.
Full-model speech and speed are unvalidated.
See [the complete hardware receipt](docs/cann-310p3-hardware-2026-10-13.json).

## Dia: original output-protocol corrections

The shared CLI/C-ABI runtime now follows the original EOS/capacity countdown,
excludes the final stopping row from the returned slice, and replaces invalid
codec indices with zero while preserving frame timing. Short caps and forced
diagnostic modes retain their original behavior. All16 original-Python witness
cases pass; old source fails12, and restoring each defect independently rejects.
Actual retained CLI/C-ABI arrays also match the original protocol. All13 general
CI jobs and10 lint checks pass on the exact tested candidate; rebase preserves
its source bytes. [Complete proof](docs/dia-output-protocol-2026-10-15.md).

The public pinned-original dumper and text-only `crispasr-diff` route now cover
all encoder/decoder layers, selected attention/block boundaries and real DAC.
Independent composed validation checks1735 payloads and88 stages in each of
the ordinary and serial-head paths, with exact bounded OFF/ON/OFF controls.
[Complete wiring proof](docs/dia-reference-wire-composed-2026-10-16.json).

Disabled-by-default CPU diagnostic controls now compose generic serial head
arithmetic, F32 cross-cache and portable original-style sampling. On the fixed
reported prompt/seed, this combination restores all1226 original input rows,
1210 code frames and WER0 through native DAC, original DAC and actual CLI.
Independent audit verifies535 payloads and every native/original sampling
choice. The F16 first fork is traced to a nucleus rank-boundary crossing on
actual logits; a model-free replay reproduces all27 decisions. Four expanded
prompt/seed controls match original input/code histories exactly and pass352
stage comparisons, but all original/native speech recognitions fail. Crossed
old/current recognizers both pass legacy/JFK and fail those four original clips;
all12 replay cells are independently audited. Complete failed evidence is
permanently public with verified anonymous hashes. Direct original generation
without added observers remains pending. [Broader results and complete proof](docs/dia-broader-speech-results-2026-10-16.md). Production defaults, model pins, GPU/Q4 and performance
acceptance remain unchanged. [Results and public proofs](docs/dia-cache-sampler-results-2026-10-16.md).

## Dia: reproducible DAC precision conversion

The explicit `models/convert-dia-dac-to-gguf.py` tool exports the critical
original F32 output weights into the pinned published DAC layout. The narrow
candidate adds **1,312 bytes** and preserves all 131 unrelated tensor payloads.
It checks the published template before importing model dependencies, verifies
the original checkpoint, and writes a complete 134-tensor physical audit.

[Conversion instructions](docs/dia-dac-conversion.md) explain the pinned
inputs, CLI, output hash and validation scope. Seven tiny original-module/CLI
contracts pass; the unchanged export implementation already reproduces the
candidate byte for byte in [hosted validation](https://github.com/CrispStrobe/CrispASR/actions/runs/38306172119),
passing 48/48 startup and 40/40 full-history comparisons against original
references. A dedicated workflow checks real conversion bytes and every tensor
audit row whenever the converter changes.

This converter is integrated independently of the opt-in runtime controls.
Registry companion pins and broad native/GPU/Q4 acceptance are unchanged. Fresh [integration CI](docs/dia-dac-converter-hosted-2026-10-13.json) also passes real conversion, exact candidate bytes and all 134 physical tensor rows.

## Backend discovery and CI coverage

Generated capability tables and both feature matrices now include the complete
140-backend catalogue in an ONNX-enabled build, including the optional ONNX
adapters. The CI native tier installs the pinned ONNX SDK before regeneration
and compares actual CLI and C ABI capabilities. An earlier 136-backend build
omitted optional ONNX coverage; that output is retained in history and superseded
by the complete verified regeneration.

The CANN SDK jobs are manual and separate from the ordinary platform matrix.
The operator-validation tool requires the requested backend to finish, every
requested operator to execute at least one case, and actual passing counts.
An upstream zero exit with no tested operator is rejected. Unsupported cases
remain visible in the complete logs. A physical Ascend operator pass would
still require separate full-model and quantized speech validation.

## Validation and evidence

| Integration | Completed evidence | Scope still unproven |
|---|---|---|
| Dolphin #541 | 64 stage comparisons, 128 complete arrays, exact original CTC transcripts, automatic C ABI/Python sessions; 45 passing checks, one intentional skip | Reporter-specific Kylin/GCC9/private clip and GPU inference |
| Breeze GPU initialization #542 | 36 passing checks, two intentional skips; identical initialization prefix exercised on real T4 in the validated composition | Full standalone-binary GPU equivalence |
| TTS contracts #531 | 2,009 native unit tests, 140 actual ONNX-enabled CLI/C ABI rows; 48 passing checks, one intentional skip | Separate model-quality reports from #530 |
| Breeze synthesis #543 | 107 original stage/magnitude comparisons, 20 real-T4 decoded speech/interface outputs; 51 passing checks, one intentional skip | Canonical BF16 complete greedy code equality remains 0/8 |
| CANN #544 | x86_64/910B ACL-graph-on and ARM64/310P ACL-graph-off compilation of CLI, shared library, diff, quantizer and ggml-cann; 46 passing checks, one intentional skip | Physical Ascend inference, quantized model quality and speed |
| Operator wrapper | Hosted ARM false-success guards; physical 310P3 restricted float controls and RMS_NORM/SOFT_MAX/ROPE pass, with complete failed broad suites retained | BF16 capability repair/rerun; full-model CANN quality and speed |

The refreshed Breeze GPU runtime files are byte-identical to the merged files;
[the merge receipt](docs/breeze-merge-2026-10-12.json) records their full SHA256
hashes. The complete immutable GPU proof is
[publicly downloadable](https://huggingface.co/datasets/cstr/crispasr-regression-fixtures/resolve/80b1a27bd5dee83a2212a706d1068a23763cdd2e/breeze-guided-20261012/cuda-speech-stage-refreshed-integration-proof.zip):
14,054,644 bytes, SHA256
`ce4baba7e1f03815224043f73dfd6fb47653255fd7088f27234fad1224c604d7`.
The archive was downloaded anonymously and verified in full. It retains the
failed canonical BF16 code diagnostic along with the passing numerical and
speech controls; the worker's final error is not relabelled as a successful run.

Breeze's maximum ASR WER across the 20 outputs is 0.111111. The independently
measured speaker-clone similarity is 0.894997 against a 0.527019 baseline. These
are fixed model/recipe controls rather than a broad benchmark. Seven compiled
server cache-policy cases separately cover transcript-sidecar changes; actual
server requests cover voice speech, reuse, unknown-voice rejection and recovery.
The policy controls are not described as model-executed sidecar tests.

Integrated main1c33cb58 now passes all13 general CI jobs,2,026 unit tests and
all10 lint checks. Bounded frozen-source package/wheel preflight already passes at998f8d18
([receipt](docs/release-preflight-2026-10-15.md)); the final release tip still
requires its own freeze, CI and tagged packaging. Individual passing PR checks do
not establish that v0.8.44 binary assets or wheels have already been built or
published. Version propagation and the annotated tag must use
`scripts/bump-version.sh` after final-main CI passes.

## Known limitations and excluded investigations

Dia production defaults and Q4 remain unaccepted for the reported short
sentence. The scoped opt-in CPU recovery above does not establish broader
texts/seeds, GPU or speed. Startup DAC precision needs the explicit converter;
production companion pins are unchanged. Earlier rejected cache, rotary,
precision and speech controls remain recorded in [HISTORY.md](HISTORY.md) and
the linked complete immutable proofs; they are not promoted by the new result.

The complete MiMo PR #492 and OmniVoice draft PR #521 remain unmerged.
Independent MiMo mel/cached-decode pieces and OmniVoice standalone audio/text
utilities are integrated. OmniVoice utility replay checks all 1,601 hosted
arrays, 738 waveform calls and 18 text cases, with terminal heavy/lint checks
passing; no production caller, preprocessing/output default or setter ABI changes. MiMo's diagnostic
F32 precision path reproduces the original codes and decoded speech but is
slower than its baseline; this does not justify a default change or complete
CANN acceptance. OmniVoice's fixed clone control also fails on the original
Python implementation, which explains that control without establishing full
native/original parity or passing the clone acceptance gate.

Echo issue #485 remains open: tested Q4_K candidates, including calibrated
variants, fail actual-T4 acceptance. Existing accepted 2B Q8/9B F16 models are
not replaced by those candidates. Canary issue #532 is not reproduced by the
completed CPU/T4 allocation/reuse controls; the reporter's private clip/device
case remains unresolved. GTX1660/MX150 (#483) and Intel Arc performance (#461)
need their actual hardware. Nyra (#456) still needs its gated model and the
missing runtime port; it is not solely a hardware blocker.

Separate model-quality reports from #530 remain unresolved even though its
session changes are merged. Full-inference F32 original Breeze controls match
5/8 CPU and 5/8 CUDA complete code grids, while canonical BF16 remains 0/8.
Those controls are documented in
[the full-inference receipt](docs/breeze-guided-full-inference-arithmetic-2026-10-12.json)
and do not replace canonical fixtures or waive the recorded diagnostic.

[Durable Dia proof retrieval](docs/dia-proof-archives.md) preserves canonical,
component, converter and rejected-speech evidence with immutable archive and
payload hashes. Original-head, cache and random-sampling investigations are
archived in HISTORY.md. Current acceptance is the scoped three-control CPU
result above; failed earlier controls retain their original gates.

The scoped 310P BF16 capability candidate also passes both SDK9.2 compile
architectures, including the upstream operator binary. Its physical rerun
passes 924 cases and the production pin uses the exact validated repair after
all 13 composed CI jobs passed. The first Dia T4 attempt remains retained as a
CPU-routing rejection; the device-request repair is proven by actual version-2
CUDA allocation/splits and all three completed cache controls. Neither result
establishes full-model NPU/CUDA speech or speed acceptance.


The source-faithful rotary-timescale CPU OFF/ON/OFF control is complete and
rejects native speech; the separate actual-GPU control also completes and rejects. Actual-source tiny coefficients at the remaining late boundary
improve from maximum error 1.20e-4 to 5.96e-8 through existing ggml frequency
factors; native speech remains rejected on CPU and actual T4. The auxiliary table is
physically audited and existing coefficient bytes are preserved. Original and
native late-step capture is tested for transparency; the first hosted attempt
is retained as a logging/count rejection, with complete failed evidence and a
corrected retry now passes all repeat/negative diagnostic controls.
The runtime experiments remain unmerged and do not change release
model pins, native quality gates or GPU defaults.


The subsequent physical Ascend310P3/SDK9.1 candidate rerun is verified: all
five broad operator suites complete with 924/924 executed cases passing,
unsupported cases retained, and both previous BF16 aborts resolved. Exact
source/compiler/archive/log provenance is recorded; all 13 composed CI jobs pass and the exact repair is integrated. Operator correctness does not establish model-quality or speed.


Actual T4 CUDA cache controls (2026-10-13) are complete: all 120 canonical
replay stages pass with real scheduler splits and sampled process GPU memory
8,528–8,734 MiB. All six native CLI/C-ABI speech checks still reject with empty
transcripts/WER 1. F16 cache shares 353 original input steps; F32 shares 979
and reaches the same step-978/head-2 divergence as CPU. OFF/repeat full arrays
match exactly. These are correctness diagnostics, not accepted native speech
or performance measurements. [Complete physical CUDA proof](docs/dia-cuda-cache-physical-2026-10-13.json).


The complete CPU/CUDA cache, partial late-layer and causal diagnostic proofs
are preserved as five immutable public archives. All payload and full anonymous
remote size/SHA checks pass; rejected speech remains rejected. [Publication
receipt](docs/dia-cache-public-proof-2026-10-13.json).


The completed original-timescale CPU control passes 120 replay rows but does
not change generated token histories or PCM; all six speech checks remain
empty/WER 1. The +608-byte auxiliary model is a rejected control. The corrected
late-capture experiment independently passes default/978/default complete-array
identity and all three rejection controls. These establish diagnostic fidelity,
not accepted native speech or speed.
[Rotary control](docs/dia-original-rope-hosted-2026-10-13.json), [complete late capture](docs/dia-late-stage-hosted-2026-10-13.json).


The scoped Ascend310P BF16 capability repair is integrated at the exact
physically tested ggml revision. It resolves both broad ADD/MUL_MAT aborts;
924/924 executed physical operator cases pass, with unsupported cases retained.
Both SDK compile architectures and all 13 composed general CI jobs pass.
The shared fork retains contributor history. Full NPU model quality and speed
remain separate validation. [Integration receipt](docs/cann-310p-bf16-integration-2026-10-13.json).


The GPU rotary ON arm shortens matching original input history from 979 to
865 steps; all six speech checks remain WER 1 despite 120 passing replay rows.
Canonical original audio passes the same GPU recognizer at WER 0. Complete
terminal logs and 61 output files are retained and hashed; the experiment stays
unmerged. [Physical rotary receipt](docs/dia-original-rope-cuda-physical-2026-10-13.json).


The exact-source CUDA v3 bundle is fully audited (27 members and seven source
files); its actual T4 speech rejection is retained separately. A weight-free
RMSNorm audit on all retained decoder boundaries isolates F32 versus double-sum
rounding in nine of 36 comparisons, with no runtime/default change or speed
claim. [CUDA package](docs/dia-cuda-package-v3-2026-10-13.json), [RMSNorm primitive](docs/dia-rmsnorm-primitive-2026-10-13.json).


The exact-shape output-head probe reproduces every retained native step-978
logit byte-for-byte. Combining the heads changes nothing; flattening batch
shape changes rounding but does not restore the differing draw. All 16
original-filter controls choose token 877 with original hidden states and 904
with native hidden states. The cause at this boundary precedes the output
head. This is isolated numerical evidence, not accepted speech or speed; the
local original projection is not byte-identical to the hosted original one.
[Projection audit](docs/dia-head-projection-isolation-2026-10-13.json).


The learned final norm is byte-identical on retained original/native layer-17
inputs and reproduces both saved hidden states exactly. Decoder embedding
captures also match exactly at four checkpoints, including step 978. Repeats
and changed-feedback/scaled-output negatives pass. The remaining hidden
variation is inside the decoder blocks or their attention inputs; these
isolated controls do not accept native speech, Q4 or speed.
[Norm/input audit](docs/dia-norm-embedding-isolation-2026-10-13.json).


Block-0 pre-attention norm is byte-identical on all four verified embedding
inputs. Q/K/V projections introduce small rounding differences (minimum cosine
0.9999999999999603, maximum row-norm error 9.19345e-8). Flattened branch batches
worsen relative L2 in all 12 controlled cases, so no runtime candidate is
introduced. Repeats and a magnitude negative pass. This is primitive isolation,
with native speech still unaccepted.
[Block-0 audit](docs/dia-block0-projection-isolation-2026-10-13.json).


The actual Q/K rotary replay confirms local error on audited first-block
projections. At step 978, composed relative L2 falls from 5.37020e-5 to
1.57296e-7 for Q and 3.55324e-5 to 2.28756e-7 for K using original timescales.
All 48 comparisons, repeats and eight negative controls are retained. This is
CPU primitive evidence; full-model source-timescale speech remains rejected,
with no runtime/default promotion or performance claim.
[Rotary audit](docs/dia-block0-rotary-isolation-2026-10-13.json).


A bounded actual-source block-zero self-attention replay on canonical steps
0–3 verifies byte-identical same-input K/V cache histories. Attention minimum
head cosine is 0.9999999999999742 and maximum norm error 1.46145e-7. Missing
cache reorder, tiled GQA heads, wrong sqrt-head scaling and doubled output all
strongly reject. New step-1/3 norm rounding differences and the stopped initial
expectation are retained. This does not establish late-history, higher-layer,
GPU or full speech acceptance; production defaults remain unchanged.
[Attention/cache audit](docs/dia-block0-attention-isolation-2026-10-13.json).

The pinned-config 3072-slot original cache also reproduces all short-prefix
Q/K/V/attention outputs and cache history bytes exactly, with exact repeats.
This checks production cache stride, without claiming late-history acceptance.

Late-history self-attention capture tooling is implemented on diagnostic source
42b111c5e. Tiny actual-original two-layer and compiled replay guards pass; GH
38438623888 queues full-model capture-transparency controls and step-978 snapshots
for all 18 layers. Production main receives tracking only. This is pending
numerical evidence, not a runtime fix, speech acceptance or performance result.
[Capture protocol](docs/dia-attention-history-capture-2026-10-13.md).

The weight-free same-input attention consumer reproduces audited prefix 0–3
original snapshots with production cache stride and native snapshots at one/four
CPU threads, byte-for-byte. Its 24 comparison rows and 12 strong negatives are
retained. Hosted snapshots require verified file hashes and capture transparency;
late-history results, cross-host identity and native speech remain unaccepted.
[Same-input replay audit](docs/dia-attention-input-replay-2026-10-13.json).

The matching CUDA 12.4 SM60/75 capture package is queued on GH at source
2328f15ce; all six native runtime/ABI/binding files match CPU capture source
42b111c5e byte-for-byte. Eight source guard tests pass. The CPU capture worker
is now running. Both hosted results and actual GPU/model acceptance remain
pending; production main receives tracking only.
[CUDA package protocol](docs/dia-attention-cuda-package-2026-10-13.md).

### Dia diagnostics: retained CPU/T4 evidence and receipt repair

Preserved and independently verified completed GPU work after a diagnostic
receipt serialization failure: 120 stage comparisons, 72 late self-attention
snapshots, complete OFF/ON/OFF capture transparency and original-audio GPU
ASR WER 0. Fixed integer shape serialization and atomic checkpoint writes;
eleven diagnostic worker guards pass. Complete evidence is public.

Same-input actual-original attention and block-zero cross-K/V replays now
show relative L2 differences below 7.21e-7 on retained CPU/T4 outputs;
scale and layout negatives remain effective. The original GPU job remains
ERROR, and the original reference operators in these bounded replays run CPU.
Native free-running speech and Q4 remain unaccepted. Opt-in decoder
residual/cross-Q/fused-MLP observation on diagnostic source 8d0ad886 now passes
full hosted capture transparency, sixteen actual-original hook tests and 120
ordinary stages in GH 38472367538. Independently verified boundary differences
are small (maximum relative L2 5.204e-6), and complete proof is public. These
rows include input drift. Actual fused-original/split-native MLP replay now
reproduces hosted native stages exactly and differs on matching inputs by at
most relative L2 2.399e-7, with scale/wrong-split controls rejecting. Native
speech/Q4 remain unaccepted. No production default, model or release promotion
follows. See
`docs/dia-cross-cache-input-replay-2026-10-13.md` and
`docs/dia-decoder-boundary-results-2026-10-14.md` and
`docs/dia-mlp-owned-input-results-2026-10-14.md`.

## UPDATE 2026-10-14 — Owned self-output and cross-attention verified

Actual original projections and SDPA/mask/cache expressions versus literal
native CPU graph yield 24 rows and 12 rejecting scale/missing-mask controls.
Minimum same-input cosine 0.9999999999998742, maximum row norm error
3.122e-7, maximum relative L2 2.580e-7. All four native outputs exactly
reproduce hosted native snapshots; repeat/thread controls are exact. Actual
mask is conditional-derived for BOTH CFG branches and permits the 39-token
prefix of the 1,024-token source context. Synthetic finite masked tails
verify invariance only, not original unretained padded caches. Complete
40-payload proof is public and anonymously verified. No runtime math change,
full speech, GPU, Q4, speed or release acceptance. Evidence:
`docs/dia-cross-owned-input-results-2026-10-14.md` and `.json`.

## Ascend 910B3 physical follow-up — 2026-10-14

New current-main/ggml21acf3162 physical 910B3, ARM64, SDK9.1, ACL-graph-on
archive is retained; all 25 member hashes and seven unchanged-wrapper
assessments verify. Broad ADD154, RMS_NORM51, SOFT_MAX107 and MUL_MAT927
pass. RoPE fails 450/452 and repeats the same two F32 NEOX failures.
This is not broad 910B acceptance and does not change the earlier scoped
310P result. Native Git/compiler provenance verifies; the binary hash is
producer-reported because no executable is included. No full-model,
quantized speech or performance proof. Cache-width allocation is a candidate
explanation awaiting regression/device proof. Evidence and next step:
`docs/cann-hardware-validation-2026-10-12.md` and
`docs/cann-910b3-hardware-2026-10-14.json`.

## CANN rotary requested-capacity defect reproduced — 2026-10-14

A bounded pre-fix regression compiles the actual tested allocation block
and cache metadata. Constant-512-position widths64/64/256/128 leave
131,072-byte sin/cos requests when 524,288 and 262,144 bytes are required.
Width growth with fewer positions also fails. Four tests: two expected
failures, two positive controls, complete logs retained. This is allocation
coverage only, not RoPE math/device validation or a repair. Immutable test
source7b77e0b remains diagnostic; production pin/runtime unchanged.
Public 14-payload proof is anonymously verified. Evidence:
`docs/cann-rope-capacity-2026-10-14.md` and `.json`.

## Dia full free-running transparency verified — 2026-10-14

Run38527734866 completes; independent whole-delivery/source/history/PCM/WER
audit passes. Ten unforced arms match full PCM within CLI/C ABI and all
1,115-step logged model histories match across surfaces. Common/extended
step978 marks are transparent for the fixed CPU F32 diagnostic recipe.
Baseline speech remains rejected on both surfaces: empty ASR, WER1.0.
Public raw262-payload and complete audit archives are anonymously verified.
No speech/Q4/GPU/performance/default or release acceptance follows.
Evidence: docs/dia-free-capture-results-2026-10-14.md and .json.

## CANN rotary allocation candidate prepared — 2026-10-14

Shared ggml0a8620765 records sin/cos allocated bytes independently of
last-call dimensions. Four unchanged capacity regressions now pass; old
two failing controls remain public. Thirteen BF16 routing cases per chip
pass. SDK sourceca4c12c4 pins the candidate; run38536737288 is submitted
on 910B/graph-on and 310P/graph-off. SDK result and physical 910B3 rerun
remain pending. Production ggml21acf3162 is unchanged; no numerical
NPU/model/quantized quality/speed or release acceptance. Immutable
fresh-directory commands and public candidate proof:
`docs/cann-rope-capacity-candidate-2026-10-14.md` and .json.

### Additional validation checkpoint — October14

- Physical910B3/SDK9.1 rotary-capacity candidate passes1691/1691 broad
  operator cases and452/452 independent RoPE repeat. Both previously failing
  F32NEOX cases pass, with unchanged tolerances and unsupported cases retained.
  SDK/composed integration gates are pending; candidate is not yet production.
- Dia query/rotary owned-input replay retains14rows/16negative controls and
  complete public proof. Query projections agree closely; native cross query
  reproduces the hosted snapshot exactly. Native speech still fails WER1;
  this diagnosis does not establish speech/Q4/performance acceptance.

- Rotary-capacity candidate now also passes both official SDK9.2 beta2
  architecture builds, including CLI, shared ABI, diff, quantizer and actual
  operator target. General integration CI remains active; production pin
  unchanged. Remaining hardware/model/performance tasks are in docs/cann-todo.md.

### Integrated CANN rotary buffer repair

Fix rotary sin/cos cache allocation when rotation width grows without an
increase in position count, including width growth with fewer positions.
Allocated byte capacity is tracked independently of logical cache shape.
The exact tested ggml pin is integrated after13/13general CI, both official
SDK builds and physical910B3/SDK9.1:1691/1691 broad plus452/452 repeat.
Both prior F32NEOX failures pass at unchanged tolerances. Lightweight
capacity regression runs in both manual SDK jobs. Original failures and
unsupported cases remain retained. Full NPU model/quantized speech,
executed ACL capture/replay and performance require separate proof.

### Strict Moonshine device validation

Opt-in exact backend/device requirements reject missing or invalid devices and
GPU-disabled contradictions instead of silently running on another backend.
Encoder scheduler assignments and successful computation are recorded without
altering the graph. Moonshine diff verifies exact reference PCM, complete
shape/count, row magnitude and relative error alongside cosine, and refuses
missing encoder references. Magnitude failures preserve the actual measured
cosine. Full hosted CPU CLI/auto+explicit C ABI Session checks and13 deliberate
failure controls pass; all13 general CI jobs and2,024 unit tests pass on the
runtime candidate. The final diagnostic row/reporting correction also passes
bounded real-reference controls. Physical NPU, ACL replay, quantized model
quality and performance remain pending; no release is cut by this result.

### Portable full-model CANN validation interface

A standalone F32 Moonshine fixture runner verifies immutable model/reference
bytes, runs genuine CPU controls before requested CANN, and requires repeated
encoder, diff, CLI and auto/explicit C ABI Session output with real selected-
device work. It retains incremental failure receipts and actual binary/library
snapshots. The complete CPU interface and unavailable-CANN/GPU-argument controls
pass in hosted run38602252235, with all171 returned payloads independently audited.
Eight false-success/archive/argument guards are wired into Linux CI. Diagnostic
encoder capture now explicitly forwards GPU selection; defaults preserve CPU.
Both official SDK architectures link all6 requested targets in38598516003.
Compiler cache/command artifact retention is added; execution follow-up38605386237 passes both chips; actual
compiler chip/ACL definitions and all six links per chip are independently audited. These results establish preparation and CPU controls; physical NPU,
ACL replay, quantized-model quality and performance remain separate gates.


## Integrated Dia top-k correctness repair (2026-10-15)

The shared sampler now selects exactly k temperature-scaled logit indices
instead of a rounded-probability threshold. The actual old header loses a
unique top-k1 maximum in34/64 trials; the corrected sampler loses0/64 and matches four
pinned original Python corner fixtures. All18,288 captured probability/sample
rows remain byte-exact to baseline. All13 candidate CI jobs,10 lint checks and the secret scan pass; Linux unit
testing passes2,026 tests. Header/test bytes are unchanged after rebase. No speech repair,
Q4/GPU/performance or release acceptance is inferred. Details and public proof:
[Dia top-k selection](docs/dia-topk-selection-2026-10-14.md).

## Linux packaging preflight

The bounded accepted-source Linux CLI, shared library and installed wheel pass
complete archive/dependency and relocated startup checks. The installed wheel
finds its own packaged native/helper libraries and enumerates116 C-ABI backends
from an unrelated directory without loader overrides. This validates source
998f8d18 at version0.8.43; final release assets still require the final frozen
source, version bump, applicable CI and platform matrix.

## Dia diagnostic provenance

Completed reduction, query/key, cache-history, terminal and reciprocal replay
investigations are archived in [HISTORY.md](HISTORY.md#completed-dia-release-note-investigations-archived--2026-10-16).
The accepted current-runtime scope, broader-test dependency and immutable proof
links are described in the Dia sections above. Earlier rejected source pins
retain their original outcomes and are not current-main acceptance statements.
