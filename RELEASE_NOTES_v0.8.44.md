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

This tooling is independent of the unmerged Dia runtime investigation. Native
short-speech and Q4 gates remain failed, and registry companion pins are
unchanged. Fresh [integration CI](docs/dia-dac-converter-hosted-2026-10-13.json) also passes real conversion, exact candidate bytes and all 134 physical tensor rows.

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

Final integrated-main CI and the subsequent tagged packaging jobs remain release
gates. Individual passing PR checks establish the scope shown above; they do
not establish that v0.8.44 binary assets or wheels have already been built or
published. Version propagation and the annotated tag must use
`scripts/bump-version.sh` after final-main CI passes.

## Known limitations and excluded investigations

Dia short-text/delay-reversal and reference-harness changes remain on an
unmerged branch. F16 neural stages pass the pinned original reference, but the
128-step startup DAC replay fails cosine/magnitude (cosine 0.9976444, norm error
0.04271824). Q4_K also fails neural boundaries. Complete native long-prompt
roundtrips pass, while the reported short sentence has WER 1.0 through both
CLI and C ABI: F16/Q8 are quiet and Q4 produces louder non-speech. Eight-stage
DAC tracing now isolates a final-weight precision defect. The completed canonical original
short-speech control reads the sentence verbatim (WER 0, 14.048 seconds,
RMS 0.073534). Replaying that full original history passes all 40 F16
comparisons, including DAC PCM at cosine 0.9999999 and norm error 0.00004443983.
Q4_K passes only 13/40 and first fails at encoder layer 6. This distinct
full-utterance result neither erases the startup-codec failure nor accepts
native free generation. [The complete original receipt](docs/dia-full-original-short-2026-10-12.json)
preserves full stage metrics and all 21 artifact hashes. A separately labelled
original-model control using the exact compiled native sampler also fails
short speech (WER 1), as does native free generation. Its 37-step identical
input prefix passes neural cosine/magnitude; [the sampler receipt](docs/dia-native-sampler-control-2026-10-12.json)
retains the first discrete divergence and complete proof. This implicates the
sampling path. Pinned probability replay finds matching support and maximum probability error below 3.61e-7;
these are excluded investigations. Thresholds, reference PCM and production
recipes remain unchanged. The
[complete rejected proof](https://huggingface.co/datasets/cstr/crispasr-regression-fixtures/resolve/43b643898356a14c9d7a8aa0168ef5ea0d64261d/dia-20261012/rejected-codec-and-short-speech-proof.zip)
and [receipt](docs/dia-rejected-proof-2026-10-12.json) preserve both failures.

The completed eight-boundary diagnostic isolates the startup codec failure to
the precision of the final output weights. Preserving three original F32
tensors passes all 48 canonical startup comparisons and adds only 1,312 bytes.
[The full codec receipt](docs/dia-codec-precision-2026-10-12.json) retains all
variants, rejected published results, four negative controls and 47 artifact
hashes. Standalone conversion now reproduces the validated bytes and passes fresh startup/full-history comparisons. Native free speech remains excluded from release acceptance; production codec pins are unchanged.

MiMo PR #492 and OmniVoice draft PR #521 remain unmerged. MiMo's diagnostic
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

Dia's unmerged [portable sampler replay](docs/dia-portable-random-hosted-2026-10-13.json) reproduces all 11,034 original draws and 1,210 codec frames with native filtering. In the [real speech control](docs/dia-portable-speech-hosted-2026-10-13.json), the original neural model with this sampler produces byte-identical canonical PCM and WER 0, while native F16 CLI/C-ABI speech still fails with empty transcripts. The first differing draw follows a nucleus-support boundary change despite close logits. [Completed original-head/full-F32 controls](docs/dia-neural-precision-hosted-2026-10-13.json) both pass 40/40 replay stages but reject real CLI/C-ABI speech with empty transcripts and WER 1. Full F32 extends exact shared input history to 865 steps; [Opt-in cross-cache precision control](docs/dia-cross-cache-controls-2026-10-13.json) is complete and rejected, preserving fixed F32 weights and same-host OFF/ON/OFF guards. The runtime stays opt-in and unmerged; production defaults/model pins and speech/Q4/GPU gates remain unchanged.

[Durable Dia proof retrieval](docs/dia-proof-archives.md) preserves all 77 canonical/component/converter/rejected-speech files in immutable public archives. The stdlib helper verifies full archive and payload hashes, rejects invalid extraction and preserves existing paths; eight contracts and real archive retrieval pass. Future validation input transport also passes complete [original/native-filter replay](docs/dia-public-input-replay-2026-10-13.json). These are reproducibility fixes, with native speech still rejected and cache precision under investigation.

[Complete rejected neural precision and tiny cache proofs](docs/dia-neural-public-proof-2026-10-13.json) are also durably public, with full anonymous archive verification and unchanged acceptance scopes. The TTS documentation records the short-prompt failure; the architecture guide corrects the original checkpoint’s CFG equation.


Completed fixed-cache control (GH 38357576078, source 2205d920d): all three
F16/F32/F16-repeat arms pass 40/40 canonical stages, but all six CLI/C ABI
speech controls remain empty/WER 1. OFF/repeat inputs, logits and PCM match
exactly. Actual cross-cache vectors double from 11,501,568 to 23,003,136 bytes;
F32 extends shared input history from 865 to 979 steps and reduces maximum
aligned logit norm error from 2.40e-5 to 5.01e-6. At step 978/head 2, tiny neural
changes include token 904 across the nucleus boundary; original/native filters
agree on identical logits and reproduce the recorded draws. This is a rejected
speech control, with no sampling-policy or quality-gate waiver.
Complete [cache controls](docs/dia-cross-cache-hosted-2026-10-13.json) and [first-draw replay](docs/dia-cross-cache-boundary-2026-10-13.json).


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
