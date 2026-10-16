# Dia reciprocal native sampled-decision fork — preparation

Candidateeae36591 is pushed on diag/dia-native-fork-20261016. Heavy CPU
[38851402907](https://github.com/CrispStrobe/CrispASR/actions/runs/38851402907)
and lint [38851403072](https://github.com/CrispStrobe/CrispASR/actions/runs/38851403072)
are dispatched; this checkpoint is not a hosted result. The owned worktree is
/mnt/volume1/crispasr-dia-native-fork-20261016, with local proof/scratch under
/mnt/storage/crispasr/dia-native-fork-20261016/local-v1. The detached pinned
syntax-check worktree /mnt/volume1/crispasr-dia-native-fork-runtime-20261016
contains only this experiment's scratch source/header patch.

The [accepted original-only fork](dia-original-trace-results-2026-10-16.md)
shows that replacing sampled877→904 at978/codebook2 destroys speech for this
prompt/seed in the original itself. The reciprocal test asks whether changing
only native904→877 restores speech in the fixed native trajectory. Either
outcome is retained; a successful token override is not a production repair.

The worker uses immutable8d0ad886/C36, the same physically verified full F32
neural weights, F32 cross caches, portable RNG and source-RoPE-off configuration
as the accepted native boundary. Prompt, seed42,3072-token capacity, temperature,
top_p/top_k and CFG remain unchanged. It runs the actual native C ABI and DAC in
four separate processes: pristine, observer-only OFF, identity904, restore877.
Pristine generation occurs before the scratch patch and rebuild. Original
sampling/neural code remains unchanged; the patch observes or substitutes only
the completed nine-head sample at the fixed boundary, then saves raw code grids.

Pristine/OFF/identity complete inputs, logits, position978 captures and native
PCM must be byte-exact. OFF/identity full codes, witnesses and final serialized
MT state must also be exact. Restore must preserve all earlier inputs/logits,
all position978 stages and the safe963-frame code prefix; input979 differs in
exactly one entry and matches the original row. Typed and serialized RNG states
before/after the substitution must be identical, with the same pre-fork state
in all instrumented arms. A separate immutable accepted native archive must
reproduce the observed prefix logits/stages; no prefix import or teacher forcing.

Actual native audio is recognized for all four arms. Instrumented arms' complete
code grids are also decoded with the actual pinned original DAC and recognized.
Native8d0ad886's historical stopping/reversal conventions remain fixed; no
unreported output-protocol repair is mixed into the experiment. Original
canonical codes/PCM independently validate DAC reconstruction with relative
L2 below1e-4; both original ASR positives require WER0. All ASR results use the
same final recognizer library and pinned model. Native/fork WER has no relaxed
or newly selected pass threshold: report it honestly.

Local preflight compiles the actual pinned sampler and exponential generator,
then checks OFF/identity/restore tokens, full MT-state equality and9 rejecting
controls. Repeated patching also rejects. The complete patched runtime passes
C++ syntax checking against exact8d0ad886/C36 headers. Clang-format18, Python
compilation and all4 unskipped no-secrets checks pass. These are tiny/control
checks; full model/build/codec/ASR work remains on GH CPU. No local full neural
model is loaded. Source/ELF/library/export receipts, complete arrays and original
DAC source/provenance are retained for independent artifact audit.

No production runtime/default, model pin, sampler filter, seed or acceptance
threshold changes. Native speech, Q4/GPU/performance and release promotion
remain unaccepted. The full worker method is retained on the candidate branch
at tools/ci-heavy/dia-native-fork/README.md until hosted evidence is audited.
