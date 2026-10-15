# Dia one-step suffix replay preparation — 2026-10-15

The candidate is prepared and submitted; full-model numerical gates are pending.
No production runtime or default changes are proposed by this diagnostic.

- Immutable candidate: [70f2d9de678b31f6b55f957e3a721aa91bf0af88](https://github.com/CrispStrobe/CrispASR/commit/70f2d9de678b31f6b55f957e3a721aa91bf0af88).
- [Full worker/method](https://github.com/CrispStrobe/CrispASR/blob/70f2d9de678b31f6b55f957e3a721aa91bf0af88/tools/ci-heavy/dia-suffix-replay/README.md).
- [Hosted numerical job38793672943](https://github.com/CrispStrobe/CrispASR/actions/runs/38793672943): queued at checkpoint.
- [Lint38793677203](https://github.com/CrispStrobe/CrispASR/actions/runs/38793677203): queued at checkpoint.

The pinned runtime8d0ad886/ggmlC36 imports native self-cache positions0..977
from the independently audited full-decoder Q/K artifact. It computes only
position978 with the original forced token row and native encoder/cross producer.
Baseline must reproduce all127 captured stages and final logits byte-for-byte;
all18 native post-layer replacement controls must pass the same gate before
all18 original captured post-layer replacements run. Each original arm changes
one activation independently while holding the historical self prefixes native.

Both computed and consumed activations remain actual graph outputs. An explicit
root preserves the replaced producer; the replacement input is retained by
pointer after downstream naming. Every arm captures all18 cross K/V pairs and
must match the baseline bytes. Earlier layers and current-layer self Q/K/V must
stay exact; all18 stage cosines/norms/errors, hidden and logits plus original
Torch2.6 sampler margin/ranks/support are reported. Original sampler source and
actual full-model original activation captures come from the verified prior
artifact. The full original model is not instantiated.

Large captures retain complete SHA/byte identities with a lossless encoding:
unchanged files reference baseline bytes; changed self K/V stores only the
current2×512 row after asserting all978 past rows exact. Reconstructing the full
array must reproduce its original SHA before the full redundant file is removed.
The baseline, current rows, references and complete reconstruction recipe remain
in the output. All cross caches retain one baseline plus per-arm exact hashes.

Local preparation passes the actual ggml prefix and graph ownership guard,
finite exact-size loader rejection controls, lossless cache reconstruction with
changed-prefix rejection, ZIP path/hash controls and single-use pinned patch.
The complete patched runtime passes C++17 syntax against exactC36 headers;
format18 and all4 unskipped secret guards pass. Local command receipts and logs
are under `/mnt/storage/crispasr/dia-suffix-20261015/local-v1/`, including
`local-validation.json`. These are source/component controls, not new numerical
or decoded-output evidence. The candidate branch stays separate until hosted
validation and independent artifact audit complete.

Skipping978 iterations does not reproduce prefix RNG state. The worker observes
the actual original filter before multinomial drawing; it cannot claim a sampled
trajectory, speech repair, Q4, performance or GPU acceptance. Missing simultaneous
full-model original self-Q capture remains a separate follow-up. The earlier
[Q/K factorial result](dia-query-key-results-2026-10-15.md) remains definitive
for that unchanged experiment; do not repeat it while waiting for this new probe.
