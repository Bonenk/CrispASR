# Dia full-decoder layer0 key intervention preparation

Candidate `2f04f4e6f64ab3aa607fe055b9e8c39a8b1a87ea` adds a diagnostic worker
and small runner. Production runtime/defaults remain unchanged. The worker
patches only its scratch checkout at the immutable previously tested boundary
runtime `8d0ad8862d2cad5d305a48a57f2a99b5024bc39a`.

The [GH full-model run38726696058](https://github.com/CrispStrobe/CrispASR/actions/runs/38726696058)
and [lint38726699171](https://github.com/CrispStrobe/CrispASR/actions/runs/38726699171)
are submitted and queued. No numerical intervention result is available yet.
The [method](https://github.com/CrispStrobe/CrispASR/blob/2f04f4e6f64ab3aa607fe055b9e8c39a8b1a87ea/tools/ci-heavy/dia-cache-intervention/README.md)
records the exact dispatch and predefined gates.

Local axis-ownership guards pass, including wrong shape/dtype/nonfinite
rejections. The actual compiled C++ file loader accepts valid keys and rejects
missing, truncated, extra-length and nonfinite files. The patch applies to the
exact pinned runtime, and the full patched translation unit passes a C++17
syntax check against its actual headers. All four repository secret guards
pass unskipped. These are preparation checks, not hosted/model acceptance.
The actual worker sampling observer also passes on pinned Torch2.6.0+cpu:
original/native support sets exactly match the retained first-divergence proof,
with46/904 ranks32/33 reversed. Original CFG margin is+4.6253204e-5; native
is−2.6702881e-5. This independently checks the measurement path without a new
full-model inference.


The [immutable input ZIP](https://huggingface.co/datasets/cstr/crispasr-regression-fixtures/resolve/5e2fc2430413357a21807cd0350960aeeb6fe183/dia-20261015-cache-intervention/fixed-history-inputs-v1.zip)
contains31 payloads,138612077 compressed bytes, SHA256
`a0c7e2e876394469e07a455095cf79f6e004c1d2087f3ce4fb671dedbb567d52`.
An anonymous complete download matches that SHA. It retains979 shared inputs,
complete original/native logits, original endpoint stages and captured original/
native layer0 keys with source/hash manifests and license notices. No weights
or newly reconstructed keys replace these captured inputs.

The baseline must reproduce every retained native input and all979 native
logit rows byte-for-byte. A second native-key replacement arm must preserve
all logits and captured stages, controlling the new graph input. Only then
interpret original prefix0..977, original current key978, and combined arms.
The actual consumed keys must match each requested array exactly. Earlier
logits, layer0 input/norm/Q/V and cross inputs must remain unchanged; downstream
current activations may respond naturally. Past histories stay native.

Compare all18 decoder outputs, final hidden/logits with cosine, both norms and
relative L2, plus actual original Torch2.6 filtering support and head2 CFG46/904
margin/ranks. Keep all arms and failures. This key-only intervention cannot
settle correlated joint Q/K phase effects. Null PCM is deliberate without DAC;
runner exit alone is insufficient. No speech, GPU, Q4, performance or default
promotion follows from this diagnostic. The full model runs on GH, with all
local evidence under /mnt/storage/crispasr/dia-cache-intervention-20261015.
