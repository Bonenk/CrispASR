# Dia current-runtime cache/sampler factorial — 2026-10-16

The disabled-by-default controls are integrated at ad0d1546 with executable
source bytes exact to tested5e04955e. Heavy38892690744, all13 CI and10 lint
jobs pass. Independent v3 verifies535 payloads and all30 initial stage metrics,
actual original/native DAC provenance, CLI/Session equality and ASR results.

Holding generic serial-head arithmetic fixed, the original prompt and seed42:

| Cross cache | Sampler | First differing original input | Frames | Native/original DAC/CLI WER |
|---|---|---:|---:|---|
| F16 | Standard | 1 | 1423 | 1 / 1 / 1 |
| F16 | Portable | 865 | 1099 | 1 / 1 / 1 |
| F32 | Standard | 1 | 1423 | 1 / 1 / 1 |
| F32 | Portable | None: all1226 rows exact | 1210 | 0 / 0 / 0 |

The passing arm restores all original code frames. Actual transcript from
both DACs and the CLI: "We are committed to making it secure." Native PCM RMS
is0.0735281; failed controls are around0.00016-0.00018 with empty transcripts.
Native decode replay is exact, and CLI/Session inputs/logits are byte-identical.
Original is the positive control. Pristine/OFF/identity controls remain exact.

Re-executing the actual original Python sampler on the actual native logits
reproduces every10035 F16-portable and11034 F32-portable raw head decision.
Thus F16's later divergence is neural numerical drift, not a sampler mismatch.
The different standard sampling recipe explains the first feedback difference
on original as well as native logits. The earlier head-reduction near-tie repair
remains a distinct required part of this scoped passing recipe.

Auditor v1 caught the producer excluding a nested tiny-test receipt from its
inventory. v2 encountered CIFS ENOMEM while re-extracting the same official ZIP.
v3 reuses the complete v1 extraction only after exact ZIP/local file-set and
size checks; verifies every indexed SHA, the extra receipt against fresh ZIP
bytes and its complete tiny test arrays/state, then all scientific results.
All failed logs and extraction attempts are preserved. No quality gate was
relaxed. The raw official ZIP SHA256 is
`d4753346f6f98088e28d098fe014c12b960a05ff8acbbed1648d90b7854ad25c`.

This accepts one CPU prompt/seed with opt-in serial head, F32 cross-cache and
portable RNG. Production defaults remain unchanged. Additional texts/seeds,
GPU/Q4 and profiling are separate tasks in PLAN.md. [Full audit](dia-cache-sampler-results-2026-10-16.json).
