# Dia complete native-code reconstruction and codec probe

The final emitted native row is now recoverable from retained raw sampler
predictions and the actual producer stopping/BOS logic. Complete CLI/C-ABI
code grids agree. Original-DAC decoding/ASR and independent audit are now
complete locally. Native speech still fails; the unchanged hosted job is queued.

## Complete reconstruction

The source from native run38527734866, `eb94d724`, is verified against its
immutable Git object. The literal stopping/reversal functions and literal
BOS/output-update blocks compile in a small standalone replay adapter. Every
one of1115 captured feedback rows matches; the actual first EOS is step1100,
with the final stop at1115. The resulting1100 aligned frames predict563200
PCM samples, exactly matching the retained float C-ABI and CLI sample counts.
The CLI/C-ABI aligned grids are byte-identical. A deliberately changed raw
prediction rejects at the next observed feedback row. No stopping row is
invented, and no neural model is rerun.

Selected raw inputs, sampler records, PCM, logs, native source and original
reference are rehashed against their previously accepted delivery ledgers.
The99,803,232-byte original GGUF is rehashed in full. Its original1210-frame
code grid and619520-sample PCM are retained as positive controls; the latter
is byte-identical to the separately delivered original waveform. Pinned
original source files match their recorded hashes at `4a9e29b1`.

## Actual original return-slice comparison

On Torch2.6.0+cpu, the literal original `_prepare_audio_prompt` and `generate`
methods run with the actual original `DecoderOutput` class. Neural prediction
is replaced with the retained native sampler rows; every original feedback
input matches native exactly. This tests the generation protocol, not neural
math or a new original-model generation.

| Same retained predictions | Native | Original return slice |
| --- | ---: | ---: |
| Feedback steps consumed | 1115 | 1115 |
| Returned delayed rows | 1115 | 1114 |
| Aligned code frames | 1100 | 1099 |
| Expected PCM samples (512/frame) | 563200 | 562688 |

All1114 shared emitted rows match exactly. Native retains one extra final row;
original `generated_tokens[prefill_step:dec_step+1]` excludes the newly written
stopping row. This is a measured11.61ms tail mismatch. A repair must preserve
forced-input/output diagnostics and cover natural EOS, capacity and short runs;
it is a separate runtime task. Existing free-speech failure remains unaccepted.

## Hosted original DAC/ASR test

Candidate `7b42071f41e5814f957d7397d8c73615b2cd7c9a` adds a diagnostic worker
without production changes. [Hosted run38710473906](https://github.com/CrispStrobe/CrispASR/actions/runs/38710473906)
recompiles/replays the native adapter, verifies original sources and replays the
original protocol. It downloads the hash-pinned actual DAC0.0.1 checkpoint,
checks the three actual DAC neural source files against the retained originals,
and decodes native complete codes, the original return slice and original codes.

The original positive PCM requires cosine≥0.999999, relative L2≤1e-4 and norm
error≤1e-4. The same original Whisper tiny recipe checks every decoded waveform;
the original-code positive control must retain zero WER. Complete arrays,
float WAVs, ASR inputs/results, source, binaries and receipts survive failures.
No generation seed, top-p/top-k, model/default, GPU or timing promotion follows.
The local metric controls reject scale/order/shape/nonfinite/zero-norm errors.

## Complete original-DAC diagnosis

The unchanged worker at7b42071f completed on the local VPS with isolated
Torch2.6.0+cpu, DAC1.0.0, audiotools0.7.2 and Whisper20250625. All models,
installed reference dependencies and outputs are under /mnt/storage. This is a
local CPU result; the queued hosted job is not represented as completed.

| Codes decoded by actual original DAC | Samples | RMS | ASR WER |
| --- | ---: | ---: | ---: |
| Original positive | 619520 | 0.0735338684 | 0 |
| Native complete | 563200 | 0.0001561470 | 1 |
| Native with original return slice | 562688 | 0.0001576506 | 1 |

Original positive PCM matches the retained original waveform at cosine
0.9999999999998493, relative L2 5.572414e-7 and norm error9.646689e-8, passing
all unchanged strict gates. Its ASR is exactly "We are committed to making it
secure." Both native-code variants transcribe as "You", WER1. The original DAC
does not restore speech from these native codes. Native-complete PCM through
the original versus native DAC has cosine0.999964985, relative L2 0.008929458
and norm error0.003080364; native-code RMS is0.00212347 times the positive RMS.
This localizes the missing speech to generated code content in this fixture,
without claiming that every codec implementation or precision choice is exact.

An independent auditor rehashes all77 raw worker payloads, both checkpoint
files and the input bundle; recomputes PCM/magnitude metrics and WER; verifies
float WAV/array equality and actual ffmpeg resampling against all retained ASR
inputs; checks source identity, recipe and raw segments.
The first965 aligned code frames match original exactly (11.203628s), and
99.9996567% of original waveform energy follows that prefix. A matched mostly
silent prefix therefore does not establish successful spoken output.

The repaired production protocol is also compiled/replayed on both real retained
CLI/C-ABI sampler captures. All1115 feedback rows,1114 returned rows and1099
aligned codec frames match actual original methods exactly. This establishes
protocol correctness on those captures, without a new neural generation or
speech acceptance. The runtime candidate remains pending applicable CI.

[Complete117-payload diagnosis and retained-protocol proof](https://huggingface.co/datasets/cstr/crispasr-regression-fixtures/resolve/668196e7450b1ce35bded6355ec58464235deb1e/dia-20261015-full-codec/complete-original-dac-diagnosis-v1.zip)
is23,699,070 bytes, SHA256
`9ff7dda447d1d8cbd75dcbbbb2316db70d6ad84ba9ad0cc12fed88a74db2862c`,
fully anonymously downloaded and rehashed. It includes original/native complete
PCM and WAVs, ASR inputs/results, source, raw logs, method/environment setup,
independent audit, actual protocol executable and all retained arrays. The
[pinned upstream Dia license](https://huggingface.co/datasets/cstr/crispasr-regression-fixtures/resolve/949b03fd276ecf968ba589e336c2788f80a4e445/dia-20261015-full-codec/Dia-LICENSE)
accompanies the proof alongside the included DAC/CrispASR notices.

## Public preparation proof

[Complete protocol/preparation proof](https://huggingface.co/datasets/cstr/crispasr-regression-fixtures/resolve/ea07987236fd47882f62c348dad2f80d764229b1/dia-20261015-full-codec/complete-protocol-preparation-v1.zip):14 outer payloads,5,261,390 bytes,
SHA256 `63baa2f12bc1da0775e511d274fbadbd0d2aa0cf059ed2299dcd0817e3bb3b9d`, fully anonymously downloaded/rehashed. It includes the complete
42-payload [input bundle](https://huggingface.co/datasets/cstr/crispasr-regression-fixtures/resolve/e22cb7451fa0b073e3b497273266338913243831/dia-20261015-full-codec/retained-inputs-v1.zip), actual original-method replay source/results,
all native reconstruction inputs/adapter/source, positive waveform/code controls,
parent hashes and notices. The input bundle is5,220,208 bytes, SHA256
`b711e33d4bb60f6c421397a3ca2eaf8600facd196a23c03e2bce20bafbee6126`.
[Machine receipt](dia-full-codec-preparation-2026-10-15.json).
