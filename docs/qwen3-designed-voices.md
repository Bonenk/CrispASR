# Designed Qwen Base voices: provenance and redistribution

The production route is fresh Qwen VoiceDesign output, encoded by the matching
Base model, then packaged as GGUF. The old compatibility reference is excluded
from this route. Candidate generation does not automatically promote an asset.

The bake script is [qwen3_default_voice_bake.py](../tools/ci-heavy/qwen3_default_voice_bake.py).
It runs on the heavy CPU GitHub workflow, with one original F32 model per process.
It uses two original generic descriptions, an original neutral transcript and
neutral selectors `default` and `clear`. It supplies no recording, named-person
target, preset speaker ID or downloaded prompt. Both references carry an
AI-generated RIFF comment and an explicit filename. No claim is made that these
voices cannot resemble a person.

## License basis

The publisher declares these exact models Apache-2.0:

| Purpose | Model | Revision |
|---|---|---|
| Reference generation | [Qwen3-TTS-12Hz-1.7B-VoiceDesign](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign) | `5ecdb67327fd37bb2e042aab12ff7391903235d3` |
| 1.7B encoding | [Qwen3-TTS-12Hz-1.7B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base) | `fd4b254389122332181a7c3db7f27e918eec64e3` |
| 0.6B encoding | [Qwen3-TTS-12Hz-0.6B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base) | `5d83992436eae1d760afd27aff78a71d676296fc` |
| Shared tokenizer declaration | [Qwen3-TTS-Tokenizer-12Hz](https://huggingface.co/Qwen/Qwen3-TTS-Tokenizer-12Hz) | `7dd38ad4e9bad454aae9cd937d0cd577604fe229` |

The actual embedded tokenizer/config/weight files are hashed within each source
snapshot; the separate tokenizer declaration does not substitute for those hashes.
The code is the hash-pinned `qwen-tts==0.1.1` wheel. Executed package Python files
must match that wheel exactly. The bundle preserves its license and package
metadata, and the full upstream Apache license at source commit
`022e286b98fbec7e1e916cb940cdf532cd9f488e`.

For each model, retain the exact-revision publisher API response and model card,
and any supplied LICENSE/NOTICE/COPYING files. A model-card license declaration
must be described as such, not misrepresented as a separate license file or a
performer release. Missing notice files are recorded explicitly.

Voice assets are offered under Apache-2.0 to the extent we hold licensable rights.
The bundled `recipe.py` remains MIT-licensed CrispASR code. Include the verbatim
repository MIT license/copyright notices as `LICENSE-RECIPE.txt`, alongside the
Apache `LICENSE` and attribution `NOTICE`; the GGUF embeds both license texts.
Our distributions retain the full provenance bundle and record modifications
prominently. Provenance retention is recommended to recipients, not an additional
condition on either open-source license.
Do not assert exclusive copyright in machine-generated material where none
exists, or imply endorsement by Qwen/Alibaba. The resulting bundle includes no
model weights or third-party recordings.

The [Apache license](https://www.apache.org/licenses/LICENSE-2.0) grants rights
from its contributors, requires specified notices on redistribution, and includes
warranty limitations. It is not a universal clearance of third-party rights.
Generated output does not itself prove absence of identifiable likeness; see the
[US Copyright Office digital replicas report](https://www.copyright.gov/ai/Copyright-and-Artificial-Intelligence-Part-1-Digital-Replicas-Report.pdf).
We cannot independently verify the upstream training corpus or provide a worldwide
non-infringement guarantee. This provenance audit is not a legal opinion.

## Candidate and promotion evidence

Retain these together with each candidate:

- Full recipe, original descriptions/transcript, seeds, generation settings,
  dependency versions, source model revisions and source file hashes.
- Saved reference WAVs, exact speaker/code arrays, model-specific GGUFs and
  checksums. Prompts must be extracted from the saved WAVs, not a different
  pre-quantized waveform.
- Full LICENSE, NOTICE, publisher declarations and package licensing evidence.
- Reference transcription checks and native CPU/F16/Q8 plus real CUDA speech
  results, including both selectors and model-specific default discovery.
- Listening and resemblance review status, with any recognized-person concern
  recorded and resolved before that candidate is promoted.

Successful extraction/readback is candidate evidence only. Native speech
acceptance is required before wiring defaults. Human listening feedback is
recorded separately and cannot establish identity or upstream training rights.
Missing or contradictory license terms, unlicensed input recordings, or specific
unresolved likeness concerns block the affected candidate. No automatic publisher
license tag or synthetic-audio label can override those gates.

## Current execution

The [native runtime build and existing CPU controls](https://github.com/CrispStrobe/CrispASR/actions/runs/39935555861)
passed at source `7411898f8d62add2d8e03ce1f7f26233e17875c3`. That result does
not validate the fresh designed voices.

The fresh bake uses the updated recipe with both license texts and explicit
unknown speaker identity. Earlier queued compatibility, pre-embedded-notice and
pre-MIT-notice bakes were canceled before execution; they are not release inputs.
In particular, run39935552877 was canceled to complete the recipe licensing scope.

After both inputs succeed, `qwen3_designed_voice_cpu.py` accepts their exact run
IDs and separate producer/runtime source revisions, verifies the native archive and every bundled file, compares the
candidate recipe and MIT notice to the producer git blobs, and runs the independent validator.
The wrapper checks through the normal CLI and Python Session/C ABI. References
must transcribe exactly; “Hello world” must transcribe exactly; other test speech
allows WER at most 0.1. Both model sizes and both 1.7B F16/Q8 variants are tested.
Both model-specific defaults must coexist with the old incompatible generic pack;
renamed-model discovery and explicit incompatible-pack rejection are included.

A one-voice selection fixture lets the existing CLI consume `clear` without a
new CLI feature: speaker/code arrays are copied unchanged, read back exactly,
and retain embedded license, attribution and provenance. It is a test fixture,
not a claim that unrelated model embeddings can be converted by changing shape.
Session tests use normal marked `synthesize`, and CLI tests retain normal marking
while omitting the spoken prefix for scoring. Actual CUDA uses the same validator,
requires per-child GPU residency and records native backend logs.

The licensing/array auditor has also been exercised with **schema-only synthetic
fixtures**. A matching fixture passes; a changed full license, unknown publisher
license, mismatched saved embedding and missing MIT recipe notice are each
rejected even when manifest checksums are refreshed. This tests the auditor, not the original model or speech.

## Reference qualification and default selection

[Fresh bake39938969713](https://github.com/CrispStrobe/CrispASR/actions/runs/39938969713)
at `3ef249810a3036ad1658e64461687c63ce0b6173` completed successfully. Its actual
license/array audit passes. An independent original Whisper base.en diagnostic
(weights SHA256 `25a8566e1d0c1e2231d1c762132cd20e0f96a85d16145c3a00adf5d1ac670ead`,
verified against the official downloader) transcribed only the first sentence
of the raw masculine `default` reference, WER8/15. It is rejected. The `clear`
reference transcribed the entire original text exactly, WER0.

The release candidates select only `clear`, named `default` inside each
model-sized pack. Speaker and codec arrays are unchanged from the original
Base encoder outputs. The raw two-voice pack is not installed or promoted.
Metadata records the selected source voice, its description, original producer
manifest hash and the curation recipe hash. License/attribution texts are kept
inside the GGUF and as sidecars; the MIT curation script accompanies the result.

The hosted matrix must recheck the selected reference at WER0, then test the
curated defaults through real CLI/C ABI CPU and CUDA paths. Packaging success
and this lightweight diagnostic do not replace those native acceptance gates.

## Published designed defaults

[Release9ab96c5](https://huggingface.co/cstr/qwen3-tts-voices-GGUF/tree/9ab96c5ba3022bd2ed46306ded0489a24096f8b5)
contains the tested clear-as-default packs, readable Apache/MIT notices, source
declarations, reference WAV, original arrays and reproduction recipes. All37
files passed anonymous full-byte readback; the existing legacy file is unchanged.
The model card license was independently read back as `apache-2.0`, with the
separate MIT code scope explicit. Human listening feedback is still pending;
speaker identity remains unverified and no upstream-training-rights warranty
is represented. This status is preserved, not counted as an approval.

CPU39942586420 and actual CUDA `issue555-designed-voices-cuda-v1` each score
all13 reference/speech controls WER0. Both default files are byte-identical across
CPU/CUDA; native CUDA backend logs and per-child residency are retained.
The auto-download entries select only their own 1024/2048-d pack and include
readable license/attribution plus the complete provenance ZIP.
