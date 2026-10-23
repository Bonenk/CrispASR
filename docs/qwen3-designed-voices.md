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

Distribute the pack and our licensable contributions under Apache-2.0, retaining
LICENSE, NOTICE and the provenance bundle. Record modifications prominently.
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

Successful extraction/readback is candidate evidence only. Native speech and
listening review remain separate recorded gates. Unresolved source rights or
recognized-person concerns block the affected candidate. No automatic publisher
license tag or synthetic-audio label can override those gates.
