# Model license support policy

Permissive licenses (including MIT, Apache-2.0 and BSD), CC-BY, and CC-BY-SA
are eligible for support with their attribution and redistribution obligations.
CC-BY-SA is a share-alike license, not a non-commercial license.

Ordinary CC-BY-NC and CC-BY-NC-SA models remain eligible **only with explicit
license acceptance**, clear non-commercial labeling, and no silent download via
`-m auto` or `--auto-download`. Managed download acceptance uses the exact license
tag through `--accept-license` or `CRISPASR_ACCEPT_LICENSE`; bindings expose the
same bundle metadata. These controls do not grant commercial rights. Local-file
loading is not a rights-verification system.

Research-only terms and additional use, output, revenue, user-count or hosted
service restrictions need a separate review before support or promotion. An
acceptance switch records the user's acknowledgment; it is not project approval
of a license. Existing custom-license entries are not automatically certified
by this policy. Do not add new restricted models under a permissive or ordinary
NC label. Review the main weights, tokenizer/codec, other companions and source
code separately, retaining immutable revisions and the actual terms. Unknown
terms are not evidence of permission.

## Follow-up audit, 2026-10-20

- Drama-3: the Fish Audio Community License includes revenue and hosted-service
  conditions. Keep issues545/546 outside implementation/release scope.
- Nyra: model terms restrict research/commercial use and the generated alignment
  outputs; issue456 requires separate review, rather than an ordinary NC switch.
- OmniVoice: the original-generation model revision
  `c5fdb5ccb189668d56333f77ba2629f4cd7535f4` already distinguishes Apache-2.0 code
  from CC-BY-NC weights. This is not merely a later change to today's card. The
  converted-model card's blanket Apache label needs reconciliation. Its named
  HiggsAudioV2 companion has Boson community terms in that same revision under
  `audio_tokenizer/LICENSE`; its text is independently downloaded and hashed.
  The converted-model card is corrected at HF revision
  `3425b20adc97e495b2710cb3a97aac577460f21f`, with anonymous readback verification
  and no weight changes. The registry uses the
  distinct `omnivoice-terms` gate for the bundle, not plain NC acceptance. PR521
  remains subject to both this review and its unchanged speech-quality gates.
- Supertonic-3: OpenRAIL permits some commercial uses but adds use restrictions.
  It now requires explicit `openrail-m` acceptance rather than merely printing a
  notice. Further support/promotion needs the separate review above.

Sources: [Fish terms](https://huggingface.co/fishaudio/drama-3/blob/main/LICENSE),
[Nyra terms](https://github.com/nyrahealth/nyra-forced-aligner#license),
[pinned OmniVoice card](https://huggingface.co/k2-fsa/OmniVoice/blob/c5fdb5ccb189668d56333f77ba2629f4cd7535f4/README.md),
[codec terms](https://huggingface.co/bosonai/higgs-audio-v2-tokenizer/blob/main/LICENSE),
[Supertonic](https://huggingface.co/Supertone/supertonic-3),
[CC-BY-SA](https://creativecommons.org/licenses/by-sa/4.0/).
