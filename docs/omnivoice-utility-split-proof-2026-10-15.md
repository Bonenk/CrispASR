# OmniVoice standalone utility proof and pending integration

The utility header, standalone probe and original-function oracle have been
split from [draft PR521](https://github.com/CrispStrobe/CrispASR/pull/521).
First split commit `a140b3d7` preserves the original author and exact three
source files from original commit `a1ae00df`, as present at PR head `381f3a3a`.
Candidate `69d9eac4b10e1cc857e4e2e90b1101fd40983b6b` adds full evidence retention
and upstream license attribution. Its tested source bytes are unchanged from
`a479139977210766cf93860f77fa8fdf99118d38`.

No production speech funnel includes the helper. Reference preprocessing,
output processing, generation, params/setter ABI and defaults are unchanged.
The full PR remains a draft: both original and native fixed clone controls
fail the unchanged 20% WER gate at 2/9 words. Upstream issue256 attributes the
reported onset to an abruptly truncated reference; utility parity does not
prove that the onset or cloned speech has been repaired.

## Complete local original/native replay

On isolated Python 3.12, NumPy 1.26.4 and pydub 0.25.1, the inherited oracle
passes 738 waveform calls and 18 multilingual text cases. An independent
auditor then loads the actual compiled native probe and re-executes the pinned
original Python functions against every stored input:

| Check | Result |
| --- | --- |
| PCM16 silence-cleanup calls | 369, byte-exact |
| Fade/pad calls | 369, maximum absolute error 0 |
| Text punctuation calls | 18, exact |
| Full stored arrays checked | 1,601 |
| Rejecting witnesses | Unprocessed input, doubled amplitude, reversed samples |

The fade gate remains equal lengths and maximum absolute error at most 2e-7;
it was not relaxed. Original upstream audio/text source hashes are pinned at
`08be0b4ccbac3e13e374e86fbfead4b4cac343e2`. Every raw array, source and actual
probe hash is checked. The native helper/probe are byte-identical to the PR's
previously audited files. Compiler command/version, actual ELF and linked-library
hashes are retained. Actual probe SHA256:
`123675aa7f009292c1eac38eee9c4f545c4d8025bbc1c5c1c2c05c796935f583`.

## Public proof and remaining gates

The [complete immutable proof](https://huggingface.co/datasets/cstr/crispasr-regression-fixtures/resolve/1ac18a4e22a6b23ca7769849e68608cba0eae26a/omnivoice/utility-split-20261015/complete-utility-split-v1.zip) retains
1620 payloads and 37100271 compressed bytes, SHA256
`c825004f1e04378d7b7fa99dc7b98bafeabc1d06fcf1e18b9ec42865052f2d1f`. An anonymous full download matches that hash. It contains all
inputs and both complete output arrays, original/native source, actual probe,
compiler/link receipts, auditor, manifests and license notices. No model weights
or private speech are required for this utility diagnostic.

[Hosted utility run38739872601](https://github.com/CrispStrobe/CrispASR/actions/runs/38739872601)
and [lint38739877120](https://github.com/CrispStrobe/CrispASR/actions/runs/38739877120)
are submitted on the exact tested source and remain queued. Do not call those
checks green or integrate the split before applicable terminal checks pass.
Afterward, rebase/ff only this scope, verify tested bytes and preserve original
authorship. No full-PR, cloned-speech, click-removal, GPU or default promotion.

The [candidate method](https://github.com/CrispStrobe/CrispASR/blob/69d9eac4b10e1cc857e4e2e90b1101fd40983b6b/docs/omnivoice-utility-split-2026-10-15.md)
records reproduction details. All large arrays, isolated dependencies and proof
are under `/mnt/storage/crispasr/omnivoice-utility-split-20261015`.
