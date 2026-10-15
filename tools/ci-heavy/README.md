# tools/ci-heavy: heavy CPU jobs on GitHub Actions

Scripts here run on GitHub-hosted runners through `.github/workflows/heavy-cpu.yml`,
triggered by hand. Use this for CPU work that used to go to Kaggle: Python reference
dumps, parity diffs, ASR roundtrips, and download → convert → quantize → upload.

| Runner | CPU / RAM | Disk | Notes |
|---|---|---|---|
| `ubuntu-latest` (default) | 4 vCPU / 16 GB | ~14 GB free on `/`, plus the `/mnt` scratch disk; the workflow also deletes unused toolchains | fp32 references up to ~3B params |
| `ubuntu-22.04` | x86 CPU | as above | Pinned older Ubuntu image for compatibility checks |
| `ubuntu-24.04-arm` | 4 vCPU / 16 GB | as above, minus `/mnt` | aarch64 numerics |
| `macos-14` | 3 vCPU (M1) / 7 GB | ~14 GB | Metal |

Runners are free for public repos and a job can run up to 6 h.

**Keep on Kaggle** (`tools/kaggle/`) only what needs a real GPU: CUDA or Vulkan
speed and parity on hardware, and PyTorch on CUDA. Kaggle allows one account per
person, with GPU sessions used for GPU work.

## Running a job

```bash
gh workflow run heavy-cpu.yml -f script=tools/ci-heavy/<name>.py \
    -f pip="torch torchvision transformers>=4.57 accelerate pillow"
gh run watch "$(gh run list -w heavy-cpu.yml -L 1 --json databaseId -q '.[0].databaseId')"
gh run download <run-id>           # the $HEAVY_OUT artifact
```

`dia_key_history.py` and [its native producer](dia-key-history/README.md)
reconstruct original layer0 keys over979 retained shared steps with only the
required F32 coefficients. Original cache reconstruction gates native analysis;
full stage arrays, batched/one-step distinctions and rejecting controls remain
retained. This component diagnosis does not establish model or speech parity.

## Script contract

`dia_full_codec_crossdecode.py` replays retained native stopping/BOS/delay
records, verifies every feedback row and complete code/PCM frame count, then
runs the actual pinned original DAC on the native codes and original-code
positive control. It also replays the original Python generation protocol on
those retained predictions to audit its returned slice, without neural-model
inference. Full PCM/norm/RMS and fixed original Whisper ASR results survive;
positive PCM/zero-WER controls are required, while native speech remains a
separate unaccepted model gate. No production graph/default changes.
Run with `-f pip="numpy==1.26.4 torch==2.6.0 torchaudio==2.6.0 torchvision==0.21.0 pydantic==2.11.3 descript-audio-codec==1.0.0 descript-audiotools==0.7.2 openai-whisper==20250625"`.
Use `--self-test` for scale/order/shape/nonfinite/zero-norm rejection controls.

`cann_model_fixture.py` prepares a CPU-only Moonshine Tiny fixture for future
CANN model validation. It freshly converts immutable original weights, checks
every converted coefficient against the physically F32 regression model, uses the
original model-card generation recipe and `dump_reference.py`, and compares
native encoder arrays with cosine, norm and relative-error gates. It also runs
`crispasr-diff`, CLI and repeated auto/explicit Python Session dispatch. Full
logs, reference, model/tokenizer, raw arrays and receipts survive failures.
Hosted run38555394740 and its independently audited artifact pass; see
[full CPU fixture proof](../../docs/moonshine-hosted-cpu-fixture-2026-10-14.md).
No CANN hardware procedure is published by this worker.
Run with `-f pip="numpy==1.26.4 torch==2.6.0 transformers==4.57.6 huggingface_hub==0.36.0 safetensors==0.5.3 pillow librosa==0.11.0 gguf"`.

`dia_dac_converter_validate.py` checks the isolated Dia DAC conversion tool on
main without needing the diagnostic runtime branch. It runs seven tiny actual
original-module/CLI contracts, then converts the pinned published template using
the original DAC checkpoint. The GGUF SHA/size and all 134 physical audit rows
must match the prior standalone candidate that passed startup/full-history
numerical comparisons. `dia-dac-export.yml` calls this worker automatically on
converter/helper/test changes. Run manually with
`-f pip="numpy torch==2.6.0 torchaudio==2.6.0 torchvision==0.21.0 pydantic safetensors huggingface_hub descript-audio-codec==1.0.0 gguf"`.
This is conversion integration evidence, not new free-speech or GPU acceptance.

- **`$HEAVY_OUT`**: write results here. The whole directory becomes the run's artifact, kept for 14 days.
- **`$HEAVY_SCRATCH`**: large scratch space for models, builds and caches. Not uploaded. `HF_HOME` points inside it.
- **`$HF_TOKEN`**: the repository secret (fine-grained, CI-scoped); may be empty.
- **`$HEAVY_OUT/summary.md`**: if present, appended to the run's summary page.
- **Exit code**: non-zero when the check fails, so the run shows red. A readout that prints the same thing whether it passed or failed isn't a check.
- **Paths**: the workflow only accepts `.py` / `.sh` files under `tools/ci-heavy/`.
- **Porting a Kaggle kernel**: replace `/kaggle/working` with `$HEAVY_OUT` and `/kaggle/input/...` with a download into `$HEAVY_SCRATCH`. Drop the GPU and ccache datasets.

`miotts_sample_rate.py` checks the pinned MioCodec-v2 model through the native
live test, session rate getter, CLI WAV header and both TTS→ASR roundtrips.
It also checks 24 kHz and missing-key metadata dispatch using temporary copies;
those copies are not claimed to be legacy codec speech models. Run with
`-f pip="numpy gguf huggingface_hub"` on Linux x86 or ARM.

`mimo_cabi_device_params.py` checks four device/verbosity combinations and four
failed-open default restores through the actual shared library and an init hook.
Removing the two forwarding assignments must fail; restoring them must pass.
Run on Linux x86 with no extra pip packages. No model/GPU inference claim.

`data2vec_reference_parity.py` verifies the existing independent Python CTC
transcript using pinned logits/vocabulary/sample hashes, rejects four broken
provenance/target controls, then runs the actual Q4 CLI and F16 stage regression.
Zero-WER upstream parity remains hard; the original model's 4.55% human WER
is reported separately. Run with `-f pip="numpy gguf huggingface_hub"`.

`cann_model_strict.py` reuses the full Moonshine CPU producer, then runs the
opt-in exact-device path through diff, CLI and auto/explicit C ABI Sessions.
It records allocated encoder backend/op counts and successful computation,
and rejects scaled/nonfinite/wrong-shape/missing-stage reference archives,
changed input PCM, missing CANN and invalid/unavailable device ordinals.
Use the same dependency pins as `cann_model_fixture.py`. Hosted run38575701613 passes all13 controls with independently audited raw
archives; [acceptance proof](../../docs/moonshine-strict-device-validation-2026-10-14.md).
These CPU controls cannot establish physical NPU work or ACL capture/replay.

`cann_model_validator.py` is the full portable model-runner control job. It
builds the original/native CPU fixture, then exercises `validate_cann_model.py`
with the immutable accepted archive through diff, repeated CLI and automatic/
explicit Sessions. It must reject a CANN request on the CPU build after CPU
controls, and checks the real public encoder GPU=true forwarding independently.
Use the same dependency pins as `cann_model_fixture.py`. Hosted run38602252235 passes the full interface;
[complete acceptance and binary-snapshot proof](../../docs/cann-model-validator-preparation-2026-10-14.md).
Actual physical CANN model acceptance remains pending.
