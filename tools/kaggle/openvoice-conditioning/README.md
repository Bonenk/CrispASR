# OpenVoice clip conditioning and shared-noise proof

Prepared follow-up to `openvoice-bake`; submission was rejected because both
batch-GPU slots are occupied by queued jobs. No conditioning run was created. Save the previous
kernel's terminal log/output before replacing or superseding a live job, and
respect the two-GPU-session cap. The bootstrap pins the shared worker by commit
and SHA256; it enables conditioning mode before loading it. Only the bootstrap
is uploaded by Kaggle; the shared worker is fetched by immutable URL.

Native source6be9fda36 supplies the exact original posterior-noise replay and
explicit equal-weight clip averaging APIs. Three cases include tau0 and0.3,
27/28 strict stages, uninstrumented C API/cache output, clip-mean embedding and
waveform controls, independent ASR/speaker similarity and retained model/input
hashes. No ggml/CUDA/full-SDK build or Torch reinstall. Actual VAD selection and
mixed-rate resampler equivalence remain separate scope.

See [OpenVoice baking](../../../docs/openvoice2-baking.md) for gates, limitations
and source identities. A successful submission/build is not model acceptance.
