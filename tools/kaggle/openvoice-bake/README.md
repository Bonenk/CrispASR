# OpenVoice2 CUDA bake

See [the recipe and acceptance gates](../../../docs/openvoice2-baking.md).
`openvoice_bake.py` pins native source, original Python source and an audited
CUDA runtime archive. It compiles only changed model/driver translation units
against identical prebuilt ggml headers/libraries, then runs genuine CUDA
vocoder parity and separate speech-quality controls. It never builds ggml/CUDA
or reinstalls Torch. Only the script is uploaded by Kaggle.

Use `tools/kaggle/kpush.py` with private environment credentials after quota and
concurrency checks. Save terminal logs before any version push. Inspect actual
hardware, source pins, all required stages and quality receipts; a completed
kernel is not acceptance by itself. No automatic hardware retries.
