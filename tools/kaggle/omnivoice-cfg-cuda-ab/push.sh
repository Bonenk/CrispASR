#!/usr/bin/env bash
# Push the OmniVoice CFG CUDA A/B kernel under ${KAGGLE_ACCOUNT}.
set -e
export KAGGLE_API_TOKEN=KGAT_REVOKED_REMOVED  # ${KAGGLE_ACCOUNT}
cd "$(dirname "$0")"
python -m kaggle kernels push -p .
