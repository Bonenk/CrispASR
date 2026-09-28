#!/usr/bin/env bash
# Push the OmniVoice CFG CUDA A/B kernel under ${KAGGLE_ACCOUNT}.
set -e
# Kaggle CLI auth: ~/.kaggle/access_token (never commit a token; see tests/test_no_secrets.py)
cd "$(dirname "$0")"
python -m kaggle kernels push -p .
