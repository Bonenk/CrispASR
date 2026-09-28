#!/bin/bash
# Push under ${KAGGLE_ACCOUNT} (CrispASR kernel convention).
# Kaggle CLI auth: ~/.kaggle/access_token (never commit a token; see tests/test_no_secrets.py)
cd "$(dirname "$0")" && cp ../kaggle_harness.py . && kaggle kernels push -p .
