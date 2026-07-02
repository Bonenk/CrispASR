#!/bin/bash
# Push under ${KAGGLE_ACCOUNT} (CrispASR kernel convention).
export KAGGLE_API_TOKEN=KGAT_REVOKED_REMOVED
cd "$(dirname "$0")" && cp ../kaggle_harness.py . && kaggle kernels push -p .
