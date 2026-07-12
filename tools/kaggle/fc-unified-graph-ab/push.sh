#!/bin/bash
# Push under ${KAGGLE_ACCOUNT} (${KAGGLE_ACCOUNT} is running the requant kernel).
export KAGGLE_API_TOKEN=KGAT_REVOKED_REMOVED
cd "$(dirname "$0")" && cp ../kaggle_harness.py . && kaggle kernels push -p .
