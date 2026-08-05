#!/bin/bash
# Push under ${KAGGLE_ACCOUNT} (CrispASR kernel convention).
# ⚠ If a run of this kernel is already live: `yes | kaggle kernels delete
# ${KAGGLE_ACCOUNT}/crispasr-madlad-quants` FIRST — a re-push STACKS a second GPU session
# and both keep burning quota (kaggle_usage gotcha #25).
export KAGGLE_API_TOKEN=KGAT_REVOKED_REMOVED
cd "$(dirname "$0")" && cp ../kaggle_harness.py . && kaggle kernels push -p .
