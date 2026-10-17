# Dia attention arithmetic isolation

This is a bounded CPU diagnostic. It loads no full neural model and changes no
production inference, model pin, filtering recipe or default.

`../dia_attention_arithmetic.py` builds `bridge.cpp` against the exact retained
c36dab89 headers/libraries. The graph follows the pinned runtime8d0ad886 explicit
attention path: consecutive GQA head expansion, scale=1 QK dot products, softmax,
and probability-weighted values. The baseline must reproduce independently
captured hosted native attention byte-for-byte. The original-module Q/K/V replay
must reproduce retained actual Python SDPA output exactly before interpretation.

Small fixtures establish separate facts:

- Zero queries give exactly uniform probabilities and analytically known values;
  distinct batches/KV heads detect incorrect grouping or batch order.
- Nonuniform scores compare scale=1 probabilities and output against float64.
  Actual wrong-scale, reversed-head and doubled-value input executions reject.
- Four cancellation terms measure finite-precision reduction behavior against
  the exact answer. This measures a numerical difference, not a tolerance pass.

For real owned Q/K/V, each case retains native scores, probabilities and weighted
values, plus fresh repeats with one/four threads. Two injection modes replace
scores or probabilities with the explicit Python shadow arrays to isolate the
softmax and weighted-value reductions. Used tensors must equal their requested
arrays exactly. The unchanged original `Attention.forward` SDPA call is executed
separately; the explicit shadow is never described as an internal fused-kernel
capture. For these recorded CPU fixtures the shadow and actual SDPA match exactly;
that must not be assumed on other shapes, builds or devices.

```sh
python tools/ci-heavy/dia_attention_arithmetic.py \
  --query-proof /path/to/query-replay-v2 \
  --native /path/to/previous-hosted-key-intervention/out \
  --native-query-proof /path/to/native-query-producer-v1 \
  --output ${CRISPASR_STORAGE}/crispasr/fresh-attention-arithmetic
```

Use the pinned Python3.12/Torch2.6.0+cpu/NumPy1.26.4 reference environment.
Every output directory must be fresh. Full methods, exact ELF/library hashes,
raw inputs/outputs, logs and source snippets are retained. Initial run-v1 is
preserved; run-v2 extends it with actual negative executions, cancellation and
float64 comparisons. The original cache capacity3072/strides are retained.

The independent auditor imports no producer code, checks every payload/hash,
reexecutes actual original SDPA and the shadow, recalculates all metric rows,
checks every thread/repeat/control, and freshly invokes every actual ELF mode:

```sh
python tools/ci-heavy/dia-attention-arithmetic/audit.py \
  --proof ${CRISPASR_STORAGE}/crispasr/fresh-attention-arithmetic \
  --output ${CRISPASR_STORAGE}/crispasr/fresh-independent-attention-audit.json
python tools/ci-heavy/dia-attention-arithmetic/reduce_worst.py \
  --proof ${CRISPASR_STORAGE}/crispasr/fresh-attention-arithmetic \
  --output ${CRISPASR_STORAGE}/crispasr/fresh-single-dot-reduction
```

The reduction helper selects the largest disagreement with fixed original
probabilities, retains just its979 scalar probability/value pairs and runs the
same native weighted-value operator with one output scalar. That native scalar
must reproduce the full attention component exactly. It compares original
matrix, isolated Torch dot/matmul, native and float64-product `math.fsum` results.
Torch's scalar and original matrix dispatch can differ; no scalar result is
silently substituted for the actual original matrix result.

`../dia_cfg_margin.py` independently explains the recorded token46/904 margin
using four incoming logits. It verifies original/native ranks and exact
conditional/unconditional error contributions, checking float32 against
float64 CFG without sampling or inference:

```sh
python tools/ci-heavy/dia_cfg_margin.py \
  --native /path/to/previous-hosted-key-intervention/out \
  --output ${CRISPASR_STORAGE}/crispasr/fresh-cfg-margin
```

These probes establish component arithmetic. They do not assign additive causal
percentages, establish full-model query provenance, restore the disputed token
support, or accept speech/CUDA/Q4/performance. The separate full-decoder joint
Q/K factorial remains necessary; do not restart it for these standalone tools.
