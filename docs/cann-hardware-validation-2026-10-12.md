# Ascend operator validation

`tools/validate_backend_ops.py` runs the existing ggml correctness tests against
CPU reference math, preserving upstream tolerances. It requires the requested
backend to complete and each requested operator to run nonzero passing cases.
Unsupported cases remain in the complete logs; they are not counted as passes.
Receipts include binary/script-checkout provenance, log hashes and actual counts.

[Hosted ARM CPU validation](https://github.com/CrispStrobe/CrispASR/actions/runs/38281476089)
executes one real passing ADD case and three real upstream false-success cases:
missing backend, empty parameter selection and unknown operator. The upstream
binary exits zero for all three, while the wrapper rejects each. See
[the complete receipt](cann-hardware-wrapper-2026-10-12.json).

For the contributor hardware experiment, use a CANN-enabled `test-backend-ops`
binary built against this repository's ggml revision
`c36dab89b662838f0f5d4826c399198c0b90bbfc`. The wrapper requires only Python's
standard library. Start with the same bounded case exercised on CPU:

```bash
python3 tools/validate_backend_ops.py \
  --binary /path/to/CANN/test-backend-ops --backend CANN0 --ops ADD \
  --params '^type=f32,ne=\[1,1,1,1\],nr=\[32,1,1,1\],nf=1,perm1=0,src_overlap=0$' \
  --out cann-add-proof
```

Then request the wider existing operator suite, in a fresh output directory:

```bash
python3 tools/validate_backend_ops.py \
  --binary /path/to/CANN/test-backend-ops --backend CANN0 \
  --ops ADD,RMS_NORM,SOFT_MAX,ROPE,MUL_MAT --out cann-operators-proof
```

Return the complete output directories, device/chip model, SDK version and
whether ACL graphs were enabled in the binary. Operation failures/timeouts
retain logs and receipts and produce a failing process exit. The command
interface and empty-test guards are verified on CPU; the contributor returned physical Ascend 310P3 execution on 2026-10-13.
The broad suite fails; see the verified results below. Passing operator tests alone
cannot establish full-model inference, quantized-model speech quality or speed.


## Physical Ascend 310P3 results — 2026-10-13

The contributor returned [the complete archive](https://github.com/user-attachments/files/33372242/crispasr-cann-310p3-validation-2026-10-13.tar.gz)
in [PR #544 comment 6140622935](https://github.com/CrispStrobe/CrispASR/pull/544#issuecomment-6140622935).
Its SHA-256 matches the reported
`83cc620570c9345153545c02418a6171f5b5914d5dc43f743dbc4b12a162feb7`.
All 17 member hashes are retained and all five receipt assessments were
recomputed against their complete logs with the unchanged wrapper. The
[verified receipt](cann-310p3-hardware-2026-10-13.json) preserves failures too.

Hardware: two Ascend 310P3 devices installed, `CANN0` tested on ARM64;
SDK 9.1.0, ACL graphs disabled, ggml
`c36dab89b662838f0f5d4826c399198c0b90bbfc`.

| Requested control | Physical result |
|---|---|
| Bounded F32 ADD | 1/1 pass |
| All F32 ADD cases | 56/56 pass |
| F32 weights / F32 input MUL_MAT | 272/272 pass |
| F16 weights / F32 input MUL_MAT | 282/282 pass |
| RMS_NORM | 51/51 pass |
| SOFT_MAX | 107/107 pass; 107 unsupported cases retained |
| ROPE | 109/109 pass; 361 unsupported cases retained |
| Broad ADD | SIGABRT after 92 observed passing cases; no completed suite |
| Broad MUL_MAT | SIGABRT after 257 observed passing cases; 1,488 unsupported cases retained |

Both aborts explicitly report unsupported `DT_BFLOAT16`: `aclnnAdd` rejects
its input and `aclnnMm` rejects `matA`. In the pinned source, ADD's capability
check accepts BF16 unconditionally; MUL_MAT excludes BF16 weights on 310P but
misses BF16 activations. A scoped capability repair is being prepared and must
be compiled and rerun on the device. Production ggml remains pinned to the
reported version. Partial case counts and successful restricted suites do not
turn the broad suite into a pass. Full-model diff/ASR, quantized quality and
performance remain unvalidated on this NPU.


## BF16 capability candidate: physical rerun

Diagnostic CrispASR source
`bfb4943ac1e4488e377257a47163a0edcea7d6cd` pins shared ggml candidate
`21acf3162beeb4e51260f0f86deda80995423712`. Only 310P capability routing
changes: BF16 ADD input/output and BF16 MUL_MAT activation inputs are rejected
so the scheduler can use a supported backend. The 910B predicate is unchanged.
Thirteen isolated routing cases pass under each chip configuration; the old
310P predicate fails six. This does not prove a running model's CPU fallback.
[SDK matrix 38365667723](https://github.com/CrispStrobe/CrispASR/actions/runs/38365667723)
is submitted; compilation and the physical candidate rerun remain pending.
See [the candidate receipt](cann-310p-bf16-candidate-2026-10-13.json).

Use a fresh directory and the same activated CANN SDK 9.1.0 environment as the
reported run. The following builds only the upstream operator test binary:

```bash
git clone --no-checkout https://github.com/CrispStrobe/CrispASR.git CrispASR-cann-bf16-proof
cd CrispASR-cann-bf16-proof
git checkout --detach bfb4943ac1e4488e377257a47163a0edcea7d6cd
git submodule update --init --recursive
python3 tests/test_cann_bf16_routing.py -v
cmake -S ggml -B build-cann-ops \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DGGML_CANN=ON -DCANN_INSTALL_DIR="$ASCEND_TOOLKIT_HOME" \
  -DSOC_TYPE=Ascend310P -DUSE_ACL_GRAPH=OFF \
  -DGGML_BUILD_TESTS=ON -DGGML_BUILD_EXAMPLES=OFF
cmake --build build-cann-ops --target test-backend-ops -j4
python3 tools/validate_backend_ops.py \
  --binary build-cann-ops/bin/test-backend-ops --backend CANN0 \
  --ops ADD,RMS_NORM,SOFT_MAX,ROPE,MUL_MAT --out cann-bf16-candidate-proof
```

Return the complete fresh proof directory and configure/build stdout/stderr,
`build-cann-ops/CMakeCache.txt`, `build-cann-ops/compile_commands.json`, device
information and SDK versions. This tests the broad suite that previously
aborted, retaining unsupported BF16 cases explicitly. Keep the original failed
archive; do not overwrite it or filter BF16 out through `--params` to obtain a
pass. A passing candidate must finish both broad operators under the unchanged
upstream tolerances. Full-model diff, decoded ASR and scheduler fallback still
need separate validation before any NPU model or speed claim.
